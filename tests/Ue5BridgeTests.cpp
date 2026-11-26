// The ABI 1.8 glue of kyty_gpu (src/hybrid/ue5HostBridge.h) driven with a fake host: the real extract trampolines (C callbacks
// through KytyGpuUe5Query), the checked guest reader on real mapped memory, the main-thread hop with a fake main queue, and the
// whole guest_ready flow (fingerprint read, extraction, store, search mark) over them. kyty_gpu.dylib itself cannot be loaded
// without a GPU (KytyGpuGetApi -> start_graphics needs the renderer), so this is the closest buildable equivalent: everything
// of the bridge that kyty_gpu installs at init runs here from the same header. Synthetic data only.
#include "graphics/host_gpu/renderer/pipeline/pipelineCodeStore.h"
#include "graphics/host_gpu/renderer/pipeline/ue5CodeWarm.h"
#include "hybrid/ue5HostBridge.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <random>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace Warm = Libs::Graphics::Ue5CodeWarm;
using Libs::Graphics::PipelineCodeStore;
using Libs::Graphics::PipelineList::Hash128;

static int g_failed = 0;
#define CHECK(expr)                                                                                  \
	do {                                                                                             \
		if (!(expr)) {                                                                               \
			std::printf("Ue5BridgeTests: FAILED: %s (line %d)\n", #expr, __LINE__);                  \
			++g_failed;                                                                              \
		}                                                                                            \
	} while (0)

namespace {

// ---- a fake host -------------------------------------------------------------------------------------------
struct FakeGroups {
	std::vector<std::vector<uint8_t>>  bytes;
	std::vector<std::vector<uint32_t>> freq;
	bool                               null_bytes_group = false; // an extra group with bytes == nullptr
	int                                code             = 0;
	uint64_t                           decoder          = 0x1000;
	std::atomic<int>                   calls {0};
} g_fake;

extern "C" int FakeExtract(void* user, const KytyGpuUe5Query* q, KytyGpuUe5Result* out) {
	(void)user;
	g_fake.calls++;
	if (q->struct_size != sizeof(KytyGpuUe5Query) || q->abi != 1 || q->decoder != g_fake.decoder) return -1;
	uint32_t done = 0;
	for (size_t i = 0; i < g_fake.bytes.size(); i++) {
		KytyGpuUe5Group g {};
		g.library      = 0;
		g.group        = static_cast<uint32_t>(i);
		g.shader_count = static_cast<uint32_t>(g_fake.freq[i].size());
		g.raw_bytes    = static_cast<uint32_t>(g_fake.bytes[i].size());
		g.frequencies  = g_fake.freq[i].data();
		g.bytes        = g_fake.bytes[i].data();
		const int stop = q->group(q->arg, &g);
		q->progress(q->arg, ++done, static_cast<uint32_t>(g_fake.bytes.size() + (g_fake.null_bytes_group ? 1 : 0)));
		if (stop) {
			out->groups_decoded = done;
			return -3;
		}
	}
	if (g_fake.null_bytes_group) {
		KytyGpuUe5Group g {};
		g.group = 99;
		g.raw_bytes = 4096; // a wrong host: a size without bytes must not crash the split
		(void)q->group(q->arg, &g);
		(void)q->group(q->arg, nullptr);
		q->progress(q->arg, ++done, static_cast<uint32_t>(g_fake.bytes.size() + 1));
	}
	out->groups_total   = static_cast<uint32_t>(g_fake.bytes.size());
	out->groups_decoded = done;
	out->raw_bytes      = 12345;
	out->ms             = 7;
	return g_fake.code;
}

// A fake main queue: `run_on_main_thread` queues; a "main thread" runs the jobs when told to.
struct FakeQueue {
	std::mutex                             m;
	std::deque<std::pair<void (*)(void*), void*>> jobs;
	int                                    result = 0;
	std::atomic<bool>                      stop {false};
	std::thread::id                        main_id;
	std::atomic<int>                       queued {0};
} g_queue;

extern "C" int FakeRunOnMain(void* user, void (*fn)(void*), void* arg, int wait) {
	(void)user;
	(void)wait;
	if (g_queue.result != 0) return g_queue.result;
	std::scoped_lock lock(g_queue.m);
	g_queue.jobs.emplace_back(fn, arg);
	g_queue.queued++;
	return 0;
}
void PumpOne() {
	std::pair<void (*)(void*), void*> job {nullptr, nullptr};
	{
		std::scoped_lock lock(g_queue.m);
		if (g_queue.jobs.empty()) return;
		job = g_queue.jobs.front();
		g_queue.jobs.pop_front();
	}
	job.first(job.second);
}

std::vector<uint8_t> RandomCode(uint32_t seed, size_t bytes) {
	std::mt19937         random(seed);
	std::vector<uint8_t> code(bytes);
	for (auto& b: code) b = static_cast<uint8_t>(random() | 0x01u);
	for (auto& b: code)
		if (b == '1') b = '2';
	return code;
}

// One group with one shader in the layout ue5Group.h documents.
std::vector<uint8_t> MakeGroup(const std::vector<uint8_t>& code) {
	std::vector<uint8_t> group(24, 0x81);
	const size_t         marker = group.size();
	group.resize(marker + 0x100 + code.size() + 48, 0);
	const uint8_t m[8] = {'1', '2', '3', '4', 0x18, 0, 0, 0};
	std::memcpy(group.data() + marker, m, 8);
	const uint32_t off = 0x100, size = static_cast<uint32_t>(code.size());
	std::memcpy(group.data() + marker - 0x14, &off, 4);
	std::memcpy(group.data() + marker - 0x0c, &size, 4);
	std::memcpy(group.data() + marker + 0x44, &size, 4);
	std::memcpy(group.data() + marker + 0x100, code.data(), code.size());
	return group;
}

} // namespace

int main() {
	KytyGpuHostCallbacks host {};
	host.struct_size               = sizeof(host);
	host.abi_version               = KYTY_GPU_ABI_VERSION;
	host.extract_ue5_shader_groups = FakeExtract;
	host.run_on_main_thread        = FakeRunOnMain;

	// Guest memory: a mapping standing in for the guest window (host = guest + bias).
	const size_t   page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
	void*          mem  = mmap(nullptr, page * 4, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
	CHECK(mem != MAP_FAILED);
	const uint64_t guest_lo = 0x40000000ull, guest_hi = guest_lo + page * 4;
	KytyGpuHostInfo info {};
	info.guest_lo   = guest_lo;
	info.guest_hi   = guest_hi;
	info.guest_bias = reinterpret_cast<uint64_t>(mem) - guest_lo;
	info.flags      = KYTY_GPU_HOST_MAIN_QUEUE;
	uint8_t decoder[64];
	for (size_t i = 0; i < 64; i++) decoder[i] = static_cast<uint8_t>(i * 5 + 3);
	std::memcpy(static_cast<uint8_t*>(mem) + 0x10, decoder, 64);
	const uint64_t decoder_guest = guest_lo + 0x10;
	g_queue.main_id = std::this_thread::get_id();

	std::vector<std::pair<int, std::string>> logs;
	bool                                     drains = true;
	Hybrid::Ue5Bridge::Env env;
	env.host              = &host;
	env.info              = &info;
	env.on_main_thread    = [] { return std::this_thread::get_id() == g_queue.main_id; };
	env.main_queue_drains = [&] { return drains; };
	env.log               = [&](int level, const std::string& text) { logs.emplace_back(level, text); };
	Hybrid::Ue5Bridge::Install(&env);
	CHECK(Warm::HostHasReader());

	// ---- the guest reader: bounds and an unmapped page ---------------------------------------------------------
	uint8_t buf[64];
	CHECK(Warm::GetSeam().read_guest(decoder_guest, buf) && std::memcmp(buf, decoder, 64) == 0);
	CHECK(Warm::FingerprintMatches(decoder_guest, Warm::Fnv1a64(decoder)));
	CHECK(!Warm::FingerprintMatches(decoder_guest, Warm::Fnv1a64(decoder) ^ 1));
	CHECK(!Warm::GetSeam().read_guest(guest_lo - 1, buf));            // below the window
	CHECK(!Warm::GetSeam().read_guest(guest_hi - 63, buf));           // ends past the window
	CHECK(!Warm::GetSeam().read_guest(0xFFFFFFFFFFFFFFF0ull, buf));   // wraps
	CHECK(!Warm::GetSeam().read_guest(0, buf));
	CHECK(Warm::GetSeam().read_guest(guest_hi - 64, buf));            // the last 64 bytes are fine
	CHECK(mprotect(static_cast<uint8_t*>(mem) + page, page, PROT_NONE) == 0);
	CHECK(!Warm::GetSeam().read_guest(guest_lo + page + 8, buf));     // inside the window, unreadable page
	{
		KytyGpuHostInfo empty {}; // a host that gave no window
		CHECK(!Hybrid::Ue5Bridge::ReadGuest(empty, 0x1000, buf));
	}

	// ---- the extract trampolines --------------------------------------------------------------------------------
	const auto code_a = RandomCode(1, 96), code_b = RandomCode(2, 160);
	g_fake.bytes      = {MakeGroup(code_a), MakeGroup(code_b)};
	g_fake.freq       = {{0x11}, {0x14}};
	int  seen = 0;
	uint32_t progress_total = 0;
	auto res = Warm::GetSeam().extract(
	    0x1000, 1,
	    [&](const Warm::GroupView& v) {
		    seen++;
		    CHECK(v.shader_count == 1 && v.frequencies.size() == 1 && !v.bytes.empty());
		    return false;
	    },
	    [&](uint32_t, uint32_t total) { progress_total = total; });
	CHECK(res.code == 0 && seen == 2 && res.groups_total == 2 && res.decoded == 2 && res.raw_bytes == 12345 && res.ms == 7);
	CHECK(progress_total == 2);
	// stop propagates as the host's -3
	res = Warm::GetSeam().extract(0x1000, 1, [&](const Warm::GroupView&) { return true; }, [](uint32_t, uint32_t) {});
	CHECK(res.code == -3 && res.decoded == 1);
	// a host that hands null bytes with a size, and a null group: nothing crashes, the split sees nothing
	g_fake.null_bytes_group = true;
	int empty_groups = 0;
	res = Warm::GetSeam().extract(
	    0x1000, 1, [&](const Warm::GroupView& v) { empty_groups += v.bytes.empty() ? 1 : 0; return false; }, [](uint32_t, uint32_t) {});
	CHECK(res.code == 0 && empty_groups == 1);
	g_fake.null_bytes_group = false;
	// wrong query values are the host's -1 (decoder and abi travel through the trampoline unchanged)
	CHECK(Warm::GetSeam().extract(0x2000, 1, [](const Warm::GroupView&) { return false; }, [](uint32_t, uint32_t) {}).code == -1);

	// ---- the main-thread hop -----------------------------------------------------------------------------------
	{
		std::atomic<bool> pump {true};
		std::thread       main_thread([&] {
            g_queue.main_id = std::this_thread::get_id();
            while (pump.load()) {
                PumpOne();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
		while (g_queue.main_id == std::this_thread::get_id()) std::this_thread::yield();
		std::thread::id ran_on;
		Warm::RunOnMain([&] { ran_on = std::this_thread::get_id(); });
		CHECK(ran_on == g_queue.main_id && ran_on != std::this_thread::get_id()); // ran on "main", and was waited for
		pump = false;
		main_thread.join();
	}
	g_queue.main_id = std::this_thread::get_id();
	{
		// m6: a main queue that never drains (KYTY_WINDOW_STEPPED=0): no wait, runs inline with a log
		drains = false;
		g_queue.main_id = std::thread::id(); // this thread is not the main thread
		g_queue.queued = 0;
		int ran = 0;
		Warm::RunOnMain([&] { ran++; });
		CHECK(ran == 1 && g_queue.queued == 0 && !logs.empty() && logs.back().second.find("KYTY_WINDOW_STEPPED=0") != std::string::npos);
		drains = true;
		// the host cannot queue: inline
		g_queue.result = -5;
		Warm::RunOnMain([&] { ran++; });
		CHECK(ran == 2);
		g_queue.result = 0;
		// on the main thread itself (here: this one): inline
		g_queue.main_id = std::this_thread::get_id();
		g_queue.queued = 0;
		Warm::RunOnMain([&] { ran++; });
		CHECK(ran == 3 && g_queue.queued == 0);
		// m12: the window closes while the job waits in a queue nobody serves: the wait ends, the job never runs
		g_queue.main_id = std::thread::id();
		Warm::ResetCancel();
		std::thread closer([] {
			std::this_thread::sleep_for(std::chrono::milliseconds(80));
			Warm::RequestCancel();
		});
		const auto begin = Warm::SteadyMs();
		Warm::RunOnMain([&] { ran += 100; });
		closer.join();
		CHECK(Warm::SteadyMs() - begin < 3000 && ran == 3);
		PumpOne(); // the main thread finally gets to it: the abandoned job does nothing
		CHECK(ran == 3);
		Warm::ResetCancel();
		g_queue.main_id = std::this_thread::get_id();
	}

	// ---- the whole guest_ready flow over the real bridge ----------------------------------------------------------
	{
		namespace L = Libs::Graphics;
		const auto path = (std::filesystem::temp_directory_path() / ("ue5bridge_" + std::to_string(getpid()))).string();
		std::error_code ec;
		std::filesystem::remove(path, ec);
		std::filesystem::remove(path + ".lock", ec);
		const auto hash_of = [](const std::vector<uint8_t>& c) {
			const auto h = XXH3_128bits(c.data(), c.size());
			return Hash128 {h.low64, h.high64};
		};
		const auto code_missing = RandomCode(3, 64); // listed but in no group
		Warm::WantedMap wanted;
		wanted[hash_of(code_a)]       = static_cast<uint32_t>(code_a.size());
		wanted[hash_of(code_b)]       = static_cast<uint32_t>(code_b.size());
		wanted[hash_of(code_missing)] = static_cast<uint32_t>(code_missing.size());
		PipelineCodeStore store;
		CHECK(store.Open(path, "PPSA00000", "00.000.000", Warm::Fnv1a64(decoder)));
		Warm::Config cfg;
		cfg.decoder  = decoder_guest; // the fingerprint is read through the real guest reader
		cfg.fp       = Warm::Fnv1a64(decoder);
		cfg.abi      = "oodlelz14";
		cfg.words_ok = true;
		cfg.max_s    = 30;
		g_fake.decoder = decoder_guest;
		std::vector<std::string> lines;
		Warm::Session session(cfg, &store, wanted, 3, [&](const std::string& l) { lines.push_back(l); }, {}, Warm::SteadyMs, 0x77);
		auto hooks = session.MakeHooks([] {}, [] {}, [] {});
		Warm::Flow flow;
		flow.deferred = true;
		g_fake.code = 0;
		const auto out = Warm::RunGuestReady(flow, hooks);
		CHECK(out == Warm::Outcome::Extracted && store.Count() == 2);
		CHECK(store.Size(hash_of(code_a)) == code_a.size() && store.Size(hash_of(code_b)) == code_b.size());
		const auto mark = store.Mark();
		CHECK(mark.state == PipelineCodeStore::SearchMark::Complete && mark.key == 0x77); // one code is not in the library
		bool summary = false;
		for (const auto& l: lines) summary |= l.rfind("UE5 code: 2 groups", 0) == 0 && l.find("found 2 of 3") != std::string::npos;
		CHECK(summary);
		store.Release();
		std::filesystem::remove(path, ec);
		std::filesystem::remove(path + ".lock", ec);
	}
	munmap(mem, page * 4);
	if (g_failed != 0) {
		std::printf("Ue5BridgeTests: %d check(s) failed\n", g_failed);
		return 1;
	}
	std::printf("Ue5BridgeTests: all passed\n");
	return 0;
}
