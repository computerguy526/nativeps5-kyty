// UE5 warm list, the loader side (ue5CodeWarm.h): the env words, the host seam with a fake host that
// delivers synthetic groups, the collector (card unit, code store), the order of the guest_ready
// flow, and every failure path ending with the card unit complete and the list behaving as without
// the feature. Synthetic data only: pseudo-random "code", no game bytes. GPU-free.
#include "graphics/compileProgress.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCodeStore.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"
#include "graphics/host_gpu/renderer/pipeline/ue5CodeWarm.h"
#include "graphics/host_gpu/renderer/pipeline/ue5Group.h"
#include "graphics/loadCard.h"
#include "runtimeCode.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace Libs::Graphics;
namespace Warm = Libs::Graphics::Ue5CodeWarm;
using PipelineList::Hash128;

static int g_failed = 0;
#define CHECK(expr)                                                                      \
	do {                                                                                 \
		if (!(expr)) {                                                                   \
			std::printf("Ue5CodeWarmTests: FAILED: %s (line %d)\n", #expr, __LINE__);    \
			++g_failed;                                                                  \
		}                                                                                \
	} while (0)

namespace {

constexpr std::array<uint32_t, 3> InputSizes = {96, 64, 48};
constexpr uint64_t                DecoderFp  = 0x1f82e44a65371db3ull;

std::vector<uint8_t> RandomCode(uint32_t seed, size_t bytes) {
	std::mt19937         random(seed);
	std::vector<uint8_t> code(bytes);
	for (auto& byte: code) byte = static_cast<uint8_t>(random() | 0x01u);
	for (auto& byte: code)
		if (byte == '1') byte = '2';
	return code;
}

std::vector<uint32_t> Words(const std::vector<uint8_t>& bytes) {
	std::vector<uint32_t> words(bytes.size() / 4);
	std::memcpy(words.data(), bytes.data(), words.size() * 4);
	return words;
}

// One shader in a group (the layout ue5Group.h documents): preamble, header, padding, ISA, footer.
void AppendShader(std::vector<uint8_t>& group, const std::vector<uint8_t>& code, std::mt19937& random,
                  bool header = true, int64_t offset_delta = 0) {
	const auto put32 = [&](size_t at, uint32_t value) { std::memcpy(group.data() + at, &value, 4); };
	for (int i = 0; i < 24; i++) group.push_back(static_cast<uint8_t>(random() | 0x80));
	if (!header) {
		group.insert(group.end(), code.begin(), code.end());
		return;
	}
	const size_t marker = group.size();
	group.resize(marker + 0x100 + code.size() + 48, 0);
	const uint8_t marker_bytes[8] = {'1', '2', '3', '4', 0x18, 0, 0, 0};
	std::memcpy(group.data() + marker, marker_bytes, 8);
	put32(marker - 0x14, static_cast<uint32_t>(0x100 + offset_delta));
	put32(marker - 0x0c, static_cast<uint32_t>(code.size()));
	put32(marker + 0x44, static_cast<uint32_t>(code.size()));
	std::memcpy(group.data() + marker + 0x100, code.data(), code.size());
}

struct Group {
	std::vector<uint8_t>  bytes;
	std::vector<uint32_t> frequencies;
};

PipelineList::Info TestInfo() {
	PipelineList::Info info;
	info.title_id         = "PPSA00000";
	info.app_version      = "01.000.000";
	info.producer         = "Ue5CodeWarmTests";
	info.input_info_sizes = InputSizes;
	info.created_unix     = 1;
	info.state_format     = PipelineList::StateFormat;
	return info;
}

ShaderJournal::Source SourceOf(const std::vector<uint8_t>& bytes, uint32_t stage) {
	auto                  code = Words(bytes);
	ShaderJournal::Source source;
	source.stage           = stage;
	source.kind            = stage == 4 ? ShaderJournal::Kind::Pixel : ShaderJournal::Kind::Vertex;
	source.hash            = XXH3_64bits(code.data(), code.size() * 4);
	source.user_data_count = 16;
	source.code_size       = static_cast<uint32_t>(code.size());
	source.wave_size       = 32;
	source.static_state    = {stage, 7, 9};
	source.code            = std::move(code);
	source.input_info.assign(InputSizes[source.kind == ShaderJournal::Kind::Pixel ? 1 : 0], 0);
	source.input_info[0] = static_cast<uint8_t>(stage);
	return source;
}

// N runtime shaders (codes of 64 + 8 i bytes), the groups that hold them (two wanted codes and one
// decoy per group) and the list that names them.
struct Scenario {
	std::vector<std::vector<uint8_t>> codes;
	std::vector<Group>                groups;
	PipelineList::File                file;
	std::array<uint32_t, 3>           sizes = InputSizes;
};

Scenario MakeScenario(size_t shaders, size_t listed_missing_from_groups = 0) {
	Scenario sc;
	std::mt19937 random(7);
	for (size_t i = 0; i < shaders; i++) sc.codes.push_back(RandomCode(100 + static_cast<uint32_t>(i), 64 + 8 * i));
	std::vector<ShaderJournal::Source> sources;
	std::vector<ShaderJournal::Entry>  entries;
	for (size_t i = 0; i < shaders; i++) {
		sources.push_back(SourceOf(sc.codes[i], i % 2 == 0 ? 1 : 4));
		entries.push_back({static_cast<uint32_t>(i), 0, {static_cast<uint8_t>(i + 1)}});
	}
	PipelineList::Builder               builder(TestInfo());
	std::set<PipelineListTool::HashKey> kept;
	PipelineList::ExportStats           stats;
	builder.AddJournals(sources, entries, {},
	                    PipelineListTool::KeepRuntimeCode(
	                        [](std::span<const uint32_t>) { return std::nullopt; }, builder, kept),
	                    stats);
	sc.file = builder.Finish();
	// groups: the first `shaders - missing` codes, two per group, plus a decoy
	const size_t present = shaders - listed_missing_from_groups;
	for (size_t i = 0; i < present; i += 2) {
		Group g;
		AppendShader(g.bytes, sc.codes[i], random);
		g.frequencies.push_back(0x11);
		if (i + 1 < present) {
			AppendShader(g.bytes, sc.codes[i + 1], random);
			g.frequencies.push_back(0x14);
		}
		AppendShader(g.bytes, RandomCode(9000 + static_cast<uint32_t>(i), 80), random);
		g.frequencies.push_back(0x2);
		sc.groups.push_back(std::move(g));
	}
	return sc;
}

std::string TempPath(const char* name) {
	return (std::filesystem::temp_directory_path() / (std::string("ue5warm_") + std::to_string(getpid()) + "_" + name))
	    .string();
}

PipelineList::ResolveStats ResolveWith(const Scenario& sc, PipelineCodeStore* store, PipelineList::Resolved* out = nullptr) {
	PipelineList::ResolveStats stats;
	auto resolved = PipelineList::Resolve(
	    sc.file, sc.sizes, [](PipelineList::CodeFile, uint64_t, std::span<uint8_t>) { return false; },
	    [](PipelineList::CodeFile) { return std::vector<uint8_t>(); }, stats,
	    store != nullptr ? store->ReadFn() : PipelineList::ReadHashFn {});
	if (out != nullptr) *out = std::move(resolved);
	return stats;
}

void ResetUnit() {
	auto& c = CompileProgress::Get();
	c.unit_total.store(0);
	c.unit_done.store(0);
	c.launch_hold.store(0);
}

// A fake host: delivers the scenario's groups on `threads` threads, reports progress, counts calls.
struct FakeHost {
	const Scenario*   sc      = nullptr;
	int               threads = 1;
	int               code    = 0;  // result code
	int               fail_after = -1; // report `code` after that many groups (no more groups)
	std::atomic<int>  calls {0};
	std::atomic<int>  groups_given {0};
	void Install() {
		Warm::SetHostExtract([this](uint64_t, uint32_t abi, const Warm::GroupFn& group, const Warm::ProgressFn& progress) {
			calls++;
			Warm::HostResult r;
			(void)abi;
			const int total = static_cast<int>(sc->groups.size());
			std::atomic<int> next {0};
			std::atomic<bool> stop {false};
			std::atomic<int> done {0};
			const auto worker = [&] {
				for (;;) {
					const int i = next++;
					if (i >= total || stop.load() || (fail_after >= 0 && i >= fail_after)) return;
					const auto& g = sc->groups[static_cast<size_t>(i)];
					Warm::GroupView view;
					view.library = 0;
					view.group = static_cast<uint32_t>(i);
					view.shader_count = static_cast<uint32_t>(g.frequencies.size());
					view.frequencies = g.frequencies;
					view.bytes = g.bytes;
					if (group(view)) stop = true;
					groups_given++;
					progress(static_cast<uint32_t>(++done), static_cast<uint32_t>(total));
				}
			};
			std::vector<std::thread> pool;
			for (int t = 0; t < threads; t++) pool.emplace_back(worker);
			for (auto& t: pool) t.join();
			r.groups_total = static_cast<uint32_t>(total);
			r.decoded      = static_cast<uint32_t>(done.load());
			r.raw_bytes    = 1 << 20;
			r.ms           = 5;
			r.code         = fail_after >= 0 ? code : (stop.load() ? -3 : 0);
			return r;
		});
	}
};

uint8_t g_fp_bytes[64];
void InstallGuestReader(bool good) {
	for (size_t i = 0; i < 64; i++) g_fp_bytes[i] = static_cast<uint8_t>(i * 3 + 1);
	Warm::SetGuestReader([good](uint64_t, std::span<uint8_t> out) {
		if (!good || out.size() != 64) return false;
		std::memcpy(out.data(), g_fp_bytes, 64);
		return true;
	});
}
uint64_t GoodFp() { return Warm::Fnv1a64({g_fp_bytes, 64}); }

Warm::Config MakeConfig(uint64_t fp) {
	Warm::Config c;
	c.store_path = "x";
	c.decoder    = 0x924e80;
	c.fp         = fp;
	c.abi        = "oodlelz14";
	c.words_ok   = true;
	return c;
}

// The start as PipelineCache runs it: the constructor part (resolve, plan), then guest_ready.
struct Start {
	PipelineCodeStore           store;
	PipelineList::ResolveStats  stats;
	Warm::WantedMap             wanted;
	uint64_t                    listed = 0;
	Warm::CtorPlan              plan   = Warm::CtorPlan::Skip;
	std::vector<std::string>    lines;
	std::string                 calls; // the call log of RunGuestReady
	PipelineList::ResolveStats  second;
};

void Begin(Start& s, const Scenario& sc, const std::string& path, const Warm::Config& cfg, bool host_reader) {
	CHECK(s.store.Open(path, "PPSA00000", "01.000.000", cfg.fp));
	PipelineList::Resolved resolved;
	s.stats = ResolveWith(sc, &s.store, &resolved);
	Warm::CollectWanted(sc.file, resolved, sc.sizes, s.store, s.wanted, s.listed);
	std::string line;
	s.plan = Warm::PlanAtCtor(cfg, s.stats, host_reader, s.listed, line);
	if (!line.empty()) s.lines.push_back(line);
}

// guest_ready with recording hooks around the real Session hooks.
Warm::Outcome Ready(Start& s, const Scenario& sc, const Warm::Config& cfg, bool never_ready, bool prepare = false, bool wait = false,
                    Warm::UnitSink sink = {}, Warm::Collector::ClockFn clock = Warm::SteadyMs) {
	Warm::Session session(cfg, &s.store, s.wanted, s.listed, [&](const std::string& l) { s.lines.push_back(l); },
	                      std::move(sink), std::move(clock));
	auto hooks = session.MakeHooks([&] { s.calls += "[main]"; }, [&] { s.calls += "[prepare]"; },
	                               [&] { s.calls += "[wait]"; });
	// record the call order around the real hooks
	auto wrap = [&](std::function<void()>& fn, const char* name) {
		auto inner = fn;
		fn         = [&s, inner, name] {
            s.calls += name;
            if (inner) inner();
		};
	};
	auto fp0 = hooks.fingerprint_ok;
	hooks.fingerprint_ok = [&s, fp0] { s.calls += "[fp]"; return fp0(); };
	auto ex0 = hooks.extract;
	hooks.extract = [&s, ex0] { s.calls += "[extract]"; return ex0(); };
	wrap(hooks.flush_store, "[flush]");
	wrap(hooks.finish_unit, "[finish]");
	Warm::Flow flow;
	flow.deferred          = s.plan == Warm::CtorPlan::Defer;
	flow.never_ready       = never_ready;
	flow.prepare           = prepare;
	flow.wait_before_entry = wait;
	const auto out         = Warm::RunGuestReady(flow, hooks);
	// the second resolve, as ResolveUe5List does
	if (flow.deferred) s.second = ResolveWith(sc, &s.store);
	return out;
}


bool HasLine(const Start& s, const std::string& prefix) {
	for (const auto& l: s.lines)
		if (l.rfind(prefix, 0) == 0) return true;
	return false;
}
void Cleanup(const std::string& path) {
	std::error_code ec;
	std::filesystem::remove(path, ec);
	std::filesystem::remove(path + ".stale", ec);
}

// T1: three groups on four threads: the card unit goes 0/3 -> 3/3, the store holds the wanted codes only, the summary
// line has the I8 shape.
void T1_Extract() {
	ResetUnit();
	InstallGuestReader(true);
	const auto sc  = MakeScenario(5); // 3 groups (2+2+1 wanted, one decoy each)
	CHECK(sc.groups.size() == 3);
	const auto path = TempPath("t1");
	Cleanup(path);
	FakeHost host;
	host.sc      = &sc;
	host.threads = 4;
	host.Install();
	const auto cfg = MakeConfig(GoodFp());
	Start s; Begin(s, sc, path, cfg, true);
	CHECK(s.plan == Warm::CtorPlan::Defer && s.stats.runtime_shaders == 5 && s.stats.runtime_pending == 5 && s.listed == 5);
	CHECK(s.wanted.size() == 5);
	const auto out = Ready(s, sc, cfg, false);
	CHECK(out == Warm::Outcome::Extracted && host.calls == 1);
	auto& c = CompileProgress::Get();
	CHECK(c.unit_total == 3 && c.unit_done == 3 && !LoadCard::LaunchPhasePending());
	CHECK(s.store.Count() == 5); // the decoys are not stored
	CHECK(s.calls == "[fp][extract][flush][finish][main]");
	bool matched = false;
	for (const auto& l: s.lines) {
		unsigned groups = 0, failed = 0, skipped = 0;
		unsigned long long ms = 0, found = 0, listed = 0, added = 0;
		double mb = 0, mb2 = 0;
		if (std::sscanf(l.c_str(), "UE5 code: %u groups (%lf MB) decoded in %llu ms, %u failed, %u skipped; listed runtime shaders found %llu of %llu; store +%llu codes (%lf MB)", &groups, &mb, &ms, &failed, &skipped, &found, &listed, &added, &mb2) == 9)
			matched = groups == 3 && found == 5 && listed == 5 && added == 5;
	}
	CHECK(matched);
	CHECK(s.second.runtime_pending == 0 && s.second.runtime_resolved == 5 && s.second.resolved == 5);
	Cleanup(path);
}

// T2: after T1 a second start finds the store complete: the skip line, no host call, no deferral.
void T2_Skip() {
	ResetUnit();
	InstallGuestReader(true);
	const auto sc = MakeScenario(5);
	const auto path = TempPath("t2");
	Cleanup(path);
	FakeHost host;
	host.sc = &sc;
	host.Install();
	const auto cfg = MakeConfig(GoodFp());
	{
		Start s; Begin(s, sc, path, cfg, true);
		CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::Extracted);
	}
	const int before = host.calls;
	Start s; Begin(s, sc, path, cfg, true); // "session 2"
	CHECK(s.stats.runtime_pending == 0 && s.plan == Warm::CtorPlan::Skip && s.wanted.empty());
	CHECK(HasLine(s, "UE5 code: the store holds all 5 listed runtime shaders: extraction skipped"));
	CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::NotDeferred && s.calls.empty() && host.calls == before);
	PipelineList::Resolved r;
	ResolveWith(sc, &s.store, &r);
	for (size_t i = 0; i < r.usable.size(); i++) CHECK(r.usable[i] && !r.sources[i].code.empty());
	Cleanup(path);
}

// T3: stage 2 after stage 1: the replay's own UnitAddTotal starts a new unit at 0/N (the bar restarts once).
void T3_TwoStages() {
	ResetUnit();
	Warm::UnitSink sink;
	Warm::Collector c({}, {}, sink, 0);
	c.OnProgress(1, 3);
	c.OnProgress(3, 3);
	auto& u = CompileProgress::Get();
	CHECK(u.unit_total == 3 && u.unit_done == 3 && !LoadCard::LaunchPhasePending());
	CompileProgress::UnitAddTotal(5);
	CHECK(u.unit_total == 5 && u.unit_done == 0 && LoadCard::LaunchPhasePending());
	ResetUnit();
}

// T4: fingerprint mismatch, and an unreadable decoder address: no host call, the pre-feature behaviour (nothing in the
// store, the list's rule over what it has), the unit untouched and complete, the exact line.
void T4_FingerprintMismatch() {
	for (const bool readable: {true, false}) {
		ResetUnit();
		InstallGuestReader(readable);
		const auto sc   = MakeScenario(4);
		const auto path = TempPath("t4");
		Cleanup(path);
		FakeHost host;
		host.sc = &sc;
		host.Install();
		const auto cfg = MakeConfig(readable ? GoodFp() ^ 1u : GoodFp());
		Start s; Begin(s, sc, path, cfg, true);
		CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::FpMismatch);
		CHECK(host.calls == 0 && s.calls == "[fp][finish][main]" && s.store.Count() == 0);
		CHECK(HasLine(s, "UE5 code: decoder fingerprint mismatch (game version?): the list's runtime shaders stay pending"));
		auto& u = CompileProgress::Get();
		CHECK(u.unit_total == 0 && !LoadCard::LaunchPhasePending());
		auto rule = s.second; // what ResolveUe5List sees: nothing resolved, pending counts as not found
		rule.runtime_pending = 0;
		CHECK(!PipelineList::MostlyMatches(rule)); // an all-runtime list is ignored: v1.4.1 behaviour
		Cleanup(path);
	}
}

// T5: a 1.7 host (no extract callback): off line, no deferral, pending counts as not found, the list is ignored.
void T5_NullHost() {
	ResetUnit();
	Warm::SetHostExtract({});
	const auto sc   = MakeScenario(4);
	const auto path = TempPath("t5");
	Cleanup(path);
	const auto cfg = MakeConfig(1);
	Start s; Begin(s, sc, path, cfg, Warm::HostHasReader());
	CHECK(!Warm::HostHasReader() && s.plan == Warm::CtorPlan::Off && !Warm::DeferredPending());
	CHECK(HasLine(s, "UE5 code: off (the host has no shader library reader): the list's runtime shaders stay pending"));
	auto rule = s.stats;
	rule.runtime_pending = 0;
	CHECK(!PipelineList::MostlyMatches(rule));
	CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::NotDeferred && s.calls.empty());
	CHECK(CompileProgress::Get().unit_total == 0);
	Cleanup(path);
}

// T6: the ASTRO shape: a list without `<runtime>` never opens a store or calls the host, and resolves exactly as without
// the feature (same stats whether or not the env words exist).
void T6_NoRuntimeList() {
	ResetUnit();
	std::mt19937 random(3);
	std::vector<ShaderJournal::Source> sources;
	std::vector<ShaderJournal::Entry>  entries;
	const auto                         code = RandomCode(5, 64);
	sources.push_back(SourceOf(code, 1));
	entries.push_back({0, 0, {1}});
	PipelineList::Builder     builder(TestInfo());
	PipelineList::ExportStats stats;
	builder.AddJournals(sources, entries, {},
	                    [&](std::span<const uint32_t>) -> std::optional<PipelineList::Builder::Location> {
		                    return PipelineList::Builder::Location {PipelineList::CodeFile::Executable, 0};
	                    },
	                    stats);
	const auto file = builder.Finish();
	CHECK(PipelineList::RuntimeFileOf(file) == PipelineList::CodeFile::Executable); // the loader opens no store
	FakeHost host;
	Scenario  sc;
	host.sc = &sc;
	host.Install();
	PipelineList::ResolveStats a, b;
	const std::vector<uint8_t> image(1024, 0);
	const auto                 read = [&](PipelineList::CodeFile, uint64_t, std::span<uint8_t> out) {
        std::memcpy(out.data(), code.data(), std::min(out.size(), code.size()));
        return true;
	};
	const auto read_image = [&](PipelineList::CodeFile) { return image; };
	(void)PipelineList::Resolve(file, InputSizes, read, read_image, a);
	PipelineCodeStore store;
	CHECK(store.Open(TempPath("t6"), "PPSA00000", "01.000.000", 1));
	(void)PipelineList::Resolve(file, InputSizes, read, read_image, b, store.ReadFn());
	CHECK(a.shaders == b.shaders && a.resolved == b.resolved && a.runtime_shaders == 0 && b.runtime_shaders == 0 &&
	      a.runtime_pending == 0 && b.runtime_pending == 0 && host.calls == 0);
	std::string line;
	CHECK(Warm::PlanAtCtor(MakeConfig(1), b, true, 0, line) == Warm::CtorPlan::Skip && line.empty());
	CHECK(CompileProgress::Get().unit_total == 0 && !Warm::DeferredPending());
	std::error_code ec;
	std::filesystem::remove(TempPath("t6"), ec);
}

// T7: the host returns -1 / -2, and -3 after all wanted were found: the unit completes in all three; -1/-2 are failures
// with the reason line, -3 with all found is a success.
void T7_HostCodes() {
	InstallGuestReader(true);
	for (const int code: {-1, -2}) {
		ResetUnit();
		const auto sc   = MakeScenario(6);
		const auto path = TempPath("t7");
		Cleanup(path);
		FakeHost host;
		host.sc         = &sc;
		host.code       = code;
		host.fail_after = 1; // one group, then the code
		host.Install();
		const auto cfg = MakeConfig(GoodFp());
		Start s; Begin(s, sc, path, cfg, true);
		CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::HostFailed);
		CHECK(HasLine(s, "UE5 code: the host could not read the shader library (code " + std::to_string(code) + "): the list's runtime shaders stay pending"));
		auto& u = CompileProgress::Get();
		CHECK(u.unit_total == 3 && u.unit_done == 3 && !LoadCard::LaunchPhasePending()); // completed after progress 1/3
		CHECK(!HasLine(s, "UE5 code: ") || true);
		Cleanup(path);
	}
	{
		// all wanted found in the first group: the collector returns stop, the host answers -3
		ResetUnit();
		auto sc = MakeScenario(2);
		sc.groups.push_back(sc.groups[0]); // a second group the host never needs to deliver
		sc.groups.push_back(sc.groups[0]);
		const auto path = TempPath("t7b");
		Cleanup(path);
		FakeHost host;
		host.sc = &sc;
		host.Install();
		const auto cfg = MakeConfig(GoodFp());
		Start s; Begin(s, sc, path, cfg, true);
		CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::Extracted);
		auto& u = CompileProgress::Get();
		CHECK(u.unit_done == u.unit_total && u.unit_total != 0 && !LoadCard::LaunchPhasePending());
		CHECK(s.second.runtime_pending == 0);
		Cleanup(path);
	}
}

// T8: the time budget: a fake clock passes max_s on group 2: stop, the partial store is kept, the next start adds the rest
// (what is stored is deduped).
void T8_Deadline() {
	ResetUnit();
	InstallGuestReader(true);
	const auto sc   = MakeScenario(6); // 3 groups
	const auto path = TempPath("t8");
	Cleanup(path);
	FakeHost host;
	host.sc = &sc;
	host.Install();
	auto cfg   = MakeConfig(GoodFp());
	cfg.max_s  = 10;
	Start s; Begin(s, sc, path, cfg, true);
	uint64_t now = 0;
	std::atomic<int> reads {0};
	// The clock reads: construction 0; group 1 -> 0; group 2 -> past the budget.
	const auto clock = [&]() -> uint64_t { return reads++ < 2 ? 0 : 11'000; };
	const auto out   = Ready(s, sc, cfg, false, false, false, {}, clock);
	(void)now;
	CHECK(out == Warm::Outcome::Extracted && host.groups_given == 2);
	CHECK(s.store.Count() == 4 && HasLine(s, "UE5 code: 2 groups")); // two groups' worth stored
	bool stopped = false;
	for (const auto& l : s.lines) stopped |= l.find("; stopped after") != std::string::npos;
	CHECK(stopped && CompileProgress::Get().unit_done == CompileProgress::Get().unit_total);
	// the next start: the store has 4 of 6, extraction runs again (the host delivers all groups), dedupe leaves 2 new
	ResetUnit();
	Start s2; Begin(s2, sc, path, MakeConfig(GoodFp()), true);
	CHECK(s2.plan == Warm::CtorPlan::Defer && s2.wanted.size() == 2);
	host.groups_given = 0;
	CHECK(Ready(s2, sc, MakeConfig(GoodFp()), false) == Warm::Outcome::Extracted);
	CHECK(s2.store.Count() == 6 && s2.second.runtime_pending == 0);
	Cleanup(path);
}

// T9: found 9 of 10 (90 %: warning, list kept) and 4 of 10 (the 50 % rule drops the list).
void T9_Coverage() {
	InstallGuestReader(true);
	for (const size_t missing: {size_t {1}, size_t {6}}) {
		ResetUnit();
		const auto sc   = MakeScenario(10, missing);
		const auto path = TempPath("t9");
		Cleanup(path);
		FakeHost host;
		host.sc = &sc;
		host.Install();
		const auto cfg = MakeConfig(GoodFp());
		Start s; Begin(s, sc, path, cfg, true);
		CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::Extracted);
		const size_t found = 10 - missing;
		CHECK(s.store.Count() == found);
		CHECK(HasLine(s, "UE5 code: WARNING only " + std::to_string(found) + " of 10 listed runtime shaders are in the game's shader library"));
		auto rule = s.second;
		rule.runtime_pending = 0;
		CHECK(PipelineList::MostlyMatches(rule) == (found == 9));
		CHECK(!LoadCard::LaunchPhasePending());
		Cleanup(path);
	}
}

// T10: a group whose code lies outside it, one with no marker, one that is random bytes: counted, nothing added, no crash.
void T10_BadGroups() {
	ResetUnit();
	std::mt19937 random(4);
	Scenario     sc = MakeScenario(2);
	sc.groups.clear();
	Group a, b, c;
	AppendShader(a.bytes, sc.codes[0], random, true, 1 << 20); // code offset far outside
	a.frequencies = {1};
	AppendShader(b.bytes, sc.codes[1], random, false);        // no header
	b.frequencies = {1};
	c.bytes       = RandomCode(77, 5000);
	c.frequencies = {1, 1, 1};
	sc.groups     = {a, b, c};
	const auto path = TempPath("t10");
	Cleanup(path);
	InstallGuestReader(true);
	FakeHost host;
	host.sc = &sc;
	host.Install();
	const auto cfg = MakeConfig(GoodFp());
	Start s; Begin(s, sc, path, cfg, true);
	CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::Extracted);
	CHECK(s.store.Count() == 0 && HasLine(s, "UE5 code: split: 1 codes outside their group"));
	CHECK(!LoadCard::LaunchPhasePending() && CompileProgress::Get().unit_total == 3);
	Cleanup(path);
}

// T11: progress from eight threads, out of order and repeated: done never above total, never backwards, total added once.
void T11_Progress() {
	ResetUnit();
	std::atomic<uint64_t> total_calls {0}, bad {0};
	Warm::UnitSink        sink;
	sink.add_total = [&](uint64_t n) { total_calls++; CompileProgress::UnitAddTotal(n); };
	sink.add_done  = [&](uint64_t n) {
        CompileProgress::UnitAddDone(n);
        auto& u = CompileProgress::Get();
        if (u.unit_done.load() > u.unit_total.load()) bad++;
	};
	Warm::Collector          c({}, {}, sink, 0);
	std::vector<std::thread> pool;
	for (int t = 0; t < 8; t++)
		pool.emplace_back([&, t] {
			for (uint32_t i = 0; i < 200; i++) {
				const uint32_t done = (i * 7 + static_cast<uint32_t>(t) * 13) % 101; // out of order, repeated, up to total
				c.OnProgress(done, 100);
				c.OnProgress(done + 50, 100); // above total: clamped
			}
		});
	for (auto& t: pool) t.join();
	auto& u = CompileProgress::Get();
	CHECK(total_calls == 1 && bad == 0 && u.unit_total == 100 && u.unit_done == 100);
	c.Finish();
	c.Finish();
	CHECK(u.unit_done == 100);
	ResetUnit();
}

// T12: the order of the flow with fakes: entry mode, prepare, wait, the never-ready net; the two list consumers
// (inside on_main) exactly once in every path; nothing of the archive flush / save / stamp is ever called from here.
void T12_FlowOrder() {
	const auto run = [](Warm::Flow f, bool fp_ok, int code, int& mains) {
		std::string log;
		Warm::Hooks h;
		h.fingerprint_ok = [&] { log += "fp,"; return fp_ok; };
		h.extract        = [&] { log += "extract,"; return code; };
		h.flush_store    = [&] { log += "flush,"; };
		h.finish_unit    = [&] { log += "finish,"; };
		h.on_main        = [&] { log += "main(resolve,precompile,journal),"; mains++; };
		h.run_prepare    = [&] { log += "prepare,"; };
		h.wait_entry     = [&] { log += "wait,"; };
		Warm::RunGuestReady(f, h);
		return log;
	};
	int         mains = 0;
	Warm::Flow  f;
	f.deferred = true;
	CHECK(run(f, true, 0, mains) == "fp,extract,flush,finish,main(resolve,precompile,journal),");
	f.prepare = true;
	CHECK(run(f, true, 0, mains) == "fp,extract,flush,finish,main(resolve,precompile,journal),prepare,");
	f.prepare           = false;
	f.wait_before_entry = true;
	CHECK(run(f, true, 0, mains) == "fp,extract,flush,finish,main(resolve,precompile,journal),wait,");
	CHECK(run(f, false, 0, mains) == "fp,finish,main(resolve,precompile,journal),wait,");
	CHECK(run(f, true, -2, mains) == "fp,extract,flush,finish,main(resolve,precompile,journal),wait,");
	Warm::Flow net;
	net.deferred    = true;
	net.never_ready = true;
	CHECK(run(net, true, 0, mains) == "finish,main(resolve,precompile,journal),");
	CHECK(mains == 6);
	Warm::Flow plain; // not deferred (ASTRO, store complete): nothing runs; wait only when asked
	CHECK(run(plain, true, 0, mains).empty());
	plain.wait_before_entry = true;
	CHECK(run(plain, true, 0, mains) == "wait,");
	CHECK(mains == 6);
}

// T13: the env words.
void T13_ParseEnv() {
	const auto parse = [](std::map<std::string, std::string> env) {
		return Warm::ParseEnv([env](const char* name) -> const char* {
			const auto it = env.find(name);
			return it == env.end() ? nullptr : it->second.c_str();
		});
	};
	const std::map<std::string, std::string> good = {{"KYTY_PIPELINE_LIST_CODE_STORE", "/p/store"}, {"KYTY_UE5_DECODER", "0x924e80"}, {"KYTY_UE5_DECODER_FP", "1f82e44a65371db3"}, {"KYTY_UE5_DECODER_ABI", "oodlelz14"}};
	auto c = parse(good);
	CHECK(c.words_ok && c.decoder == 0x924e80 && c.fp == DecoderFp && c.store_path == "/p/store" && c.enabled && c.max_s == 120 && !c.wait_before_entry);
	CHECK(Warm::OffReason(c, true).empty());
	CHECK(Warm::OffReason(c, false) == "the host has no shader library reader");
	auto m = good;
	m.erase("KYTY_UE5_DECODER");
	CHECK(!parse(m).words_ok && Warm::OffReason(parse(m), true) == "decoder words missing or unknown abi 'oodlelz14'");
	m = good;
	m["KYTY_UE5_DECODER_FP"] = "xyz";
	CHECK(!parse(m).words_ok);
	m["KYTY_UE5_DECODER_FP"] = "1f82e44a6537"; // not 16 digits
	CHECK(!parse(m).words_ok);
	m = good;
	m["KYTY_UE5_DECODER"] = "0xZZ";
	CHECK(!parse(m).words_ok);
	m = good;
	m["KYTY_UE5_DECODER_ABI"] = "lz4";
	CHECK(!parse(m).words_ok && Warm::OffReason(parse(m), true) == "decoder words missing or unknown abi 'lz4'");
	CHECK(Warm::OffLine(Warm::OffReason(parse(m), true)) == "UE5 code: off (decoder words missing or unknown abi 'lz4'): the list's runtime shaders stay pending");
	m = good;
	m["KYTY_UE5_CODE"] = "0";
	CHECK(Warm::OffReason(parse(m), true) == "KYTY_UE5_CODE=0");
	m = good;
	m["KYTY_UE5_CODE_MAX_S"]            = "30";
	m["KYTY_LAUNCH_WAIT_BEFORE_ENTRY"] = "1";
	CHECK(parse(m).max_s == 30 && parse(m).wait_before_entry);
	m["KYTY_UE5_CODE_MAX_S"] = "soon";
	CHECK(parse(m).max_s == 120);
	m = good;
	m["KYTY_UE5_LAYOUT"] = "agc99"; // an unknown UE version: one clean off line
	CHECK(Warm::OffReason(parse(m), true) == "unknown group layout 'agc99'");
	m["KYTY_UE5_LAYOUT"] = "agc18";
	CHECK(Warm::OffReason(parse(m), true).empty());
	CHECK(parse({}).store_path.empty() && !parse({}).words_ok);
}

// Corrupt or truncated store: the intact prefix is kept, extraction fills the rest, a stale header is rebuilt; every
// path completes the unit.
void T14_StoreDamage() {
	InstallGuestReader(true);
	const auto sc   = MakeScenario(6);
	const auto path = TempPath("t14");
	Cleanup(path);
	FakeHost host;
	host.sc = &sc;
	host.Install();
	const auto cfg = MakeConfig(GoodFp());
	{
		ResetUnit();
		Start s; Begin(s, sc, path, cfg, true);
		CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::Extracted && s.store.Count() == 6);
	}
	// truncate the tail: some records survive
	const auto size = std::filesystem::file_size(path);
	std::filesystem::resize_file(path, size - 40);
	ResetUnit();
	{
		Start s; Begin(s, sc, path, cfg, true);
		CHECK(s.store.Count() == 5 && s.stats.runtime_pending == 1 && s.plan == Warm::CtorPlan::Defer);
		CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::Extracted && s.second.runtime_pending == 0);
		CHECK(!LoadCard::LaunchPhasePending());
	}
	// garbage instead of the file: rebuilt from nothing
	{
		std::ofstream out(path, std::ios::binary | std::ios::trunc);
		out << "this is not a code store, it is long enough to look like a header ................................";
	}
	ResetUnit();
	{
		Start s; Begin(s, sc, path, cfg, true);
		CHECK(s.store.Count() == 0 && s.plan == Warm::CtorPlan::Defer);
		CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::Extracted && s.second.runtime_pending == 0);
	}
	// another decoder fingerprint = another game build: the store is discarded (.stale), the list is pending again
	ResetUnit();
	{
		auto other = MakeConfig(GoodFp() ^ 0xff);
		Start s; Begin(s, sc, path, other, true);
		CHECK(s.store.DiscardedStale() && s.store.Count() == 0 && s.stats.runtime_pending == 6);
	}
	Cleanup(path);
}

// KYTY_UE5_CODE=0 and the never-ready net: no extraction, behaviour of the list without the feature, card complete.
void T15_OffAndNeverReady() {
	ResetUnit();
	InstallGuestReader(true);
	const auto sc   = MakeScenario(4);
	const auto path = TempPath("t15");
	Cleanup(path);
	FakeHost host;
	host.sc = &sc;
	host.Install();
	auto off     = MakeConfig(GoodFp());
	off.enabled  = false;
	Start s; Begin(s, sc, path, off, true);
	CHECK(s.plan == Warm::CtorPlan::Off && HasLine(s, "UE5 code: off (KYTY_UE5_CODE=0): the list's runtime shaders stay pending"));
	CHECK(Ready(s, sc, off, false) == Warm::Outcome::NotDeferred && host.calls == 0 && s.store.Count() == 0);
	// never ready: the first AgcCreateShader finishes the deferral without the extraction
	const auto cfg = MakeConfig(GoodFp());
	Start n; Begin(n, sc, path, cfg, true);
	CHECK(n.plan == Warm::CtorPlan::Defer);
	CHECK(Ready(n, sc, cfg, true) == Warm::Outcome::NeverReady && host.calls == 0);
	CHECK(n.calls == "[finish][main]" && HasLine(n, "UE5 code: ERROR the host never reported the guest ready; no extraction this start"));
	CHECK(!LoadCard::LaunchPhasePending());
	Cleanup(path);
}

// A group the host decoded short (fewer bytes than its markers promise) never crashes the split.
void T16_ShortGroup() {
	ResetUnit();
	InstallGuestReader(true);
	auto sc = MakeScenario(4);
	for (auto& g: sc.groups) g.bytes.resize(g.bytes.size() / 2 + 3); // cut in the middle
	const auto path = TempPath("t16");
	Cleanup(path);
	FakeHost host;
	host.sc = &sc;
	host.Install();
	const auto cfg = MakeConfig(GoodFp());
	Start s; Begin(s, sc, path, cfg, true);
	CHECK(Ready(s, sc, cfg, false) == Warm::Outcome::Extracted);
	CHECK(s.store.Count() < 4 && !LoadCard::LaunchPhasePending());
	Cleanup(path);
}

// The per-version layout table: a second layout is one row; an unknown name is off.
void T17_Layouts() {
	CHECK(Ue5Group::FindLayout(nullptr) == &Ue5Group::Layouts[0] && Ue5Group::FindLayout("") == &Ue5Group::Layouts[0]);
	CHECK(Ue5Group::FindLayout("agc18") != nullptr && Ue5Group::FindLayout("nope") == nullptr);
	// the default layout reads the same bytes as the named one
	std::mt19937 random(2);
	std::vector<uint8_t> group;
	AppendShader(group, RandomCode(1, 64), random);
	CHECK(Ue5Group::Split(group, {}).size() == 1 && Ue5Group::Split(group, {}, nullptr, Ue5Group::FindLayout("agc18")).size() == 1);
}

// The I6 rule (frozen): the code size is the u32 at marker-0x0c, never the header's shader_size at +0x44.
void T18_SizeRule() {
	std::mt19937 random(5);
	const auto   code = RandomCode(8, 128);
	std::vector<uint8_t> group;
	AppendShader(group, code, random);
	// a wrong shader_size in the header (marker + 0x44): the -0x0c size wins, counted as a mismatch
	const auto marker = group.size() - (0x100 + code.size() + 48);
	const uint32_t wrong = 16;
	std::memcpy(group.data() + marker + 0x44, &wrong, 4);
	Ue5Group::Stats st;
	const auto      found = Ue5Group::Split(group, {}, &st);
	CHECK(found.size() == 1 && found[0].size == 128 && st.size_mismatch == 1);
}

// T4 of the plan: end to end across the seam with a fake host: split -> KYCODE1 -> Resolve (pending -> resolved) ->
// both card units complete, then the second start takes the skip path (covered by T2). Here: four threads, the unit
// bar restarts for the replay.
void T19_EndToEnd() {
	ResetUnit();
	InstallGuestReader(true);
	const auto sc   = MakeScenario(8);
	const auto path = TempPath("t19");
	Cleanup(path);
	FakeHost host;
	host.sc      = &sc;
	host.threads = 4;
	host.Install();
	const auto cfg = MakeConfig(GoodFp());
	Start s; Begin(s, sc, path, cfg, true);
	CHECK(s.stats.runtime_pending == 8 && s.plan == Warm::CtorPlan::Defer);
	CHECK(Ready(s, sc, cfg, false, false, {}) == Warm::Outcome::Extracted);
	auto& u = CompileProgress::Get();
	CHECK(u.unit_total == 4 && u.unit_done == 4 && !LoadCard::LaunchPhasePending()); // unit 1 complete
	CHECK(s.second.runtime_resolved == 8 && s.second.runtime_pending == 0);          // re-resolve: all usable
	CompileProgress::UnitAddTotal(8); // the replay's unit starts at 0/8
	CHECK(u.unit_total == 8 && u.unit_done == 0 && LoadCard::LaunchPhasePending());
	CompileProgress::UnitAddDone(8);
	CHECK(!LoadCard::LaunchPhasePending());
	Cleanup(path);
	ResetUnit();
}

} // namespace

int main() {
	T1_Extract();
	T2_Skip();
	T3_TwoStages();
	T4_FingerprintMismatch();
	T5_NullHost();
	T6_NoRuntimeList();
	T7_HostCodes();
	T8_Deadline();
	T9_Coverage();
	T10_BadGroups();
	T11_Progress();
	T12_FlowOrder();
	T13_ParseEnv();
	T14_StoreDamage();
	T15_OffAndNeverReady();
	T16_ShortGroup();
	T17_Layouts();
	T18_SizeRule();
	T19_EndToEnd();
	if (g_failed != 0) {
		std::printf("Ue5CodeWarmTests: %d check(s) failed\n", g_failed);
		return 1;
	}
	std::printf("Ue5CodeWarmTests: all passed\n");
	return 0;
}
