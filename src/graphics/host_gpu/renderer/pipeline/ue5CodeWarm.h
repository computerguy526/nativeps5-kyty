#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UE5CODEWARM_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UE5CODEWARM_H_

#include "graphics/compileProgress.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCodeStore.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"
#include "graphics/host_gpu/renderer/pipeline/ue5Group.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

// nps5 UE5 warm list: the logic around the host's shader library extraction (ABI 1.8) that needs no
// renderer: the env words, the host seam, the collector that turns decoded groups into code store
// entries and card progress, and the order of the guest_ready flow. Header only so
// tests/Ue5CodeWarmTests.cpp runs without a GPU. Everything is off unless
// KYTY_PIPELINE_LIST_CODE_STORE is set and the list has `<runtime>` shaders.
namespace Libs::Graphics::Ue5CodeWarm {

using Hash128 = PipelineList::Hash128;
struct HashOf {
	size_t operator()(const Hash128& h) const noexcept {
		return static_cast<size_t>(h.low ^ (h.high * 0x9E3779B97F4A7C15ull));
	}
};
using WantedMap = std::unordered_map<Hash128, uint32_t, HashOf>;

[[nodiscard]] inline uint64_t Fnv1a64(std::span<const uint8_t> bytes) {
	uint64_t h = 0xcbf29ce484222325ull;
	for (const auto b: bytes) {
		h ^= b;
		h *= 0x100000001b3ull;
	}
	return h;
}

// The decoder ABIs the host knows (KytyGpuUe5Query.abi), as data.
struct AbiRow {
	const char* name;
	uint32_t    id;
};
inline constexpr AbiRow       Abis[] = {{"oodlelz14", 1}};
[[nodiscard]] inline uint32_t AbiId(std::string_view name) {
	for (const auto& row: Abis)
		if (name == row.name) return row.id;
	return 0;
}

struct Config {
	std::string store_path;
	uint64_t    decoder = 0;
	uint64_t    fp      = 0;
	std::string abi;
	std::string layout;                   // KYTY_UE5_LAYOUT (ue5Group.h Layouts); empty = the first
	bool        enabled           = true; // KYTY_UE5_CODE != 0
	bool        words_ok          = false; // decoder, fp and abi parsed
	uint64_t    max_s             = 120;   // KYTY_UE5_CODE_MAX_S (0 = no limit)
	bool        wait_before_entry = false; // KYTY_LAUNCH_WAIT_BEFORE_ENTRY
};

using GetEnvFn = std::function<const char*(const char*)>;

// Hex with an optional 0x prefix; `exact` != 0 demands that many digits.
[[nodiscard]] inline bool ParseHex(const char* text, uint64_t& out, size_t exact = 0) {
	if (text == nullptr) return false;
	std::string_view v(text);
	if (v.starts_with("0x") || v.starts_with("0X")) v.remove_prefix(2);
	if (v.empty() || v.size() > 16 || (exact != 0 && v.size() != exact)) return false;
	uint64_t value = 0;
	for (const char c: v) {
		uint32_t d;
		if (c >= '0' && c <= '9')
			d = static_cast<uint32_t>(c - '0');
		else if (c >= 'a' && c <= 'f')
			d = static_cast<uint32_t>(c - 'a' + 10);
		else if (c >= 'A' && c <= 'F')
			d = static_cast<uint32_t>(c - 'A' + 10);
		else
			return false;
		value = (value << 4u) | d;
	}
	out = value;
	return true;
}

[[nodiscard]] inline Config ParseEnv(const GetEnvFn& get) {
	Config     c;
	const auto text = [&](const char* name) -> const char* {
		const char* v = get(name);
		return v != nullptr && *v != '\0' ? v : nullptr;
	};
	if (const auto* v = text("KYTY_PIPELINE_LIST_CODE_STORE")) c.store_path = v;
	if (const auto* v = text("KYTY_UE5_CODE")) c.enabled = std::string_view(v) != "0";
	if (const auto* v = text("KYTY_UE5_DECODER_ABI")) c.abi = v;
	if (const auto* v = text("KYTY_UE5_LAYOUT")) c.layout = v;
	if (const auto* v = text("KYTY_UE5_CODE_MAX_S")) {
		char*      end = nullptr;
		const auto n   = std::strtoull(v, &end, 10);
		if (end != v && *end == '\0') c.max_s = n;
	}
	if (const auto* v = text("KYTY_LAUNCH_WAIT_BEFORE_ENTRY"))
		c.wait_before_entry = std::string_view(v) != "0";
	c.words_ok = ParseHex(text("KYTY_UE5_DECODER"), c.decoder) && c.decoder != 0 &&
	             ParseHex(text("KYTY_UE5_DECODER_FP"), c.fp, 16) && AbiId(c.abi) != 0;
	return c;
}

// Empty: the extraction can run. Else the reason (the text inside the parentheses of OffLine).
[[nodiscard]] inline std::string OffReason(const Config& c, bool host_has_reader) {
	if (!c.enabled) return "KYTY_UE5_CODE=0";
	if (!host_has_reader) return "the host has no shader library reader";
	if (Ue5Group::FindLayout(c.layout.c_str()) == nullptr)
		return "unknown group layout '" + c.layout + "'";
	if (!c.words_ok) return "decoder words missing or unknown abi '" + c.abi + "'";
	return {};
}
[[nodiscard]] inline std::string OffLine(const std::string& reason) {
	return "UE5 code: off (" + reason + "): the list's runtime shaders stay pending";
}
[[nodiscard]] inline std::string SearchedLine(uint64_t missing) {
	return "UE5 code: " + std::to_string(missing) +
	       " listed runtime shaders are not in the game's shader library (searched): extraction "
	       "skipped";
}
[[nodiscard]] inline std::string StoreFullLine(uint64_t missing) {
	return "UE5 code: the code store is full (64 MB): " + std::to_string(missing) +
	       " listed runtime shaders cannot be stored: extraction skipped";
}
[[nodiscard]] inline std::string StoreLockedLine() {
	return "UE5 code: the code store is in use by another process: extraction skipped";
}
// After Open: what the store holds and what was done to the file.
[[nodiscard]] inline std::string StoreHealthLine(const PipelineCodeStore& store) {
	char text[200];
	std::snprintf(text, sizeof(text), "UE5 code: store %zu codes (%.1f MB)", store.Count(),
	              static_cast<double>(store.CodeBytes()) / 1048576.0);
	std::string line = text;
	if (store.TruncatedBytes() != 0)
		line += ", damaged tail of " + std::to_string(store.TruncatedBytes()) + " bytes cut";
	if (store.DiscardedStale()) line += ", an older store set aside (.stale)";
	if (store.Full()) line += ", full";
	return line;
}
[[nodiscard]] inline std::string SkipLine(uint64_t listed) {
	return "UE5 code: the store holds all " + std::to_string(listed) +
	       " listed runtime shaders: extraction skipped";
}
[[nodiscard]] inline std::string FpMismatchLine() {
	return "UE5 code: decoder fingerprint mismatch (game version?): the list's runtime shaders "
	       "stay pending";
}
[[nodiscard]] inline std::string HostFailedLine(int code) {
	return "UE5 code: the host could not read the shader library (code " + std::to_string(code) +
	       "): the list's runtime shaders stay pending";
}
[[nodiscard]] inline std::string NeverReadyLine() {
	return "UE5 code: ERROR the host never reported the guest ready; no extraction this start";
}
[[nodiscard]] inline std::string LowFoundLine(uint64_t found, uint64_t listed,
                                              bool stopped_early = false, uint64_t stopped_s = 0) {
	if (stopped_early) {
		return "UE5 code: WARNING stopped after " + std::to_string(stopped_s) + " s with " +
		       std::to_string(found) + " of " + std::to_string(listed) +
		       " listed runtime shaders found; the next start continues";
	}
	return "UE5 code: WARNING only " + std::to_string(found) + " of " + std::to_string(listed) +
	       " listed runtime shaders are in the game's shader library";
}
[[nodiscard]] inline std::string WatchdogLine(uint64_t waited_s) {
	return "UE5 code: the host did not finish reading the shader library within " +
	       std::to_string(waited_s) +
	       " s: extraction abandoned; the list's runtime shaders stay "
	       "pending";
}
// The list was ignored by the 50 % rule. `pending_excluded`: the rule did not count the runtime
// shaders still pending (a deferred start), else it counted them as not found.
[[nodiscard]] inline std::string IgnoredLine(const PipelineList::ResolveStats& real,
                                             bool pending_excluded, const std::string& producer) {
	const uint64_t shown = real.shaders - (pending_excluded ? real.runtime_pending : 0);
	// Lists without runtime shaders keep the text they always had.
	const std::string runtime = real.runtime_shaders == 0 ? std::string()
	                                                      : std::to_string(real.runtime_pending) +
	                                                            " runtime code not in the store, ";
	return "Pipeline list: only " + std::to_string(real.resolved) + " of " + std::to_string(shown) +
	       " shaders match this game's files (" + std::to_string(real.mismatched) + " differ, " +
	       std::to_string(real.other_input_layout) + " with another input layout, " + runtime +
	       "made by " + producer + "): ignored";
}
[[nodiscard]] inline bool LowFound(uint64_t found, uint64_t listed) {
	return found * 100u < listed * 95u;
}

// ---- host seam (C++ side of KytyGpuHostCallbacks::extract_ue5_shader_groups) ----
struct GroupView {
	uint32_t                  library = 0, group = 0, shader_count = 0;
	std::span<const uint32_t> frequencies;
	std::span<const uint8_t>  bytes;
};
struct HostResult {
	int      code         = 0; // 0 done, -1 unsupported, -2 decoder failed, -3 cancelled
	uint32_t groups_total = 0, decoded = 0, failed = 0, skipped = 0;
	uint64_t compressed_bytes = 0, raw_bytes = 0, ms = 0;
};
using GroupFn     = std::function<bool(const GroupView&)>; // true = stop
using ProgressFn  = std::function<void(uint32_t done, uint32_t total)>;
using ExtractFn   = std::function<HostResult(uint64_t decoder, uint32_t abi, const GroupFn& group,
                                             const ProgressFn& progress)>;
using RunOnMainFn = std::function<void(std::function<void()>)>;
using ReadGuestFn = std::function<bool(uint64_t guest_address, std::span<uint8_t> out)>;

struct Seam {
	ExtractFn   extract;
	RunOnMainFn run_on_main;
	ReadGuestFn read_guest;
};
inline Seam& GetSeam() {
	static Seam seam;
	return seam;
}
inline void SetHostExtract(ExtractFn fn) {
	GetSeam().extract = std::move(fn);
}
inline void SetRunOnMain(RunOnMainFn fn) {
	GetSeam().run_on_main = std::move(fn);
}
inline void SetGuestReader(ReadGuestFn fn) {
	GetSeam().read_guest = std::move(fn);
}
[[nodiscard]] inline bool HostHasReader() {
	return static_cast<bool>(GetSeam().extract);
}
inline void RunOnMain(std::function<void()> fn) {
	if (GetSeam().run_on_main)
		GetSeam().run_on_main(std::move(fn));
	else
		fn();
}
// FNV-1a 64 over the 64 bytes at the decoder; false when they cannot be read (= mismatch).
[[nodiscard]] inline bool FingerprintMatches(uint64_t decoder, uint64_t fp) {
	uint8_t bytes[64];
	if (!GetSeam().read_guest || !GetSeam().read_guest(decoder, bytes)) return false;
	return Fnv1a64(bytes) == fp;
}

// ---- the card unit over library groups ----
struct UnitSink {
	std::function<void(uint64_t)> add_total = [](uint64_t n) { CompileProgress::UnitAddTotal(n); };
	std::function<void(uint64_t)> add_done  = [](uint64_t n) { CompileProgress::UnitAddDone(n); };
};

inline uint64_t SteadyMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// A window close (or any exit) asks a running extraction to stop: the group callback answers
// "stop" at its next call, and RendererEnd waits (bounded) for the extraction to be out of the host
// before the process ends.
inline std::atomic<bool>     g_cancel {false};
inline std::atomic<uint32_t> g_active {0}; // extractions inside the host
inline void                  RequestCancel() {
	g_cancel.store(true, std::memory_order_release);
}
inline void ResetCancel() {
	g_cancel.store(false, std::memory_order_release);
}
[[nodiscard]] inline bool Cancelled() {
	return g_cancel.load(std::memory_order_acquire);
}
// True when no extraction is inside the host (within `max_ms`).
inline bool WaitIdle(uint64_t max_ms) {
	const auto until = SteadyMs() + max_ms;
	while (g_active.load(std::memory_order_acquire) != 0) {
		if (SteadyMs() >= until) return false;
		std::this_thread::sleep_for(std::chrono::milliseconds(2));
	}
	return true;
}

using GroupRange = PipelineCodeStore::GroupRange;
// `ranges` (sorted by library, first) with groups added/merged: adjacent or overlapping ranges of
// a library become one.
inline void MergeRanges(std::vector<GroupRange>& ranges) {
	std::sort(ranges.begin(), ranges.end(), [](const GroupRange& a, const GroupRange& b) {
		return a.library != b.library ? a.library < b.library : a.first < b.first;
	});
	std::vector<GroupRange> merged;
	for (const auto& r: ranges) {
		if (!merged.empty() && merged.back().library == r.library &&
		    (r.first <= merged.back().last || r.first - merged.back().last == 1)) {
			merged.back().last = std::max(merged.back().last, r.last);
		} else {
			merged.push_back(r);
		}
	}
	ranges = std::move(merged);
}
[[nodiscard]] inline bool RangesCover(const std::vector<GroupRange>& sorted, uint32_t library,
                                      uint32_t group) {
	const auto it = std::upper_bound(sorted.begin(), sorted.end(), GroupRange {library, group, 0},
	                                 [](const GroupRange& a, const GroupRange& b) {
		                                 return a.library != b.library ? a.library < b.library
		                                                               : a.first < b.first;
	                                 });
	if (it == sorted.begin()) return false;
	const auto& r = *(it - 1);
	return r.library == library && group >= r.first && group <= r.last;
}

// Turns decoded groups into code store entries. OnGroup / OnProgress may be called from several
// host threads at once.
class Collector {
public:
	// 1 added, 0 already stored, -1 could not be stored (cap, I/O)
	using AddFn   = std::function<int(const Hash128&, uint64_t prefix, std::span<const uint8_t>)>;
	using ClockFn = std::function<uint64_t()>;

	// `wanted`: code hash -> expected bytes, for the listed runtime codes not in the store yet.
	// `resume`: the groups an earlier, stopped pass already saw (a store mark): they are not split
	// again, and the time budget starts at the first group after them (the host still decodes the
	// skipped ones, which only the watchdog bounds).
	Collector(WantedMap wanted, AddFn add, UnitSink sink, uint64_t max_s,
	          const Ue5Group::Layout* layout = nullptr, ClockFn clock = SteadyMs,
	          std::vector<GroupRange> resume = {})
	    : m_wanted(std::move(wanted)), m_add(std::move(add)), m_sink(std::move(sink)),
	      m_max_ms(max_s * 1000u), m_layout(layout), m_clock(std::move(clock)),
	      m_resume(std::move(resume)) {
		MergeRanges(m_resume);
		m_start = m_clock();
	}

	// True = stop (all wanted found, the time budget is over, a cancel or an abandon).
	bool OnGroup(const GroupView& g) {
		if (m_abandoned.load(std::memory_order_acquire)) return true;
		const bool       skip = !m_resume.empty() && RangesCover(m_resume, g.library, g.group);
		Ue5Group::Stats  stats;
		const auto       shaders = skip ? std::vector<Ue5Group::Shader> {}
		                                : Ue5Group::Split(g.bytes, g.frequencies, &stats, m_layout);
		std::scoped_lock lock(m_mutex);
		if (m_abandoned.load(std::memory_order_relaxed)) return true;
		if (skip) {
			m_resume_skipped++;
			return m_stop.load(std::memory_order_relaxed) || Cancelled();
		}
		if (!m_resume.empty() && !m_resume_started) {
			m_resume_started = true;
			m_start          = m_clock(); // the budget counts new groups only
		}
		m_new_groups.push_back({g.library, g.group, g.group});
		m_groups++;
		m_split.markers += stats.markers;
		m_split.out_of_bounds += stats.out_of_bounds;
		m_split.bad_size += stats.bad_size;
		m_split.size_mismatch += stats.size_mismatch;
		for (const auto& shader: shaders) {
			const auto it = m_wanted.find(shader.hash);
			if (it == m_wanted.end()) continue;
			if (it->second != shader.size) {
				m_size_conflicts++;
				continue;
			}
			const int added = m_add(shader.hash, shader.prefix, shader.code);
			if (added < 0) {
				m_add_failed++;
				continue;
			}
			if (added > 0) {
				m_added++;
				m_added_bytes += shader.size;
			}
			m_wanted.erase(it);
		}
		if (m_wanted.empty()) return true;
		if (m_max_ms != 0 && m_clock() - m_start >= m_max_ms) {
			m_timed_out = true;
			return true;
		}
		return m_stop.load(std::memory_order_relaxed) || Cancelled();
	}

	// The next group callback answers "stop" (the watchdog's way into the host's -3 path).
	void RequestStop() { m_stop.store(true, std::memory_order_release); }
	// After this returns the collector never touches the store again (a host call that outlived
	// its watchdog may still deliver groups).
	void Abandon() {
		std::scoped_lock lock(m_mutex);
		m_abandoned.store(true, std::memory_order_release);
	}
	[[nodiscard]] bool Abandoned() const { return m_abandoned.load(std::memory_order_acquire); }
	// The groups this pass saw plus the ones it resumed past, merged.
	[[nodiscard]] std::vector<GroupRange> Seen() const {
		std::scoped_lock lock(m_mutex);
		auto             all = m_resume;
		all.insert(all.end(), m_new_groups.begin(), m_new_groups.end());
		MergeRanges(all);
		return all;
	}
	[[nodiscard]] uint64_t ResumeSkipped() const {
		std::scoped_lock lock(m_mutex);
		return m_resume_skipped;
	}

	void OnProgress(uint32_t done, uint32_t total) {
		std::scoped_lock lock(m_progress_mutex);
		if (total == 0) return;
		if (m_total == 0) {
			m_total = total;
			m_sink.add_total(total);
		}
		done = std::min(done, m_total);
		if (done > m_done) {
			m_sink.add_done(done - m_done);
			m_done = done;
		}
	}

	// Completes the card unit whatever happened (idempotent).
	void Finish() {
		std::scoped_lock lock(m_progress_mutex);
		if (m_total != 0 && m_done < m_total) {
			m_sink.add_done(m_total - m_done);
			m_done = m_total;
		}
	}

	[[nodiscard]] size_t Remaining() const {
		std::scoped_lock l(m_mutex);
		return m_wanted.size();
	}
	[[nodiscard]] bool TimedOut() const {
		std::scoped_lock l(m_mutex);
		return m_timed_out;
	}
	[[nodiscard]] uint64_t Added() const {
		std::scoped_lock l(m_mutex);
		return m_added;
	}
	[[nodiscard]] uint64_t AddedBytes() const {
		std::scoped_lock l(m_mutex);
		return m_added_bytes;
	}
	[[nodiscard]] uint64_t AddFailed() const {
		std::scoped_lock l(m_mutex);
		return m_add_failed;
	}
	[[nodiscard]] uint64_t SizeConflicts() const {
		std::scoped_lock l(m_mutex);
		return m_size_conflicts;
	}
	[[nodiscard]] Ue5Group::Stats SplitStats() const {
		std::scoped_lock l(m_mutex);
		return m_split;
	}
	[[nodiscard]] uint64_t ElapsedS() const { return (m_clock() - m_start) / 1000u; }

	// The I8 line. `listed`: distinct listed runtime codes (M); K = M - what is still wanted.
	[[nodiscard]] std::string Summary(const HostResult& r, uint64_t listed) const {
		const uint64_t found = listed - std::min<uint64_t>(listed, Remaining());
		char           text[320];
		std::snprintf(
		    text, sizeof(text),
		    "UE5 code: %u groups (%.1f MB) decoded in %llu ms, %u failed, %u skipped; listed "
		    "runtime shaders found %llu of %llu; store +%llu codes (%.1f MB)",
		    r.decoded, static_cast<double>(r.raw_bytes) / 1048576.0,
		    static_cast<unsigned long long>(r.ms), r.failed, r.skipped,
		    static_cast<unsigned long long>(found), static_cast<unsigned long long>(listed),
		    static_cast<unsigned long long>(Added()),
		    static_cast<double>(AddedBytes()) / 1048576.0);
		std::string line = text;
		if (ResumeSkipped() != 0)
			line += "; resumed past " + std::to_string(ResumeSkipped()) + " groups";
		if (TimedOut()) line += "; stopped after " + std::to_string(ElapsedS()) + " s";
		return line;
	}

private:
	mutable std::mutex      m_mutex, m_progress_mutex;
	WantedMap               m_wanted;
	AddFn                   m_add;
	UnitSink                m_sink;
	uint64_t                m_max_ms, m_start = 0;
	const Ue5Group::Layout* m_layout;
	ClockFn                 m_clock;
	uint64_t m_groups = 0, m_added = 0, m_added_bytes = 0, m_add_failed = 0, m_size_conflicts = 0;
	bool     m_timed_out = false;
	std::vector<GroupRange> m_resume, m_new_groups;
	bool                    m_resume_started = false;
	uint64_t                m_resume_skipped = 0;
	std::atomic<bool>       m_stop {false}, m_abandoned {false};
	Ue5Group::Stats m_split;
	uint32_t        m_total = 0, m_done = 0;
};

// ---- the guest_ready flow ----
enum class Outcome { NotDeferred, Extracted, FpMismatch, HostFailed, NeverReady };

[[nodiscard]] inline Outcome ClassifyHost(int code) {
	// -3 is our own cancel (all found, or the time budget); the host reports nothing else for it.
	return code == 0 || code == -3 ? Outcome::Extracted : Outcome::HostFailed;
}

struct Flow {
	bool deferred          = false; // the two list consumers wait for this call
	bool never_ready       = false; // the AgcCreateShader safety net: no extraction
	bool prepare           = false; // KYTY_PIPELINE_LIST_PREPARE: run the prepare, never returns
	bool wait_before_entry = false;
};
struct Hooks {
	std::function<bool()>             fingerprint_ok;
	std::function<int()>              extract; // the host code
	std::function<void()>             flush_store;
	std::function<void()>             finish_unit;
	std::function<void()>             on_main; // re-resolve, then the two list consumers
	std::function<void()>             run_prepare;
	std::function<void()>             wait_entry;
	std::function<void(Outcome, int)> report;
};

// The order of guest_ready (ue5launch section 2). Never calls the archive flush, the driver cache
// save or the stamp: the only serialize of the launch phase stays in the last replay worker.
inline Outcome RunGuestReady(const Flow& f, const Hooks& h) {
	Outcome out  = Outcome::NotDeferred;
	int     code = 0;
	if (f.deferred) {
		if (f.never_ready) {
			out = Outcome::NeverReady;
		} else if (!h.fingerprint_ok()) {
			out = Outcome::FpMismatch;
		} else {
			code = h.extract();
			out  = ClassifyHost(code);
			h.flush_store();
		}
		h.finish_unit(); // every path completes the card unit
		if (h.report) h.report(out, code);
		h.on_main();
	}
	if (Cancelled()) return out; // the process is ending: neither the prepare nor the wait
	if (f.prepare) h.run_prepare();
	if (f.wait_before_entry) h.wait_entry();
	return out;
}

// Set while the two list consumers wait for guest_ready (read by agc.cpp on every shader create).
inline std::atomic<bool>  g_deferred {false};
[[nodiscard]] inline bool DeferredPending() {
	return g_deferred.load(std::memory_order_relaxed);
}

// The listed `<runtime>` codes (front and back halves) of `file` and, of those, the ones of a
// pending shader the store does not hold yet (hash -> expected bytes). `key` (optional): a hash of
// ALL listed codes (and the layout word), the identity of the wanted set a search mark refers to.
inline void CollectWanted(const PipelineList::File& file, const PipelineList::Resolved& resolved,
                          const std::array<uint32_t, 3>& sizes, const PipelineCodeStore& store,
                          WantedMap& wanted, uint64_t& listed, uint64_t* key = nullptr,
                          const std::string& layout = {}) {
	std::unordered_set<Hash128, HashOf> seen;
	const auto                          runtime_file = PipelineList::RuntimeFileOf(file);
	for (size_t i = 0; i < file.shaders.size(); i++) {
		const auto& s = file.shaders[i];
		if (runtime_file == PipelineList::CodeFile::Executable || s.file != runtime_file ||
		    s.input_info.size() != sizes[static_cast<size_t>(s.kind)])
			continue;
		const auto note = [&](const Hash128& hash, uint32_t words) {
			if (words == 0 || !seen.insert(hash).second) return;
			if (resolved.pending[i] && store.Size(hash) != words * 4u) wanted[hash] = words * 4u;
		};
		note({s.code_hash_low, s.code_hash_high}, s.code_size);
		note({s.back_hash_low, s.back_hash_high}, s.back_code_size);
	}
	listed = seen.size();
	if (key != nullptr) {
		std::vector<std::pair<uint64_t, uint64_t>> all;
		for (const auto& h: seen)
			all.emplace_back(h.low, h.high);
		std::sort(all.begin(), all.end());
		std::vector<uint64_t> flat;
		for (const auto& [low, high]: all) {
			flat.push_back(low);
			flat.push_back(high);
		}
		const auto seed = Fnv1a64({reinterpret_cast<const uint8_t*>(layout.data()), layout.size()});
		*key            = XXH3_64bits_withSeed(flat.data(), flat.size() * sizeof(uint64_t), seed);
		if (*key == 0) *key = 1; // 0 means "no marks"
	}
}

// What the store already knows about this wanted set (see PipelineCodeStore's marks).
struct SearchInfo {
	bool     searched   = false; // a complete pass for this wanted set ran
	bool     store_full = false;
	bool     locked     = false; // another process owns the store
	uint64_t missing    = 0;     // listed codes the store lacks
};
[[nodiscard]] inline SearchInfo SearchInfoOf(const PipelineCodeStore& store, uint64_t key,
                                             uint64_t missing) {
	SearchInfo info;
	const auto mark = store.Mark();
	info.searched =
	    key != 0 && mark.state == PipelineCodeStore::SearchMark::Complete && mark.key == key;
	info.store_full = store.Full();
	info.locked     = store.LockedByOther();
	info.missing    = missing;
	return info;
}

// What the constructor does with a list that has `<runtime>` shaders and a store (ue5launch
// section 4). `line` is the log line to print (empty: none).
enum class CtorPlan { Skip, Off, Defer };
[[nodiscard]] inline CtorPlan PlanAtCtor(const Config& c, const PipelineList::ResolveStats& stats,
                                         bool host_has_reader, uint64_t listed, std::string& line,
                                         const SearchInfo& search = {}) {
	line.clear();
	if (stats.runtime_pending == 0) {
		if (stats.runtime_shaders != 0) line = SkipLine(listed);
		return CtorPlan::Skip;
	}
	if (const auto reason = OffReason(c, host_has_reader); !reason.empty()) {
		line = OffLine(reason);
		return CtorPlan::Off;
	}
	// The listed codes the store lacks will not appear by reading the library again:
	if (search.locked) {
		line = StoreLockedLine();
		return CtorPlan::Skip;
	}
	if (search.searched) {
		line = SearchedLine(search.missing);
		return CtorPlan::Skip;
	}
	if (search.store_full) {
		line = StoreFullLine(search.missing);
		return CtorPlan::Skip;
	}
	return CtorPlan::Defer;
}

// The host call's watchdog return code (not the host's own).
inline constexpr int WatchdogCode = -4;

// One guest_ready extraction: owns the collector and the host result, and builds the hooks of
// RunGuestReady (the caller adds the renderer side: on_main, prepare, wait).
class Session {
public:
	using LogFn = std::function<void(const std::string&)>;

	// `key`: the wanted set's identity for the store's search marks (0 = no marks).
	Session(Config config, PipelineCodeStore* store, WantedMap wanted, uint64_t listed, LogFn log,
	        UnitSink sink = {}, Collector::ClockFn clock = SteadyMs, uint64_t key = 0)
	    : m_config(std::move(config)), m_store(store), m_wanted(std::move(wanted)),
	      m_listed(listed), m_log(std::move(log)), m_sink(std::move(sink)),
	      m_clock(std::move(clock)), m_key(key) {}

	// The watchdog on the host call: after `hard_ms` without the call returning the next group
	// callback answers "stop"; `grace_ms` later the call is abandoned (it keeps running on its own
	// thread, delivering into nothing). 0 hard_ms = derived from KYTY_UE5_CODE_MAX_S, none for 0.
	void SetWatchdog(uint64_t hard_ms, uint64_t grace_ms) {
		m_hard_ms  = hard_ms;
		m_grace_ms = grace_ms;
	}

	[[nodiscard]] Hooks MakeHooks(std::function<void()> on_main, std::function<void()> run_prepare,
	                              std::function<void()> wait_entry) {
		Hooks h;
		h.fingerprint_ok = [this] { return FingerprintMatches(m_config.decoder, m_config.fp); };
		h.extract        = [this] { return Extract(); };
		h.flush_store    = [this] { Flush(); };
		h.finish_unit    = [this] {
			if (m_collector != nullptr) m_collector->Finish();
		};
		h.report = [this](Outcome outcome, int code) {
			switch (outcome) {
				case Outcome::FpMismatch: m_log(FpMismatchLine()); break;
				case Outcome::HostFailed:
					m_log(code == WatchdogCode ? WatchdogLine(m_waited_ms / 1000u)
					                           : HostFailedLine(code));
					break;
				case Outcome::NeverReady: m_log(NeverReadyLine()); break;
				default: break;
			}
		};
		h.on_main     = std::move(on_main);
		h.run_prepare = std::move(run_prepare);
		h.wait_entry  = std::move(wait_entry);
		return h;
	}

	[[nodiscard]] const Collector* GetCollector() const { return m_collector.get(); }

private:
	struct Shared {
		std::mutex              mutex;
		std::condition_variable cv;
		bool                    done = false;
		HostResult              result;
	};

	int Extract() {
		const auto extract = GetSeam().extract; // a copy: the host call may outlive this session
		if (!extract) return -1;
		std::vector<GroupRange> resume;
		if (m_key != 0) {
			const auto mark = m_store->Mark();
			if (mark.state == PipelineCodeStore::SearchMark::Partial && mark.key == m_key)
				resume = mark.done;
		}
		const bool resumed = !resume.empty();
		m_collector        = std::make_shared<Collector>(
		    m_wanted,
		    [this](const Hash128& hash, uint64_t prefix, std::span<const uint8_t> code) {
			    if (m_store->Add(hash, prefix, code)) return 1;
			    return m_store->Size(hash) != 0 ? 0 : -1;
		    },
		    m_sink, m_config.max_s, Ue5Group::FindLayout(m_config.layout.c_str()), m_clock,
		    std::move(resume));
		const auto collector = m_collector;
		const auto shared    = std::make_shared<Shared>();
		const auto decoder   = m_config.decoder;
		const auto abi       = AbiId(m_config.abi);
		g_active.fetch_add(1, std::memory_order_acq_rel);
		// On a thread of its own so the watchdog can give up on a host call that never returns.
		std::thread([extract, collector, shared, decoder, abi] {
			auto result = extract(
			    decoder, abi,
			    [collector](const GroupView& view) { return collector->OnGroup(view); },
			    [collector](uint32_t done, uint32_t total) { collector->OnProgress(done, total); });
			{
				std::scoped_lock lock(shared->mutex);
				shared->result = result;
				shared->done   = true;
			}
			shared->cv.notify_all();
			g_active.fetch_sub(1, std::memory_order_acq_rel);
		}).detach();

		const uint64_t max_ms = m_config.max_s * 1000u;
		uint64_t       hard   = m_hard_ms;
		if (hard == 0 && max_ms != 0) hard = (resumed ? 2 * max_ms : max_ms) + 5000u;
		const auto begin     = SteadyMs();
		uint64_t   stop_at   = 0; // when the stop was asked for
		bool       abandoned = false;
		for (;;) {
			std::unique_lock lock(shared->mutex);
			if (shared->cv.wait_for(lock, std::chrono::milliseconds(10),
			                        [&] { return shared->done; })) {
				m_result = shared->result;
				break;
			}
			lock.unlock();
			const auto now = SteadyMs();
			if (stop_at == 0 && ((hard != 0 && now - begin >= hard) || Cancelled())) {
				stop_at = now;
				collector->RequestStop();
			} else if (stop_at != 0 &&
			           now - stop_at >= (Cancelled() ? m_cancel_grace_ms : m_grace_ms)) {
				collector->Abandon();
				m_result.code = WatchdogCode;
				abandoned     = true;
				break;
			}
		}
		m_waited_ms = SteadyMs() - begin;
		(void)abandoned;
		return m_result.code;
	}

	void WriteSearchMark() {
		if (m_key == 0 || m_collector == nullptr || m_store == nullptr) return;
		if (m_collector->Remaining() == 0) return; // the store has everything: nothing to remember
		PipelineCodeStore::SearchMark mark;
		mark.key       = m_key;
		const int code = m_result.code;
		if (code == 0 && !m_collector->TimedOut() && m_result.failed == 0 &&
		    m_result.skipped == 0) {
			mark.state = PipelineCodeStore::SearchMark::Complete;
		} else if (code == 0 || code == -3 || code == WatchdogCode) {
			mark.state = PipelineCodeStore::SearchMark::Partial;
			mark.done  = m_collector->Seen();
		} else {
			return; // -1 / -2: the library could not be read at all
		}
		if (!m_store->WriteMark(mark) && mark.state == PipelineCodeStore::SearchMark::Partial)
			m_log("UE5 code: the search progress could not be saved: the next start begins again");
	}

	void Flush() {
		if (m_collector == nullptr) return;
		WriteSearchMark();
		if (!m_store->Flush()) m_log("UE5 code: the code store could not be flushed");
		if (m_store->Full()) m_log("UE5 code: the code store is full (64 MB)");
		if (m_result.code != 0 && m_result.code != -3) return; // the report hook says why
		m_log(m_collector->Summary(m_result, m_listed));
		const auto split = m_collector->SplitStats();
		if (split.out_of_bounds + split.bad_size + split.size_mismatch != 0 ||
		    m_collector->SizeConflicts() != 0 || m_collector->AddFailed() != 0) {
			m_log("UE5 code: split: " + std::to_string(split.out_of_bounds) +
			      " codes outside their group, " + std::to_string(split.bad_size) +
			      " with a bad size, " + std::to_string(split.size_mismatch) +
			      " size mismatches, " + std::to_string(m_collector->SizeConflicts()) +
			      " size conflicts, " + std::to_string(m_collector->AddFailed()) + " not stored");
		}
		const uint64_t found = m_listed - std::min<uint64_t>(m_listed, m_collector->Remaining());
		if (m_collector->TimedOut() && m_collector->Remaining() != 0)
			m_log(LowFoundLine(found, m_listed, true, m_collector->ElapsedS()));
		else if (LowFound(found, m_listed))
			m_log(LowFoundLine(found, m_listed));
	}

	Config                     m_config;
	PipelineCodeStore*         m_store;
	WantedMap                  m_wanted;
	uint64_t                   m_listed;
	LogFn                      m_log;
	UnitSink                   m_sink;
	Collector::ClockFn         m_clock;
	uint64_t                   m_key;
	uint64_t                   m_hard_ms = 0, m_grace_ms = 10000, m_cancel_grace_ms = 3000;
	uint64_t                   m_waited_ms = 0;
	std::shared_ptr<Collector> m_collector;
	HostResult                 m_result;
};

// ---- the launch phase around guest_ready, without a renderer ----

// The list and the store of a UE5 start that has to extract (the PipelineCache keeps one while
// the two list consumers wait).
struct ListWarm {
	Config                              cfg;
	PipelineCodeStore                   store;
	std::unique_ptr<PipelineList::File> file_holder;
	std::shared_ptr<void>               source; // what read / read_image use, kept alive
	PipelineList::ReadCodeFn            read;
	PipelineList::ReadImageFn           read_image;
	std::array<uint32_t, 3>             sizes {};
	WantedMap                           wanted;     // listed codes the store lacks: hash -> bytes
	uint64_t                            listed = 0; // distinct listed runtime codes
	uint64_t                            key    = 0; // identity of the wanted set (search marks)
};

using LogLineFn = std::function<void(const std::string&)>;
using PathFn    = std::function<std::filesystem::path(const std::string&)>;

struct CtorInput {
	PipelineList::File*       file = nullptr;
	std::array<uint32_t, 3>   sizes {};
	PipelineList::ReadCodeFn  read;
	PipelineList::ReadImageFn read_image;
	std::string               title_id, version;
	bool                      host_has_reader = false;
	GetEnvFn                  getenv;
	LogLineFn                 log;
	PathFn                    to_path; // UTF-8 text to a path (default: the text as UTF-8)
};
struct CtorResult {
	PipelineList::Resolved     resolved;
	PipelineList::ResolveStats real; // the stats of the first Resolve, pending included
	bool                       ignored   = false; // the 50 % rule dropped the list (logged)
	bool                       defer     = false; // the consumers wait for guest_ready
	bool                       keep_file = false; // BuildListState copies levels/pipelines
	CtorPlan                   plan      = CtorPlan::Skip; // what the store/host/marks decided
	std::unique_ptr<ListWarm>  warm;                       // set when defer
};

// The first Resolve of InitializePipelineList and what follows from it for a UE5 start: the code
// store, the plan (skip / off / defer) and its log lines, the 50 % rule. Shared by the renderer and
// the tests. The caller keeps `file` (and, when defer, moves it into warm->file_holder).
[[nodiscard]] inline CtorResult ResolveAtCtor(const CtorInput& in) {
	CtorResult                r;
	auto&                     file = *in.file;
	std::unique_ptr<ListWarm> warm;
	if (PipelineList::RuntimeFileOf(file) != PipelineList::CodeFile::Executable) {
		auto config = ParseEnv(in.getenv);
		if (!config.store_path.empty()) {
			warm      = std::make_unique<ListWarm>();
			warm->cfg = std::move(config);
			const auto path =
			    in.to_path ? in.to_path(warm->cfg.store_path)
			               : std::filesystem::path(std::u8string(warm->cfg.store_path.begin(),
			                                                     warm->cfg.store_path.end()));
			if (!warm->store.Open(path, in.title_id, in.version, warm->cfg.fp)) {
				in.log("UE5 code: the code store " + warm->cfg.store_path +
				       " cannot be opened: the list's runtime shaders stay pending");
				warm.reset();
			} else {
				in.log(StoreHealthLine(warm->store));
			}
		}
	}
	PipelineList::ResolveStats stats;
	r.resolved =
	    PipelineList::Resolve(file, in.sizes, in.read, in.read_image, stats,
	                          warm != nullptr ? warm->store.ReadFn() : PipelineList::ReadHashFn {});
	r.real = stats;
	if (warm != nullptr) {
		CollectWanted(file, r.resolved, in.sizes, warm->store, warm->wanted, warm->listed,
		              &warm->key, warm->cfg.layout);
		std::string line;
		const auto  plan = PlanAtCtor(warm->cfg, stats, in.host_has_reader, warm->listed, line,
		                              SearchInfoOf(warm->store, warm->key, warm->wanted.size()));
		if (!line.empty()) in.log(line);
		r.plan = plan;
		if (plan == CtorPlan::Defer)
			r.defer = true;
		else
			warm.reset();
	}
	// Pending code that nobody will extract is not found: it counts against the 50 % rule.
	if (!r.defer) stats.runtime_pending = 0;
	if (!PipelineList::MostlyMatches(stats)) {
		in.log(IgnoredLine(r.real, r.defer, file.info.producer));
		r.ignored = true;
		r.defer   = false;
		return r;
	}
	r.keep_file = r.defer;
	if (r.defer) {
		warm->sizes      = in.sizes;
		warm->read       = in.read;
		warm->read_image = in.read_image;
		r.warm           = std::move(warm);
	}
	return r;
}

// The deferred start is in place (the list built from what the store holds, `warm` adopted its
// file and source): tells so and makes the list consumers wait.
inline void ArmDeferred(const ListWarm& warm, const LogLineFn& log) {
	log("UE5 code: " + std::to_string(warm.wanted.size()) + " of " + std::to_string(warm.listed) +
	    " listed runtime shaders are not in the store yet: the game's shader library is read when "
	    "it is ready");
	g_deferred.store(true, std::memory_order_release);
}

// BuildListState's hand-over of the list's tables: copied while the file is kept for the second
// Resolve, moved otherwise.
template <typename Levels, typename Pipelines>
inline void TakeListTables(PipelineList::File& file, bool keep_file, Levels& levels,
                           Pipelines& pipelines) {
	if (keep_file) {
		levels    = file.levels;
		pipelines = file.pipelines;
	} else {
		levels    = std::move(file.levels);
		pipelines = std::move(file.pipelines);
	}
}

// The second Resolve with the code the store holds now. False: the 50 % rule drops the list (the
// pending codes count as not found). `resolved` / `real` are the second pass's.
[[nodiscard]] inline bool ResolveAfterExtraction(ListWarm& warm, PipelineList::Resolved& resolved,
                                                 PipelineList::ResolveStats& real,
                                                 const LogLineFn&            log) {
	PipelineList::ResolveStats stats;
	resolved = PipelineList::Resolve(*warm.file_holder, warm.sizes, warm.read, warm.read_image,
	                                 stats, warm.store.ReadFn());
	real     = stats;
	stats.runtime_pending = 0; // what the store still lacks is not found
	if (!PipelineList::MostlyMatches(stats)) {
		log(IgnoredLine(real, false, warm.file_holder->info.producer));
		return false;
	}
	return true;
}

// The renderer side of the launch phase.
struct LaunchOps {
	// Builds the list state from the second Resolve (BuildListState, keep_file = false).
	std::function<void(PipelineList::Resolved&&, const PipelineList::ResolveStats& real)>
	    build_list;
	std::function<void(const PipelineList::ResolveStats& real)>
	                      drop_list;       // the second Resolve dropped the list
	std::function<void()> start_consumers; // InitializeShaderPrecompile + InitializePipelineJournal
	std::function<void()> run_prepare;
	std::function<void()> wait_entry;
	std::function<void()> release; // after the main-thread part: drop the warm state
	LogLineFn             log;
};

// guest_ready (or the AgcCreateShader safety net, never_ready): extraction, then the main-thread
// part, then the optional prepare and wait. `warm` null = nothing deferred (only the wait).
inline Outcome RunLaunch(ListWarm* warm, const Config& config, bool never_ready, bool prepare,
                         const LaunchOps& ops, UnitSink sink = {},
                         Collector::ClockFn clock = SteadyMs, uint64_t hard_ms = 0,
                         uint64_t grace_ms = 10000) {
	Flow flow;
	flow.deferred          = warm != nullptr;
	flow.never_ready       = never_ready;
	flow.prepare           = flow.deferred && prepare;
	flow.wait_before_entry = !never_ready && config.wait_before_entry;
	if (!flow.deferred && !flow.wait_before_entry) return Outcome::NotDeferred;
	Session session(config, warm != nullptr ? &warm->store : nullptr,
	                warm != nullptr ? warm->wanted : WantedMap {},
	                warm != nullptr ? warm->listed : 0, ops.log, std::move(sink), std::move(clock),
	                warm != nullptr ? warm->key : 0);
	session.SetWatchdog(hard_ms, grace_ms);
	auto hooks = session.MakeHooks(
	    [&] {
		    if (Cancelled()) { // the window was closed: nothing is started, nobody is waited for
			    g_deferred.store(false, std::memory_order_release);
			    ops.release();
			    return;
		    }
		    RunOnMain([&] {
			    PipelineList::Resolved     resolved;
			    PipelineList::ResolveStats real;
			    if (ResolveAfterExtraction(*warm, resolved, real, ops.log))
				    ops.build_list(std::move(resolved), real);
			    else
				    ops.drop_list(real);
			    ops.start_consumers();
			    g_deferred.store(false, std::memory_order_release);
		    });
		    ops.release();
	    },
	    ops.run_prepare, ops.wait_entry);
	return RunGuestReady(flow, hooks);
}

// The pre-entry wait of KYTY_LAUNCH_WAIT_BEFORE_ENTRY, bounded by ONE deadline: when it gives up
// the hold is released, so the first draw does not wait again for the same limit.
struct WaitResult {
	uint64_t waited_ms = 0;
	bool     gave_up   = false;
};
inline WaitResult WaitForLaunchHold(const std::function<bool()>& done, uint64_t max_s,
                                    const std::function<uint64_t()>&     now_ms,
                                    const std::function<void(uint64_t)>& sleep_ms,
                                    const std::function<void()>&         release) {
	WaitResult r;
	const auto begin = now_ms();
	while (!done()) {
		if (now_ms() - begin >= max_s * 1000u || Cancelled()) {
			r.gave_up = true;
			release();
			break;
		}
		sleep_ms(10);
	}
	r.waited_ms = now_ms() - begin;
	return r;
}

} // namespace Libs::Graphics::Ue5CodeWarm

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UE5CODEWARM_H_
