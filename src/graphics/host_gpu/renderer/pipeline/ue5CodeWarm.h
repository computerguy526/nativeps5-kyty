#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UE5CODEWARM_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UE5CODEWARM_H_

#include "graphics/compileProgress.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCodeStore.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"
#include "graphics/host_gpu/renderer/pipeline/ue5Group.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
[[nodiscard]] inline std::string LowFoundLine(uint64_t found, uint64_t listed) {
	return "UE5 code: WARNING only " + std::to_string(found) + " of " + std::to_string(listed) +
	       " listed runtime shaders are in the game's shader library";
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

// Turns decoded groups into code store entries. OnGroup / OnProgress may be called from several
// host threads at once.
class Collector {
public:
	// 1 added, 0 already stored, -1 could not be stored (cap, I/O)
	using AddFn   = std::function<int(const Hash128&, uint64_t prefix, std::span<const uint8_t>)>;
	using ClockFn = std::function<uint64_t()>;

	// `wanted`: code hash -> expected bytes, for the listed runtime codes not in the store yet.
	Collector(WantedMap wanted, AddFn add, UnitSink sink, uint64_t max_s,
	          const Ue5Group::Layout* layout = nullptr, ClockFn clock = SteadyMs)
	    : m_wanted(std::move(wanted)), m_add(std::move(add)), m_sink(std::move(sink)),
	      m_max_ms(max_s * 1000u), m_layout(layout), m_clock(std::move(clock)) {
		m_start = m_clock();
	}

	// True = stop (all wanted found, or the time budget is over).
	bool OnGroup(const GroupView& g) {
		Ue5Group::Stats  stats;
		const auto       shaders = Ue5Group::Split(g.bytes, g.frequencies, &stats, m_layout);
		std::scoped_lock lock(m_mutex);
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
		return false;
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
	if (f.prepare) h.run_prepare();
	if (f.wait_before_entry) h.wait_entry();
	return out;
}

// The listed `<runtime>` codes (front and back halves) of `file` and, of those, the ones of a
// pending shader the store does not hold yet (hash -> expected bytes).
inline void CollectWanted(const PipelineList::File& file, const PipelineList::Resolved& resolved,
                          const std::array<uint32_t, 3>& sizes, const PipelineCodeStore& store,
                          WantedMap& wanted, uint64_t& listed) {
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
}

// What the constructor does with a list that has `<runtime>` shaders and a store (ue5launch
// section 4). `line` is the log line to print (empty: none).
enum class CtorPlan { Skip, Off, Defer };
[[nodiscard]] inline CtorPlan PlanAtCtor(const Config& c, const PipelineList::ResolveStats& stats,
                                         bool host_has_reader, uint64_t listed, std::string& line) {
	line.clear();
	if (stats.runtime_pending == 0) {
		if (stats.runtime_shaders != 0) line = SkipLine(listed);
		return CtorPlan::Skip;
	}
	if (const auto reason = OffReason(c, host_has_reader); !reason.empty()) {
		line = OffLine(reason);
		return CtorPlan::Off;
	}
	return CtorPlan::Defer;
}

// One guest_ready extraction: owns the collector and the host result, and builds the hooks of
// RunGuestReady (the caller adds the renderer side: on_main, prepare, wait).
class Session {
public:
	using LogFn = std::function<void(const std::string&)>;

	Session(Config config, PipelineCodeStore* store, WantedMap wanted, uint64_t listed, LogFn log,
	        UnitSink sink = {}, Collector::ClockFn clock = SteadyMs)
	    : m_config(std::move(config)), m_store(store), m_wanted(std::move(wanted)),
	      m_listed(listed), m_log(std::move(log)), m_sink(std::move(sink)),
	      m_clock(std::move(clock)) {}

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
				case Outcome::HostFailed: m_log(HostFailedLine(code)); break;
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
	int Extract() {
		const auto& extract = GetSeam().extract;
		if (!extract) return -1;
		m_collector = std::make_unique<Collector>(
		    m_wanted,
		    [this](const Hash128& hash, uint64_t prefix, std::span<const uint8_t> code) {
			    if (m_store->Add(hash, prefix, code)) return 1;
			    return m_store->Size(hash) != 0 ? 0 : -1;
		    },
		    m_sink, m_config.max_s, Ue5Group::FindLayout(m_config.layout.c_str()), m_clock);
		auto* c  = m_collector.get();
		m_result = extract(
		    m_config.decoder, AbiId(m_config.abi),
		    [c](const GroupView& view) { return c->OnGroup(view); },
		    [c](uint32_t done, uint32_t total) { c->OnProgress(done, total); });
		return m_result.code;
	}

	void Flush() {
		if (m_collector == nullptr) return;
		if (!m_store->Flush()) m_log("UE5 code: the code store could not be flushed");
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
		if (LowFound(found, m_listed)) m_log(LowFoundLine(found, m_listed));
	}

	Config                     m_config;
	PipelineCodeStore*         m_store;
	WantedMap                  m_wanted;
	uint64_t                   m_listed;
	LogFn                      m_log;
	UnitSink                   m_sink;
	Collector::ClockFn         m_clock;
	std::unique_ptr<Collector> m_collector;
	HostResult                 m_result;
};

// Set while the two list consumers wait for guest_ready (read by agc.cpp on every shader create).
inline std::atomic<bool>  g_deferred {false};
[[nodiscard]] inline bool DeferredPending() {
	return g_deferred.load(std::memory_order_relaxed);
}

} // namespace Libs::Graphics::Ue5CodeWarm

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UE5CODEWARM_H_
