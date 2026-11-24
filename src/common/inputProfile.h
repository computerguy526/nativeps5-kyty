#ifndef EMULATOR_SRC_COMMON_INPUTPROFILE_H_
#define EMULATOR_SRC_COMMON_INPUTPROFILE_H_

// nps5 KYTY_INPUT_PROFILE=1: LOCAL-ONLY research capture of the values real draws and dispatches
// read (default OFF; NOT part of the flight recorder, which never holds game bytes).
//
// Purpose: realistic inputs for the offline shader validator (work/mslnative/autoval). Per guest
// shader (64-bit shader hash + stage) and per bound input slot it keeps VALUE STATISTICS, one
// DwordStat per dword position of the first KYTY_INPUT_PROFILE_STAT_KB KiB (default 4: value class
// bits, unsigned min/max, float min/max, change count), one DwordStat per 64-dword block of the
// rest of the read window (KYTY_INPUT_PROFILE_KB, default 64, max 64), plus at most
// KYTY_INPUT_PROFILE_RAW (default 2) distinct raw snapshots of the first KYTY_INPUT_PROFILE_RAW_KB
// KiB (default 16) of each slot.
// Slots: 'U' shader data (user SGPR dwords as Kyty uploads/pushes them), 'R' flattened SRT,
// 'B' read-only guest buffer (V#-described; the read window above),
// 'W' written/atomic buffer (descriptor only, never read), 'V' vertex buffer (statistics folded by
// stride over the first 64 vertices), 'T' texture T# and 'S' sampler S# words (up to 4 distinct).
//
// Sampling (cheap enough for one normal title run): the first KYTY_INPUT_PROFILE_FIRST (8) draws
// of each shader, then at most one per KYTY_INPUT_PROFILE_INTERVAL_MS (1000) up to
// KYTY_INPUT_PROFILE_MAX (64) samples. Guest memory is read on the CPU at bind time (the bytes the
// buffer cache uploads anyway); ranges the GPU modified are skipped (no read-back). Bounded memory
// (KYTY_INPUT_PROFILE_MAX_SLOTS, default 20000; raw bytes KYTY_INPUT_PROFILE_RAW_MB, default 128).
//
// Output: KYTY_INPUT_PROFILE_OUT (default ./kyty_input_profile.jsonl), rewritten whole (tmp +
// rename) every KYTY_INPUT_PROFILE_FLUSH_S (30) seconds by a background thread and at exit. One
// JSON object per line; the reader is work/mslnative/autoval/inprof.py. The file holds game data:
// keep it under work/ and never publish it.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace InputProfile {

// Value class bits of one dword.
enum : uint32_t {
	kZero     = 1u << 0u,  // 0
	kSmallInt = 1u << 1u,  // 1..255
	kInt16    = 1u << 2u,  // 256..65535
	kNegInt   = 1u << 3u,  // -65536..-1 as int32
	kFloatPos = 1u << 4u,  // normal float > 0 with |exponent| <= 40
	kFloatNeg = 1u << 5u,  // normal float < 0 with |exponent| <= 40
	kFloatBig = 1u << 6u,  // normal float outside that exponent range (and not an int class above)
	kDenorm   = 1u << 7u,  // f32 denormal (not an int class above)
	kNan      = 1u << 8u,
	kInf      = 1u << 9u,
	kNegZero  = 1u << 10u, // 0x80000000
};

constexpr uint32_t kBlockDwords = 64;
constexpr uint32_t kChunkBytes  = 4096; // readability granule of AddData's valid_chunks mask

[[nodiscard]] uint32_t Classify(uint32_t v) noexcept;

struct DwordStat {
	uint32_t n       = 0;
	uint32_t flags   = 0;
	uint32_t umin    = UINT32_MAX;
	uint32_t umax    = 0;
	float    fmin    = 0.0f; // valid when flags has a float class
	float    fmax    = 0.0f;
	uint32_t last    = 0;
	uint32_t changes = 0;     // samples whose value differs from the previous sample

	void Add(uint32_t v) noexcept;
};

struct Config {
	bool        on            = false;
	std::string out           = "kyty_input_profile.jsonl";
	uint32_t    first         = 8;
	uint32_t    max_samples   = 64;
	uint32_t    interval_ms   = 1000;
	uint32_t    stat_bytes    = 4096;      // per-position statistics
	uint32_t    read_bytes    = 64 * 1024; // whole read window (blocks after stat_bytes)
	uint32_t    raw_bytes     = 16 * 1024; // raw snapshot length
	uint32_t    raw           = 2;
	uint64_t    raw_budget    = uint64_t {128} * 1024 * 1024;
	uint32_t    max_slots     = 20000;
	uint32_t    flush_s       = 30;
	uint32_t    vertex_count  = 64;

	static Config FromEnv();
};

// One input slot of one shader.
struct Slot {
	std::vector<DwordStat>             dw;  // per dword position (stat window)
	std::vector<DwordStat>             blk; // per kBlockDwords block after the stat window
	uint32_t                           samples   = 0;
	uint32_t                           unread    = 0; // samples without readable bytes (GPU-modified, unmapped)
	uint64_t                           bytes_max = 0; // largest bound size seen
	uint32_t                           info      = 0; // kind-specific flags
	uint32_t                           stride    = 0; // 'V': fold stride in bytes (0 = linear)
	std::vector<std::vector<uint32_t>> metas;         // distinct descriptor/meta words (up to 4)
	std::vector<std::string>           raw;           // raw snapshots (bytes)
	bool                               dirty     = true;
	std::string                        line;          // cached serialisation
};

// Statistics registry (thread-safe). Production code uses Global(); tests make their own.
class Registry {
public:
	explicit Registry(Config cfg);

	// Sampling gate of one shader. True: this draw/dispatch is sampled (counts the sample).
	// `gate` separates independent gates of one shader (0 bindings, 1 vertex buffers).
	bool BeginSample(uint64_t hash, uint32_t stage, uint32_t gate, uint64_t now_ms);

	// Adds one sample of a data slot. data may be null (meta only, counted as unread).
	// stride != 0 folds the bytes by stride (vertex buffers). valid_chunks: bit c set = bytes
	// [c * kChunkBytes, (c + 1) * kChunkBytes) are readable (others are skipped; bit 31 covers the rest).
	void AddData(uint64_t hash, uint32_t stage, char kind, uint32_t slot, const void* data,
	             size_t bytes, uint64_t bound_bytes, const uint32_t* meta, uint32_t meta_n,
	             uint32_t info, uint32_t stride = 0, uint32_t valid_chunks = ~0u);
	// Descriptor-only slot ('T', 'S', 'W').
	void AddDescriptor(uint64_t hash, uint32_t stage, char kind, uint32_t slot, const uint32_t* meta,
	                   uint32_t meta_n, uint64_t bound_bytes = 0, uint32_t info = 0);

	// Whole profile as JSON lines (re-serialises only the slots changed since the last call).
	[[nodiscard]] std::string Serialize();
	bool                      WriteFile(const std::string& path);

	[[nodiscard]] size_t   SlotCount();
	[[nodiscard]] uint64_t RawBytes() const { return m_raw_bytes.load(std::memory_order_relaxed); }
	[[nodiscard]] const Config& Cfg() const { return m_cfg; }

private:
	using Key = std::tuple<uint64_t, uint32_t, char, uint32_t>;
	struct Gate {
		uint64_t draws   = 0;
		uint32_t samples = 0;
		uint64_t last_ms = 0;
	};
	Slot* Find(const Key& key);
	static void AddMeta(Slot& s, const uint32_t* meta, uint32_t meta_n);
	void        SerializeSlot(const Key& key, Slot& s);

	Config                                                 m_cfg;
	std::mutex                                             m_mutex;
	std::map<Key, std::unique_ptr<Slot>>                   m_slots;
	std::map<std::tuple<uint64_t, uint32_t, uint32_t>, Gate> m_gates;
	std::atomic<uint64_t>                                  m_raw_bytes {0};
	uint32_t                                               m_dropped = 0;
};

namespace Detail {
extern bool g_on;
} // namespace Detail

// KYTY_INPUT_PROFILE=1 (read once at static initialisation).
[[nodiscard]] inline bool On() noexcept {
	return Detail::g_on;
}
// The process-wide registry (created on first use; starts the flush thread and the exit flush).
Registry& Global();
// Milliseconds of a monotonic clock.
[[nodiscard]] uint64_t NowMs() noexcept;

std::string Base64(const void* data, size_t bytes);

} // namespace InputProfile

#endif /* EMULATOR_SRC_COMMON_INPUTPROFILE_H_ */
