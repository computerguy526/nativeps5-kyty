#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECODESTORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECODESTORE_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// KYCODE1: the per-player shader code store of a pipeline list's `<runtime>` shaders (pipelineList.h).
// The code comes from the player's own game files at run time; the store never ships and never leaves the
// machine. Append-only; dedupes by the code's XXH3-128; capped; a damaged tail keeps the intact prefix.
//
// File: a 64-byte header (magic "KYCODE1\0", u32 version 1, u32 header bytes 64, char title[16],
// char app_ver[16], u64 decoder fingerprint, u64 reserved), then records
//   { u32 'CODE' (0x45444F43), u32 code_bytes, u64 hash_low, u64 hash_high, u64 prefix_hash,
//     code[code_bytes], zero padding to 8, u64 XXH3-64 of everything from the magic to the padding }.
// A header whose title, app_ver or fingerprint differs from the run's is discarded: the file is renamed
// `<path>.stale` (replacing an older one) and a fresh store is started.
class PipelineCodeStore {
public:
	static constexpr char     FileMagic[8]    = {'K', 'Y', 'C', 'O', 'D', 'E', '1', '\0'};
	static constexpr uint32_t FormatVersion   = 1;
	static constexpr uint32_t HeaderBytes     = 64;
	static constexpr uint32_t RecordMagic     = 0x45444F43u; // "CODE"
	static constexpr uint32_t RecordHead      = 32;          // magic, code_bytes, hash low/high, prefix hash
	static constexpr uint64_t CapBytes        = 64ull << 20u; // the file never grows past this
	static constexpr uint32_t MaxCodeBytes    = 16u << 20u;

	PipelineCodeStore() = default;
	~PipelineCodeStore();
	PipelineCodeStore(const PipelineCodeStore&)            = delete;
	PipelineCodeStore& operator=(const PipelineCodeStore&) = delete;

	// Opens (or creates) the store at `path` for appending. False when the file cannot be created or read.
	[[nodiscard]] bool Open(const std::filesystem::path& path, const std::string& title, const std::string& app_ver,
	                        uint64_t decoder_fp);
	// Reads an existing store without any change to the file: any header is accepted, a damaged tail is
	// ignored (the tools: check --code-store). Add fails on such a store.
	[[nodiscard]] bool OpenReadOnly(const std::filesystem::path& path);

	// The code of `hash` into `out` (which must be exactly Size(hash) bytes); false when it is not stored.
	[[nodiscard]] bool Find(const PipelineList::Hash128& hash, std::span<uint8_t> out);
	// Its size in bytes; 0 when it is not stored.
	[[nodiscard]] uint32_t Size(const PipelineList::Hash128& hash) const;
	// Appends the code; true when it is new. False for a duplicate (nothing written), a code that would
	// take the file past the cap (Full() then reports it) or an empty/oversized one.
	bool Add(const PipelineList::Hash128& hash, uint64_t prefix_hash, std::span<const uint8_t> code);
	[[nodiscard]] size_t Count() const;
	[[nodiscard]] bool   Flush();
	// True when every hash in `hashes` is stored (an empty list: true).
	[[nodiscard]] bool ContainsAll(const std::vector<PipelineList::Hash128>& hashes) const;

	[[nodiscard]] bool     Full() const;
	[[nodiscard]] uint64_t FileBytes() const;
	[[nodiscard]] uint64_t CodeBytes() const; // payload bytes of the stored codes
	[[nodiscard]] uint64_t DuplicatesIgnored() const;
	// 1 when the file was renamed .stale at Open (another title, version or decoder), 0 otherwise.
	[[nodiscard]] bool     DiscardedStale() const { return m_stale; }
	// Bytes of a damaged tail cut off at Open.
	[[nodiscard]] uint64_t TruncatedBytes() const { return m_truncated; }
	[[nodiscard]] uint64_t DecoderFingerprint() const { return m_fp; }

	// The store as a ReadHashFn for PipelineList::Resolve (the store outlives the function).
	[[nodiscard]] PipelineList::ReadHashFn ReadFn();

private:
	struct Entry {
		uint64_t offset = 0; // of the code in the file
		uint32_t bytes  = 0;
	};
	struct HashOf {
		size_t operator()(const PipelineList::Hash128& h) const noexcept {
			return static_cast<size_t>(h.low ^ (h.high * 0x9E3779B97F4A7C15ull));
		}
	};

	bool Scan(bool allow_any_header);
	void Close();

	mutable std::mutex                                           m_mutex;
	std::FILE*                                                   m_file = nullptr;
	std::filesystem::path                                        m_path;
	std::string                                                  m_title, m_app_ver;
	uint64_t                                                     m_fp = 0;
	bool                                                         m_read_only = false;
	bool                                                         m_stale     = false;
	bool                                                         m_dirty     = false;
	uint64_t                                                     m_size      = 0; // file bytes (the append position)
	uint64_t                                                     m_code_bytes = 0, m_duplicates = 0, m_truncated = 0;
	std::unordered_map<PipelineList::Hash128, Entry, HashOf>     m_index;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECODESTORE_H_
