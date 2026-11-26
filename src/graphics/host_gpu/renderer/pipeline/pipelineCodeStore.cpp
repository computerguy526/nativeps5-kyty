#include "graphics/host_gpu/renderer/pipeline/pipelineCodeStore.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <system_error>
#include <xxhash.h>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

namespace {

using Hash128 = PipelineList::Hash128;

constexpr uint64_t Align8(uint64_t value) {
	return (value + 7u) & ~uint64_t {7};
}

// The 16-byte, zero-padded form of a title or version stamp (longer text is cut).
std::array<char, 16> Stamp(const std::string& text) {
	std::array<char, 16> out {};
	std::memcpy(out.data(), text.data(), std::min<size_t>(text.size(), out.size()));
	return out;
}

template <typename T>
void PutAt(std::vector<uint8_t>& out, size_t at, T value) {
	std::memcpy(out.data() + at, &value, sizeof(value));
}

template <typename T>
T GetAt(const uint8_t* bytes) {
	T value;
	std::memcpy(&value, bytes, sizeof(value));
	return value;
}

// Header layout: magic 0, version 8, header bytes 12, title 16, app_ver 32, fingerprint 48,
// reserved 56.
std::vector<uint8_t> MakeHeader(const std::string& title, const std::string& app_ver, uint64_t fp) {
	std::vector<uint8_t> header(PipelineCodeStore::HeaderBytes, 0);
	std::memcpy(header.data(), PipelineCodeStore::FileMagic, 8);
	PutAt<uint32_t>(header, 8, PipelineCodeStore::FormatVersion);
	PutAt<uint32_t>(header, 12, PipelineCodeStore::HeaderBytes);
	std::memcpy(header.data() + 16, Stamp(title).data(), 16);
	std::memcpy(header.data() + 32, Stamp(app_ver).data(), 16);
	PutAt<uint64_t>(header, 48, fp);
	return header;
}

// fopen of a filesystem path (UTF-16 paths on Windows).
std::FILE* OpenFile(const std::filesystem::path& path, const char* mode) {
#if defined(_WIN32)
	std::wstring wide_mode;
	for (const char* c = mode; *c != '\0'; c++)
		wide_mode.push_back(static_cast<wchar_t>(*c));
	return _wfopen(path.c_str(), wide_mode.c_str());
#else
	return std::fopen(path.c_str(), mode);
#endif
}

bool ReadExact(std::FILE* file, void* out, size_t bytes) {
	return std::fread(out, 1, bytes, file) == bytes;
}

// Cuts the open file to `bytes` (after flushing what is buffered).
bool TruncateFile(std::FILE* file, uint64_t bytes) {
	if (std::fflush(file) != 0) return false;
#if defined(_WIN32)
	return _chsize_s(_fileno(file), static_cast<long long>(bytes)) == 0;
#else
	return ftruncate(fileno(file), static_cast<off_t>(bytes)) == 0;
#endif
}

} // namespace

PipelineCodeStore::~PipelineCodeStore() {
	Close();
}

void PipelineCodeStore::Close() {
	if (m_file != nullptr) {
		std::fflush(m_file);
		std::fclose(m_file);
		m_file = nullptr;
	}
#if !defined(_WIN32)
	if (m_lock_fd >= 0) {
		close(m_lock_fd); // drops the flock
		m_lock_fd = -1;
	}
#endif
}

void PipelineCodeStore::Release() {
	std::lock_guard lock(m_mutex);
	Close();
}

// One writer per store file: an exclusive, non-blocking flock on a sidecar file. False when another
// process holds it (this one then opens the store read-only). Always true where there is no flock.
bool PipelineCodeStore::TakeLock() {
#if defined(_WIN32)
	return true;
#else
	auto lock_path = m_path;
	lock_path += ".lock";
	const int fd = open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
	if (fd < 0) return true; // cannot make the lock file: no protection, but no refusal either
	if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
		close(fd);
		return false;
	}
	m_lock_fd = fd;
	return true;
#endif
}

// Reads the header and records of the file at m_path into the index. `allow_any_header`: the tool's
// read-only open takes the stamps as they are (and keeps them in m_fp). Returns false when the
// header is not this store's (or, without allow_any_header, is another title/version/decoder's).
// `m_size` is left at the end of the last intact record; a damaged tail is counted in m_truncated.
bool PipelineCodeStore::Scan(bool allow_any_header) {
	std::FILE* in = OpenFile(m_path, "rb");
	if (in == nullptr) return false;
	std::fseek(in, 0, SEEK_END);
	const auto total = static_cast<uint64_t>(std::ftell(in));
	std::fseek(in, 0, SEEK_SET);
	std::vector<uint8_t> header(HeaderBytes);
	bool                 ok = total >= HeaderBytes && ReadExact(in, header.data(), header.size()) &&
	                          std::memcmp(header.data(), FileMagic, 8) == 0 &&
	                          GetAt<uint32_t>(header.data() + 8) == FormatVersion &&
	                          GetAt<uint32_t>(header.data() + 12) == HeaderBytes;
	if (ok) {
		const auto want = MakeHeader(m_title, m_app_ver, m_fp);
		if (allow_any_header) {
			m_fp = GetAt<uint64_t>(header.data() + 48);
		} else {
			ok = std::memcmp(header.data() + 16, want.data() + 16, 40) == 0;
		}
	}
	if (!ok) {
		std::fclose(in);
		return false;
	}
	m_index.clear();
	m_mark                  = {};
	m_mark_at               = 0;
	m_code_bytes            = 0;
	uint64_t             at = HeaderBytes;
	std::vector<uint8_t> record;
	while (at + RecordHead + 8 <= total) {
		std::fseek(in, static_cast<long>(at), SEEK_SET);
		uint8_t head[RecordHead];
		if (!ReadExact(in, head, sizeof(head))) break;
		const auto code_bytes = GetAt<uint32_t>(head + 4);
		const bool is_mark    = GetAt<uint32_t>(head) == MarkMagic;
		if (is_mark) {
			if (code_bytes % 12u != 0 || code_bytes > MaxMarkRanges * 12u) break;
		} else if (GetAt<uint32_t>(head) != RecordMagic || code_bytes == 0 ||
		           code_bytes > MaxCodeBytes) {
			break;
		}
		const uint64_t body = RecordHead + Align8(code_bytes);
		if (at + body + 8 > total) break;
		record.resize(static_cast<size_t>(body + 8));
		std::memcpy(record.data(), head, sizeof(head));
		if (!ReadExact(in, record.data() + RecordHead, static_cast<size_t>(body + 8 - RecordHead)))
			break;
		if (XXH3_64bits(record.data(), static_cast<size_t>(body)) !=
		    GetAt<uint64_t>(record.data() + body))
			break;
		if (is_mark) {
			const auto state = GetAt<uint64_t>(head + 16);
			if (state != SearchMark::Partial && state != SearchMark::Complete) break;
			m_mark.state = static_cast<uint32_t>(state);
			m_mark.key   = GetAt<uint64_t>(head + 8);
			m_mark.done.clear();
			for (uint32_t i = 0; i < code_bytes / 12u; i++) {
				const uint8_t* r = record.data() + RecordHead + i * 12u;
				m_mark.done.push_back(
				    {GetAt<uint32_t>(r), GetAt<uint32_t>(r + 4), GetAt<uint32_t>(r + 8)});
			}
			m_mark_at = at; // the last record, unless another follows
			at += body + 8;
			continue;
		}
		if (m_mark_at != 0 || m_mark.state != SearchMark::None) m_mark = {}; // stale: a code came after it
		m_mark_at = 0;
		const Hash128 hash {GetAt<uint64_t>(head + 8), GetAt<uint64_t>(head + 16)};
		if (m_index.try_emplace(hash, Entry {at + RecordHead, code_bytes}).second)
			m_code_bytes += code_bytes;
		at += body + 8;
	}
	std::fclose(in);
	m_size      = at;
	m_truncated = total - at;
	return true;
}

bool PipelineCodeStore::Open(const std::filesystem::path& path, const std::string& title,
                             const std::string& app_ver, uint64_t decoder_fp) {
	std::lock_guard lock(m_mutex);
	Close();
	m_index.clear();
	m_path       = path;
	m_title      = title;
	m_app_ver    = app_ver;
	m_fp         = decoder_fp;
	m_read_only  = false;
	m_stale      = false;
	m_dirty      = false;
	m_truncated  = 0;
	m_duplicates = 0;
	m_code_bytes = 0;
	m_size       = 0;
	m_locked_by_other = false;
	m_mark            = {};
	m_mark_at         = 0;
	std::error_code error;
	if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), error);
	if (!TakeLock()) {
		// Another process writes this store: read what is intact, change nothing.
		m_locked_by_other = true;
		m_read_only       = true;
		if (std::filesystem::exists(path, error) && Scan(false)) {
			m_file = OpenFile(path, "rb");
			return m_file != nullptr;
		}
		m_index.clear();
		m_code_bytes = 0;
		return true;
	}
	const auto existing =
	    std::filesystem::exists(path, error) ? std::filesystem::file_size(path, error) : 0;
	bool fresh = true;
	if (existing != 0) {
		if (Scan(false)) {
			fresh = false;
			if (m_truncated != 0)
				std::filesystem::resize_file(path, m_size, error); // the intact prefix stays
			if (error) return false;
		} else {
			// Another title, version or decoder (or not a store): kept aside, a fresh one started.
			auto aside = path;
			aside += ".stale";
			std::filesystem::remove(aside, error);
			std::filesystem::rename(path, aside, error);
			if (error) return false;
			m_stale = true;
			m_index.clear();
			m_code_bytes = 0;
			m_truncated  = 0;
		}
	}
	if (fresh) {
		m_file = OpenFile(path, "w+b");
		if (m_file == nullptr) return false;
		const auto header = MakeHeader(title, app_ver, decoder_fp);
		if (std::fwrite(header.data(), 1, header.size(), m_file) != header.size()) {
			Close();
			return false;
		}
		std::fflush(m_file);
		m_size = HeaderBytes;
	} else {
		m_file = OpenFile(path, "r+b");
		if (m_file == nullptr) return false;
	}
	return true;
}

bool PipelineCodeStore::OpenReadOnly(const std::filesystem::path& path) {
	std::lock_guard lock(m_mutex);
	Close();
	m_locked_by_other = false;
	m_index.clear();
	m_path       = path;
	m_read_only  = true;
	m_stale      = false;
	m_dirty      = false;
	m_duplicates = 0;
	if (!Scan(true)) return false;
	m_file = OpenFile(path, "rb");
	return m_file != nullptr;
}

bool PipelineCodeStore::Find(const Hash128& hash, std::span<uint8_t> out) {
	std::lock_guard lock(m_mutex);
	const auto      it = m_index.find(hash);
	if (m_file == nullptr || it == m_index.end() || it->second.bytes != out.size()) return false;
	if (m_dirty) {
		std::fflush(m_file);
		m_dirty = false;
	}
	return std::fseek(m_file, static_cast<long>(it->second.offset), SEEK_SET) == 0 &&
	       ReadExact(m_file, out.data(), out.size());
}

uint32_t PipelineCodeStore::Size(const Hash128& hash) const {
	std::lock_guard lock(m_mutex);
	const auto      it = m_index.find(hash);
	return it == m_index.end() ? 0 : it->second.bytes;
}

bool PipelineCodeStore::Add(const Hash128& hash, uint64_t prefix_hash,
                            std::span<const uint8_t> code) {
	std::lock_guard lock(m_mutex);
	if (m_file == nullptr || m_read_only || code.empty() || code.size() > MaxCodeBytes)
		return false;
	if (m_index.contains(hash)) {
		m_duplicates++;
		return false;
	}
	const uint64_t body = RecordHead + Align8(code.size());
	// A mark at the end is overwritten by this record (Flush writes it again); room for it stays free.
	const uint64_t at = m_mark_at != 0 ? m_mark_at : m_size;
	if (at + body + 8 + MarkReserveBytes > CapBytes) return false;
	if (m_mark_at != 0) {
		if (!TruncateFile(m_file, m_mark_at)) return false;
		m_size    = m_mark_at;
		m_mark_at = 0;
	}
	std::vector<uint8_t> record(static_cast<size_t>(body + 8), 0);
	PutAt<uint32_t>(record, 0, RecordMagic);
	PutAt<uint32_t>(record, 4, static_cast<uint32_t>(code.size()));
	PutAt<uint64_t>(record, 8, hash.low);
	PutAt<uint64_t>(record, 16, hash.high);
	PutAt<uint64_t>(record, 24, prefix_hash);
	std::memcpy(record.data() + RecordHead, code.data(), code.size());
	PutAt<uint64_t>(record, static_cast<size_t>(body),
	                XXH3_64bits(record.data(), static_cast<size_t>(body)));
	if (std::fseek(m_file, static_cast<long>(m_size), SEEK_SET) != 0 ||
	    std::fwrite(record.data(), 1, record.size(), m_file) != record.size()) {
		return false;
	}
	m_index.emplace(hash, Entry {m_size + RecordHead, static_cast<uint32_t>(code.size())});
	m_size += record.size();
	m_code_bytes += code.size();
	m_dirty = true;
	return true;
}

size_t PipelineCodeStore::Count() const {
	std::lock_guard lock(m_mutex);
	return m_index.size();
}

bool PipelineCodeStore::AppendMarkLocked() {
	const uint64_t payload = m_mark.done.size() * 12ull;
	const uint64_t body    = RecordHead + Align8(payload);
	if (m_size + body + 8 > CapBytes) return false;
	std::vector<uint8_t> record(static_cast<size_t>(body + 8), 0);
	PutAt<uint32_t>(record, 0, MarkMagic);
	PutAt<uint32_t>(record, 4, static_cast<uint32_t>(payload));
	PutAt<uint64_t>(record, 8, m_mark.key);
	PutAt<uint64_t>(record, 16, m_mark.state);
	for (size_t i = 0; i < m_mark.done.size(); i++) {
		PutAt<uint32_t>(record, RecordHead + i * 12, m_mark.done[i].library);
		PutAt<uint32_t>(record, RecordHead + i * 12 + 4, m_mark.done[i].first);
		PutAt<uint32_t>(record, RecordHead + i * 12 + 8, m_mark.done[i].last);
	}
	PutAt<uint64_t>(record, static_cast<size_t>(body),
	                XXH3_64bits(record.data(), static_cast<size_t>(body)));
	if (std::fseek(m_file, static_cast<long>(m_size), SEEK_SET) != 0 ||
	    std::fwrite(record.data(), 1, record.size(), m_file) != record.size()) {
		return false;
	}
	m_mark_at = m_size;
	m_size += record.size();
	m_dirty = true;
	return true;
}

PipelineCodeStore::SearchMark PipelineCodeStore::Mark() const {
	std::lock_guard lock(m_mutex);
	return m_mark;
}

bool PipelineCodeStore::WriteMark(const SearchMark& mark) {
	std::lock_guard lock(m_mutex);
	if (m_file == nullptr || m_read_only) return false;
	SearchMark next = mark;
	if (next.state == SearchMark::Partial && next.done.size() > MaxMarkRanges) return false;
	if (next.state != SearchMark::Complete) next.state = next.done.empty() ? SearchMark::None : next.state;
	if (m_mark_at != 0) {
		if (!TruncateFile(m_file, m_mark_at)) return false;
		m_size    = m_mark_at;
		m_mark_at = 0;
	}
	m_mark = next;
	if (m_mark.state == SearchMark::None) return true;
	return AppendMarkLocked();
}

bool PipelineCodeStore::Flush() {
	std::lock_guard lock(m_mutex);
	if (m_file == nullptr) return false;
	if (!m_read_only && m_mark.state != SearchMark::None && m_mark_at == 0) AppendMarkLocked();
	m_dirty = false;
	return std::fflush(m_file) == 0;
}

bool PipelineCodeStore::ContainsAll(const std::vector<Hash128>& hashes) const {
	std::lock_guard lock(m_mutex);
	return std::all_of(hashes.begin(), hashes.end(),
	                   [&](const Hash128& hash) { return m_index.contains(hash); });
}

bool PipelineCodeStore::Full() const {
	std::lock_guard lock(m_mutex);
	return (m_mark_at != 0 ? m_mark_at : m_size) + RecordHead + 16 + MarkReserveBytes > CapBytes;
}

uint64_t PipelineCodeStore::FileBytes() const {
	std::lock_guard lock(m_mutex);
	return m_size;
}

uint64_t PipelineCodeStore::CodeBytes() const {
	std::lock_guard lock(m_mutex);
	return m_code_bytes;
}

uint64_t PipelineCodeStore::DuplicatesIgnored() const {
	std::lock_guard lock(m_mutex);
	return m_duplicates;
}

PipelineList::ReadHashFn PipelineCodeStore::ReadFn() {
	return [this](const Hash128& hash, uint32_t bytes, std::span<uint8_t> out) {
		return out.size() == bytes && Find(hash, out);
	};
}

} // namespace Libs::Graphics
