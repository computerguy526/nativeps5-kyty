// kyty_pipeline_list export --base and --scan (kytyPipelineList.cpp): the parts that need nothing of the
// emulator, so that a small test can hold them to their byte formats.
#ifndef KYTY_TOOLS_PIPELINE_LIST_EXPORT_BASE_H_
#define KYTY_TOOLS_PIPELINE_LIST_EXPORT_BASE_H_

#include <cstdint>
#include <cstring>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace PipelineListTool {

// A stored resource specialization (shader/recompiler/ir/ProgramCodec.cpp, EncodeSpecialization): a
// count and that many buffer records, then a count and that many image records. Counts and words are
// 32-bit little endian; a flag is one byte that is 0 or 1. A buffer record is three words and a flag.
// An image record of the int21 release is eight words and the flags cube and fmask; this build's has
// four more fields after them: the flags bindless and runtime, the word fmask_fragments and the flag
// clamp_half_float.
inline constexpr size_t SpecializationBufferBytes     = 3 * 4 + 1;
inline constexpr size_t SpecializationImageInt21Bytes = 8 * 4 + 2;
inline constexpr size_t SpecializationImageBytes      = SpecializationImageInt21Bytes + 2 + 4 + 1;

// Whether `bytes` read as a specialization with the int21 release's image records (`int21`) or
// with this build's, as the emulator's decoders decide it (a count larger than the bytes left, a flag
// byte above 1 or bytes left over: no). `images_at` gets the offset of the first image record,
// `images` their count.
inline bool ReadsAsSpecialization(std::span<const uint8_t> bytes, bool int21, size_t* images_at, uint32_t* images) {
	size_t     pos  = 0;
	const auto word = [&](uint32_t& value) {
		if (bytes.size() - pos < 4) return false;
		std::memcpy(&value, bytes.data() + pos, 4);
		pos += 4;
		return true;
	};
	const auto words = [&](size_t count) {
		if (bytes.size() - pos < count * 4) return false;
		pos += count * 4;
		return true;
	};
	const auto flag = [&] {
		if (pos >= bytes.size() || bytes[pos] > 1) return false;
		pos++;
		return true;
	};
	uint32_t buffers = 0;
	if (!word(buffers) || buffers > bytes.size() - pos) return false;
	for (uint32_t i = 0; i < buffers; i++) {
		if (!words(3) || !flag()) return false;
	}
	uint32_t count = 0;
	if (!word(count) || count > bytes.size() - pos) return false;
	*images_at = pos;
	*images    = count;
	for (uint32_t i = 0; i < count; i++) {
		if (!words(8) || !flag() || !flag()) return false;
		if (!int21 && (!flag() || !flag() || !words(1) || !flag())) return false;
	}
	return pos == bytes.size();
}

// The emulator loads a list of the int21 release by decoding every specialization that does not
// read as this build's with int21's image record and encoding it again (pipelineCache.cpp with
// DecodeSpecializationInt21): the added fields get their defaults (false, false, 0, false). This
// does the same on the bytes. True when `bytes` were rewritten; bytes that read as this build's, or
// as neither, stay as they are.
// v1.3.x lists: the image record has one more word, `indirect_group`, before the flags cube and fmask (DecodeSpecializationV13);
// a record that names a group is refused (it names a resource this build cannot bind). The emulator reads these through
// DecodeSpecializationCompat when it loads a list, so the upgrade must write them in this build's record as well.
inline constexpr size_t SpecializationImageV13Bytes = 9 * 4 + 2;
inline bool ReadsAsSpecializationV13(std::span<const uint8_t> bytes, size_t* images_at, uint32_t* images) {
	size_t pos = 0;
	const auto word = [&](uint32_t& v) { if (bytes.size() - pos < 4) return false; std::memcpy(&v, bytes.data() + pos, 4); pos += 4; return true; };
	uint32_t buffers = 0;
	if (!word(buffers) || buffers > bytes.size() - pos) return false;
	for (uint32_t i = 0; i < buffers; i++) {
		if (bytes.size() - pos < 12 + 1 || bytes[pos + 12] > 1) return false;
		pos += 13;
	}
	uint32_t count = 0;
	if (!word(count) || count > bytes.size() - pos) return false;
	*images_at = pos;
	*images    = count;
	if (bytes.size() - pos != size_t {count} * SpecializationImageV13Bytes) return false;
	for (uint32_t i = 0; i < count; i++) {
		const auto record = pos + size_t {i} * SpecializationImageV13Bytes;
		uint32_t   group  = 0;
		std::memcpy(&group, bytes.data() + record + 32, 4);
		if (group != UINT32_MAX || bytes[record + 36] > 1 || bytes[record + 37] > 1) return false;
	}
	return true;
}

// A v1.3.x specialization (no group named) as this build's: the group word dropped, the added fields at their defaults.
inline bool UpgradeV13Specialization(std::vector<uint8_t>& bytes) {
	size_t   images_at = 0;
	uint32_t images    = 0;
	if (!ReadsAsSpecializationV13(bytes, &images_at, &images)) return false;
	std::vector<uint8_t> out(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(images_at));
	for (uint32_t i = 0; i < images; i++) {
		const auto record = bytes.begin() + static_cast<std::ptrdiff_t>(images_at + size_t {i} * SpecializationImageV13Bytes);
		out.insert(out.end(), record, record + 32);
		out.insert(out.end(), record + 36, record + 38);
		out.insert(out.end(), SpecializationImageBytes - SpecializationImageInt21Bytes, uint8_t {0});
	}
	bytes = std::move(out);
	return true;
}

// The tool's exit code for specializations that read as no known layout after the upgrade: 2 (refused) unless --allow-partial.
inline int PartialExitCode(uint64_t unreadable, bool allow_partial) { return unreadable != 0 && !allow_partial ? 2 : 0; }

inline bool UpgradeInt21Specialization(std::vector<uint8_t>& bytes) {
	size_t   images_at = 0;
	uint32_t images    = 0;
	if (ReadsAsSpecialization(bytes, false, &images_at, &images)) return false;
	if (!ReadsAsSpecialization(bytes, true, &images_at, &images)) return UpgradeV13Specialization(bytes);
	std::vector<uint8_t> out;
	out.reserve(bytes.size() + size_t {images} * (SpecializationImageBytes - SpecializationImageInt21Bytes));
	out.insert(out.end(), bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(images_at));
	for (uint32_t i = 0; i < images; i++) {
		const auto record = bytes.begin() + static_cast<std::ptrdiff_t>(images_at + size_t {i} * SpecializationImageInt21Bytes);
		out.insert(out.end(), record, record + static_cast<std::ptrdiff_t>(SpecializationImageInt21Bytes));
		out.insert(out.end(), SpecializationImageBytes - SpecializationImageInt21Bytes, uint8_t {0});
	}
	bytes = std::move(out);
	return true;
}

// The order --scan reads the game's files in when a base list is given: that list's data files first
// (new shaders of a game mostly sit in the files the known ones are in), then the other files of the
// folders those files are in, then the rest. Inside each group the order stays as given (by size,
// then by path). `path_of` gives a file's path relative to the game folder with '/' separators, as
// the list names its data files.
template <typename Entry, typename PathOf>
void OrderForScan(std::vector<Entry>& files, const std::vector<std::string>& base_files, PathOf&& path_of) {
	if (base_files.empty()) return;
	const auto folder_of = [](const std::string& path) {
		const auto slash = path.rfind('/');
		return slash == std::string::npos ? std::string() : path.substr(0, slash);
	};
	const std::set<std::string> named(base_files.begin(), base_files.end());
	std::set<std::string>       folders;
	for (const auto& path: base_files) folders.insert(folder_of(path));
	std::vector<uint8_t> group(files.size());
	for (size_t i = 0; i < files.size(); i++) {
		const std::string path = path_of(files[i]);
		group[i]               = named.contains(path) ? 0 : folders.contains(folder_of(path)) ? 1 : 2;
	}
	std::vector<Entry> ordered;
	ordered.reserve(files.size());
	for (uint8_t wanted = 0; wanted < 3; wanted++) {
		for (size_t i = 0; i < files.size(); i++) {
			if (group[i] == wanted) ordered.push_back(std::move(files[i]));
		}
	}
	files = std::move(ordered);
}

} // namespace PipelineListTool

#endif /* KYTY_TOOLS_PIPELINE_LIST_EXPORT_BASE_H_ */
