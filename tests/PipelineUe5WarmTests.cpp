// The `<runtime>` shaders of a KYPLST list (UE5 titles, pipelineList.h) and what serves them:
//  - Resolve with a ReadHashFn: found shaders, "pending" for code the store lacks (never unreadable or relocated), the
//    merged-stage back half, the 50 % rule, a list without `<runtime>` unchanged;
//  - the KYCODE1 per-player code store (pipelineCodeStore.h): add, find, dedupe, reopen, damaged tail, stale header, cap;
//  - the library group split (ue5Group.h) on synthetic groups;
//  - the tool's runtime helpers (tools/pipeline-list/runtimeCode.h).
// Synthetic data only: pseudo-random "code", no game bytes.
#include "graphics/host_gpu/renderer/pipeline/pipelineCodeStore.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"
#include "graphics/host_gpu/renderer/pipeline/ue5Group.h"
#include "runtimeCode.h"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <xxhash.h>

using namespace Libs::Graphics;
using PipelineList::Hash128;

static int g_failed = 0;
#define CHECK(expr)                                                                        \
	do {                                                                                   \
		if (!(expr)) {                                                                     \
			std::printf("PipelineUe5WarmTests: FAILED: %s (line %d)\n", #expr, __LINE__); \
			++g_failed;                                                                    \
		}                                                                                  \
	} while (0)

namespace {

constexpr std::array<uint32_t, 3> InputSizes = {96, 64, 48};

std::vector<uint32_t> Code(uint32_t seed, uint32_t words) {
	std::mt19937          random(seed);
	std::vector<uint32_t> code(words);
	for (auto& word: code) word = random();
	return code;
}

std::span<const uint8_t> Bytes(const std::vector<uint32_t>& code) {
	return {reinterpret_cast<const uint8_t*>(code.data()), code.size() * 4};
}

ShaderJournal::Source SourceOf(std::vector<uint32_t> code, ShaderJournal::Kind kind, uint32_t stage, size_t input_bytes) {
	ShaderJournal::Source source;
	source.stage           = stage;
	source.kind            = kind;
	source.hash            = XXH3_64bits(code.data(), code.size() * 4);
	source.user_data_count = 16;
	source.code_size       = static_cast<uint32_t>(code.size());
	source.wave_size       = 32;
	source.static_state    = {stage, 7, 9};
	source.code            = std::move(code);
	source.input_info.assign(input_bytes, 0);
	source.input_info[0] = static_cast<uint8_t>(stage);
	return source;
}

PipelineList::Info TestInfo() {
	PipelineList::Info info;
	info.title_id         = "PPSA00000";
	info.app_version      = "01.000.000";
	info.producer         = "PipelineUe5WarmTests";
	info.input_info_sizes = InputSizes;
	info.created_unix     = 1234;
	info.state_format     = PipelineList::StateFormat;
	return info;
}

// A store of codes kept in memory, as a ReadHashFn; counts the calls.
struct MemoryStore {
	std::map<std::pair<uint64_t, uint64_t>, std::vector<uint8_t>> codes;
	std::atomic<int>                                               calls {0};
	void Put(const std::vector<uint32_t>& code) {
		const auto hash = PipelineList::CodeHash(Bytes(code));
		codes[{hash.low, hash.high}].assign(Bytes(code).begin(), Bytes(code).end());
	}
	PipelineList::ReadHashFn Fn() {
		return [this](const Hash128& hash, uint32_t bytes, std::span<uint8_t> out) {
			calls++;
			const auto it = codes.find({hash.low, hash.high});
			if (it == codes.end() || it->second.size() != bytes || out.size() != bytes) return false;
			std::memcpy(out.data(), it->second.data(), bytes);
			return true;
		};
	}
};

PipelineList::ReadCodeFn FailingRead(int& calls) {
	return [&calls](PipelineList::CodeFile, uint64_t, std::span<uint8_t>) {
		calls++;
		return false;
	};
}

// A list of `<runtime>` shaders: a vertex, a pixel and a merged-stage shader, each with one permutation. All located
// as `<runtime>` (the way export --runtime-code keeps them).
struct RuntimeList {
	std::vector<ShaderJournal::Source> sources;
	std::vector<ShaderJournal::Entry>  entries;
	PipelineList::File                 file;
};

RuntimeList MakeRuntimeList() {
	RuntimeList r;
	r.sources.push_back(SourceOf(Code(1, 300), ShaderJournal::Kind::Vertex, 1, InputSizes[0]));
	r.sources.push_back(SourceOf(Code(2, 200), ShaderJournal::Kind::Pixel, 2, InputSizes[1]));
	auto merged      = SourceOf(Code(3, 160), ShaderJournal::Kind::Vertex, 5, InputSizes[0]);
	merged.back_code = Code(4, 96);
	merged.hash      = MergedSourceHash(merged.code, merged.back_code);
	r.sources.push_back(merged);
	r.entries = {{0, 4, {1, 2, 3}}, {1, 0, {9}}, {2, 0, {4}}};
	PipelineList::Builder builder(TestInfo());
	std::set<PipelineListTool::HashKey> kept;
	PipelineList::ExportStats           stats;
	builder.AddJournals(r.sources, r.entries, {},
	                    PipelineListTool::KeepRuntimeCode([](std::span<const uint32_t>) { return std::nullopt; }, builder, kept),
	                    stats);
	r.file = builder.Finish();
	return r;
}

void Fill(MemoryStore& store, const RuntimeList& list) {
	for (const auto& source: list.sources) {
		store.Put(source.code);
		if (!source.back_code.empty()) store.Put(source.back_code);
	}
}

// ---- the list ------------------------------------------------------------------------------------------------

void TestRuntimeListFormat() {
	const auto list = MakeRuntimeList();
	const auto& file = list.file;
	CHECK(file.shaders.size() == 3 && file.permutations.size() == 3 && file.files.size() == 1 &&
	      file.files[0] == PipelineList::RuntimeFileName && std::string(PipelineList::RuntimeFileName) == "<runtime>");
	const auto runtime = PipelineList::RuntimeFileOf(file);
	CHECK(static_cast<uint32_t>(runtime) == 1);
	for (const auto& shader: file.shaders) CHECK(shader.file == runtime && shader.offset == 0 && shader.back_offset == 0);
	CHECK(file.shaders[2].back_code_size == 96);
	// No format bump: version 2, and the file round-trips with its Files entry.
	CHECK(PipelineList::FormatVersion == 2);
	const auto         bytes = PipelineList::Encode(file);
	PipelineList::File decoded;
	CHECK(PipelineList::Decode(bytes, decoded) && decoded.files == file.files && decoded.shaders.size() == 3);
	CHECK(decoded.shaders[1].code_hash_low == file.shaders[1].code_hash_low && decoded.shaders[1].prefix_hash == file.shaders[1].prefix_hash &&
	      decoded.shaders[1].file == runtime);
	// The file carries no code.
	PipelineList::CodeWindows windows;
	for (const auto& source: list.sources) windows.Add(Bytes(source.code));
	CHECK(windows.Count() > 500 && windows.FindIn(bytes).empty());
	// A list without the entry has no runtime file.
	PipelineList::File plain;
	CHECK(PipelineList::RuntimeFileOf(plain) == PipelineList::CodeFile::Executable);
	// Merging keeps the runtime shaders and their file.
	PipelineList::Builder     merged(TestInfo());
	PipelineList::ExportStats stats;
	merged.AddList(decoded, stats);
	const auto again = merged.Finish();
	CHECK(again.shaders.size() == 3 && again.files == file.files && again.shaders[0].file == PipelineList::RuntimeFileOf(again));
	// A merge with an executable list keeps both kinds apart.
	auto exe_shader = SourceOf(Code(50, 64), ShaderJournal::Kind::Compute, 4, InputSizes[2]);
	PipelineList::Builder exe(TestInfo());
	exe.AddJournals({exe_shader}, {{0, 0, {}}}, {},
	                [](std::span<const uint32_t>) { return PipelineList::Builder::Location {PipelineList::CodeFile::Executable, 128}; },
	                stats);
	PipelineList::Builder both(TestInfo());
	both.AddList(exe.Finish(), stats);
	both.AddList(decoded, stats);
	const auto mixed = both.Finish();
	CHECK(mixed.shaders.size() == 4 && mixed.shaders[0].file == PipelineList::CodeFile::Executable &&
	      mixed.shaders[1].file == PipelineList::RuntimeFileOf(mixed) && mixed.shaders[1].offset == 0);
	// A damaged Files section or a shader naming a missing file is still refused.
	auto bad = file;
	bad.shaders[0].file = static_cast<PipelineList::CodeFile>(2);
	PipelineList::File out;
	CHECK(!PipelineList::Decode(PipelineList::Encode(bad), out));
}

void TestResolveRuntime() {
	const auto list = MakeRuntimeList();
	MemoryStore store;
	Fill(store, list);
	// All in the store: every shader resolves with the recorded code; the game files are never read for them.
	{
		int                        reads = 0;
		PipelineList::ResolveStats stats;
		const auto resolved = PipelineList::Resolve(list.file, InputSizes, FailingRead(reads), {}, stats, store.Fn());
		CHECK(reads == 0);
		CHECK(stats.shaders == 3 && stats.resolved == 3 && stats.runtime_shaders == 3 && stats.runtime_resolved == 3 &&
		      stats.runtime_pending == 0 && stats.mismatched == 0 && stats.unreadable == 0 && stats.relocated == 0);
		CHECK(resolved.usable[0] && resolved.usable[1] && resolved.usable[2]);
		CHECK(!resolved.pending[0] && !resolved.pending[1] && !resolved.pending[2]);
		CHECK(resolved.sources[0].code == list.sources[0].code && resolved.sources[2].back_code == list.sources[2].back_code);
		CHECK(resolved.entries.size() == 3 && stats.permutations_kept == 3);
		CHECK(PermutationIdentity(resolved.sources[2], resolved.entries[2]) == PermutationIdentity(list.sources[2], list.entries[2]));
		CHECK(resolved.offsets[0] == 0 && resolved.back_offsets[2] == 0);
		CHECK(PipelineList::MostlyMatches(stats));
	}
	// The pixel shader's code is not in the store yet: pending, not unreadable, not mismatched, not relocated.
	{
		MemoryStore partial;
		Fill(partial, list);
		partial.codes.erase({PipelineList::CodeHash(Bytes(list.sources[1].code)).low, PipelineList::CodeHash(Bytes(list.sources[1].code)).high});
		int                        reads = 0;
		PipelineList::ResolveStats stats;
		const auto resolved = PipelineList::Resolve(list.file, InputSizes, FailingRead(reads), [](PipelineList::CodeFile) { return std::vector<uint8_t>(64, 1); }, stats, partial.Fn());
		CHECK(reads == 0 && stats.resolved == 2 && stats.runtime_pending == 1 && stats.runtime_resolved == 2);
		CHECK(stats.mismatched == 0 && stats.unreadable == 0 && stats.relocated == 0);
		CHECK(resolved.pending[1] && !resolved.pending[0] && !resolved.usable[1] && resolved.sources[1].code.empty());
		CHECK(resolved.entries.size() == 2 && resolved.permutation_entry[1] == UINT32_MAX);
		// Later, with the code added, the same list resolves fully (what the loader's re-resolve does).
		partial.Put(list.sources[1].code);
		PipelineList::ResolveStats again;
		const auto                 resolved2 = PipelineList::Resolve(list.file, InputSizes, FailingRead(reads), {}, again, partial.Fn());
		CHECK(again.resolved == 3 && again.runtime_pending == 0 && resolved2.usable[1] && !resolved2.pending[1]);
	}
	// No store function: all pending (a host without extraction); the list is still "usable" by the 50 % rule.
	{
		int                        reads = 0;
		PipelineList::ResolveStats stats;
		const auto resolved = PipelineList::Resolve(list.file, InputSizes, FailingRead(reads), {}, stats);
		CHECK(reads == 0 && stats.resolved == 0 && stats.runtime_pending == 3 && stats.mismatched == 0 && stats.unreadable == 0);
		CHECK(resolved.pending[0] && resolved.pending[1] && resolved.pending[2] && resolved.entries.empty());
		CHECK(PipelineList::MostlyMatches(stats));
	}
	// Merged stage: the front half in the store, the back half missing: the shader is pending as a whole.
	{
		MemoryStore front_only;
		front_only.Put(list.sources[2].code);
		PipelineList::ResolveStats stats;
		int                        reads = 0;
		const auto resolved = PipelineList::Resolve(list.file, InputSizes, FailingRead(reads), {}, stats, front_only.Fn());
		CHECK(resolved.pending[2] && !resolved.usable[2] && resolved.sources[2].code.empty() && resolved.sources[2].back_code.empty());
		CHECK(stats.runtime_pending == 3 - 0 && stats.resolved == 0);
		front_only.Put(list.sources[2].back_code);
		PipelineList::ResolveStats full;
		const auto                 resolved2 = PipelineList::Resolve(list.file, InputSizes, FailingRead(reads), {}, full, front_only.Fn());
		CHECK(resolved2.usable[2] && !resolved2.pending[2] && full.resolved == 1 && full.runtime_pending == 2 &&
		      resolved2.sources[2].back_code == list.sources[2].back_code);
	}
	// A store answer whose bytes are not the hashed code is a mismatch (a damaged store), not pending.
	{
		int                        reads = 0;
		PipelineList::ResolveStats stats;
		const auto wrong = [&](const Hash128&, uint32_t bytes, std::span<uint8_t> out) {
			std::fill(out.begin(), out.end(), uint8_t {7});
			return out.size() == bytes;
		};
		const auto resolved = PipelineList::Resolve(list.file, InputSizes, FailingRead(reads), {}, stats, wrong);
		CHECK(stats.resolved == 0 && stats.mismatched == 3 && stats.runtime_pending == 0 && !resolved.pending[0]);
		CHECK(!PipelineList::MostlyMatches(stats));
	}
	// A list whose prefix hash disagrees with the code the hash names is not resolved either.
	{
		auto altered = list.file;
		altered.shaders[0].prefix_hash ^= 1;
		MemoryStore                full;
		Fill(full, list);
		int                        reads = 0;
		PipelineList::ResolveStats stats;
		(void)PipelineList::Resolve(altered, InputSizes, FailingRead(reads), {}, stats, full.Fn());
		CHECK(stats.resolved == 2 && stats.mismatched == 1);
	}
	// The other input layout still wins over pending.
	{
		MemoryStore                none;
		int                        reads = 0;
		PipelineList::ResolveStats stats;
		(void)PipelineList::Resolve(list.file, {96, 72, 48}, FailingRead(reads), {}, stats, none.Fn());
		// m4: runtime_shaders counts the matching layouts only, so it equals resolved + pending.
		CHECK(stats.other_input_layout == 1 && stats.runtime_pending == 2 && stats.runtime_shaders == 2);
		CHECK(stats.runtime_shaders == stats.runtime_resolved + stats.runtime_pending);
	}
}

// m4: a runtime shader with another input layout does not make `--require-runtime 100` fail when every
// shader with the matching layout resolved.
void TestRuntimeShadersCountMatchingLayoutOnly() {
	const auto list = MakeRuntimeList();
	MemoryStore full;
	Fill(full, list);
	int                        reads = 0;
	PipelineList::ResolveStats stats;
	(void)PipelineList::Resolve(list.file, {96, 72, 48}, FailingRead(reads), {}, stats, full.Fn());
	CHECK(stats.other_input_layout == 1 && stats.runtime_pending == 0 && stats.runtime_resolved == 2);
	CHECK(stats.runtime_shaders == 2);
	CHECK(PipelineListTool::MeetsRuntimeRequirement(stats, 100));
}

// The 50 % rule counts non-pending shaders only; a list without `<runtime>` behaves as it always did.
void TestMostlyMatches() {
	PipelineList::ResolveStats s;
	s.shaders = 10;
	s.resolved = 5;
	CHECK(PipelineList::MostlyMatches(s));
	s.resolved = 4;
	CHECK(!PipelineList::MostlyMatches(s));
	s.runtime_shaders = 8;
	s.runtime_pending = 8; // 2 left to judge, 4 resolved is impossible: use a consistent set
	s.shaders         = 10;
	s.resolved        = 1;
	CHECK(PipelineList::MostlyMatches(s)); // 1 of the 2 non-pending
	s.resolved = 0;
	CHECK(!PipelineList::MostlyMatches(s));
	PipelineList::ResolveStats empty;
	CHECK(PipelineList::MostlyMatches(empty));
}

// A list without `<runtime>` (ASTRO): the hash function is never called, nothing is pending, the same results.
void TestNoRuntimeUnchanged() {
	// Executable shaders at known offsets.
	std::vector<uint8_t>               image(4096, 0xCD);
	std::vector<ShaderJournal::Source> sources;
	std::vector<uint64_t>              offsets;
	for (uint32_t i = 0; i < 3; i++) {
		sources.push_back(SourceOf(Code(10 + i, 100), i == 1 ? ShaderJournal::Kind::Pixel : ShaderJournal::Kind::Vertex, 1 + i,
		                           i == 1 ? InputSizes[1] : InputSizes[0]));
		image.resize(image.size() + 256, 0);
		offsets.push_back(image.size());
		image.insert(image.end(), Bytes(sources.back().code).begin(), Bytes(sources.back().code).end());
	}
	image.resize(image.size() + 512, 0xEE);
	PipelineList::Builder     builder(TestInfo());
	PipelineList::ExportStats stats;
	size_t                    next = 0;
	builder.AddJournals(sources, {{0, 0, {}}, {1, 0, {}}, {2, 0, {}}}, {},
	                    [&](std::span<const uint32_t>) {
		                    return PipelineList::Builder::Location {PipelineList::CodeFile::Executable, offsets[next++]};
	                    },
	                    stats);
	const auto file = builder.Finish();
	CHECK(file.files.empty() && PipelineList::RuntimeFileOf(file) == PipelineList::CodeFile::Executable);
	const PipelineList::ReadCodeFn read = [&](PipelineList::CodeFile, uint64_t offset, std::span<uint8_t> out) {
		if (offset > image.size() || out.size() > image.size() - offset) return false;
		std::memcpy(out.data(), image.data() + offset, out.size());
		return true;
	};
	MemoryStore         store;
	PipelineList::ResolveStats with_fn, without_fn;
	const auto                 a = PipelineList::Resolve(file, InputSizes, read, [&](PipelineList::CodeFile) { return image; }, with_fn, store.Fn());
	const auto                 b = PipelineList::Resolve(file, InputSizes, read, [&](PipelineList::CodeFile) { return image; }, without_fn);
	CHECK(store.calls == 0);
	CHECK(with_fn.resolved == 3 && without_fn.resolved == 3 && with_fn.runtime_shaders == 0 && with_fn.runtime_pending == 0 &&
	      with_fn.runtime_resolved == 0);
	CHECK(a.usable == b.usable && a.entries.size() == b.entries.size() && a.offsets == b.offsets);
	CHECK(a.pending == std::vector<bool>(3, false));
	// The encoded list of an executable-only export has no Files section (as before).
	std::vector<std::pair<uint32_t, std::vector<uint8_t>>> sections;
	CHECK(PipelineList::DecodeSections(PipelineList::Encode(file), sections));
	CHECK(std::none_of(sections.begin(), sections.end(), [](const auto& s) { return s.first == static_cast<uint32_t>(PipelineList::SectionId::Files); }));
}

// A mixed list: executable shaders resolve through the files as before, runtime shaders through the store.
void TestMixedList() {
	std::vector<uint8_t> image(4096, 0xCD);
	const auto           exe = SourceOf(Code(20, 100), ShaderJournal::Kind::Vertex, 1, InputSizes[0]);
	const auto           rt1 = SourceOf(Code(21, 100), ShaderJournal::Kind::Vertex, 1, InputSizes[0]);
	const auto           rt2 = SourceOf(Code(22, 100), ShaderJournal::Kind::Vertex, 1, InputSizes[0]);
	const auto           rt3 = SourceOf(Code(23, 100), ShaderJournal::Kind::Vertex, 1, InputSizes[0]);
	image.resize(image.size() + 100, 0);
	const uint64_t offset = image.size();
	image.insert(image.end(), Bytes(exe.code).begin(), Bytes(exe.code).end());
	PipelineList::Builder builder(TestInfo());
	std::set<PipelineListTool::HashKey> kept;
	PipelineList::ExportStats           stats;
	const PipelineList::Builder::LocateFn inner = [&](std::span<const uint32_t> code) -> std::optional<PipelineList::Builder::Location> {
		if (code.size() == exe.code.size() && std::equal(code.begin(), code.end(), exe.code.begin())) {
			return PipelineList::Builder::Location {PipelineList::CodeFile::Executable, offset};
		}
		return std::nullopt;
	};
	builder.AddJournals({exe, rt1, rt2, rt3}, {{0, 0, {}}, {1, 0, {}}, {2, 0, {}}, {3, 0, {}}}, {},
	                    PipelineListTool::KeepRuntimeCode(inner, builder, kept), stats);
	CHECK(kept.size() == 3);
	const auto file = builder.Finish();
	const PipelineList::ReadCodeFn read = [&](PipelineList::CodeFile which, uint64_t at, std::span<uint8_t> out) {
		if (which != PipelineList::CodeFile::Executable || at > image.size() || out.size() > image.size() - at) return false;
		std::memcpy(out.data(), image.data() + at, out.size());
		return true;
	};
	// No store: 1 resolved, 3 pending: the list is used (the 50 % rule counts the executable shader only).
	{
		PipelineList::ResolveStats s;
		const auto resolved = PipelineList::Resolve(file, InputSizes, read, {}, s);
		CHECK(s.shaders == 4 && s.resolved == 1 && s.runtime_pending == 3 && PipelineList::MostlyMatches(s));
		CHECK(resolved.usable[0] && !resolved.pending[0] && resolved.pending[1]);
	}
	// The executable shader changed (another game version) and the store is empty: 0 of 1 non-pending: not used.
	{
		auto changed = image;
		changed[offset + 5] ^= 1;
		const PipelineList::ReadCodeFn bad = [&](PipelineList::CodeFile which, uint64_t at, std::span<uint8_t> out) {
			if (which != PipelineList::CodeFile::Executable || at > changed.size() || out.size() > changed.size() - at) return false;
			std::memcpy(out.data(), changed.data() + at, out.size());
			return true;
		};
		PipelineList::ResolveStats s;
		(void)PipelineList::Resolve(file, InputSizes, bad, [&](PipelineList::CodeFile) { return changed; }, s);
		CHECK(s.resolved == 0 && s.mismatched == 1 && s.runtime_pending == 3 && !PipelineList::MostlyMatches(s));
	}
}

// ---- the code store -------------------------------------------------------------------------------------------

struct TempDir {
	std::filesystem::path path;
	TempDir() {
		path = std::filesystem::temp_directory_path() / ("kyty_ue5warm_tests_" + std::to_string(::getpid()));
		std::filesystem::remove_all(path);
		std::filesystem::create_directories(path);
	}
	~TempDir() {
		std::error_code error;
		std::filesystem::remove_all(path, error);
	}
};

Hash128 HashOf(const std::vector<uint32_t>& code) {
	return PipelineList::CodeHash(Bytes(code));
}

std::vector<uint8_t> ReadAll(const std::filesystem::path& path) {
	std::ifstream in(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void WriteAll(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

bool AddCode(PipelineCodeStore& store, const std::vector<uint32_t>& code) {
	return store.Add(HashOf(code), PipelineList::CodePrefixHash(Bytes(code)), Bytes(code));
}

bool FindIs(PipelineCodeStore& store, const std::vector<uint32_t>& code) {
	const auto hash = HashOf(code);
	if (store.Size(hash) != code.size() * 4) return false;
	std::vector<uint8_t> out(code.size() * 4);
	return store.Find(hash, out) && std::equal(out.begin(), out.end(), Bytes(code).begin());
}

void TestStoreBasics() {
	TempDir     dir;
	const auto  path = dir.path / "PPSA00000.codestore";
	const auto  a = Code(100, 300), b = Code(101, 17), c = Code(102, 4096);
	{
		PipelineCodeStore store;
		CHECK(store.Open(path, "PPSA00000", "02.003.004", 0x1122334455667788ull));
		CHECK(store.Count() == 0 && !store.DiscardedStale() && store.FileBytes() == PipelineCodeStore::HeaderBytes);
		CHECK(AddCode(store, a) && AddCode(store, b) && AddCode(store, c));
		CHECK(store.Count() == 3 && store.Size(HashOf(a)) == a.size() * 4 && store.Size(Hash128 {1, 2}) == 0);
		// Dedupe: the same hash again is ignored, nothing written.
		const auto before = store.FileBytes();
		CHECK(!AddCode(store, a) && store.Count() == 3 && store.FileBytes() == before && store.DuplicatesIgnored() == 1);
		// Find before any flush (the store reads its own appends); wrong size is not found.
		CHECK(FindIs(store, a) && FindIs(store, b) && FindIs(store, c));
		std::vector<uint8_t> wrong(8);
		CHECK(!store.Find(HashOf(a), wrong));
		CHECK(!store.Find(Hash128 {9, 9}, wrong));
		CHECK(store.ContainsAll({HashOf(a), HashOf(c)}) && store.ContainsAll({}) && !store.ContainsAll({HashOf(a), Hash128 {5, 5}}));
		CHECK(store.CodeBytes() == (a.size() + b.size() + c.size()) * 4);
		// An empty code is refused.
		CHECK(!store.Add(Hash128 {7, 7}, 0, {}));
		CHECK(store.Flush());
		// ReadFn works with Resolve's convention.
		const auto fn = store.ReadFn();
		std::vector<uint8_t> out(b.size() * 4);
		CHECK(fn(HashOf(b), static_cast<uint32_t>(out.size()), out) && std::equal(out.begin(), out.end(), Bytes(b).begin()));
		CHECK(!fn(HashOf(b), 4, std::span<uint8_t>(out).first(4)));
	}
	// The header: magic, version, header bytes, stamps, fingerprint.
	{
		const auto file = ReadAll(path);
		CHECK(file.size() > 64 && std::memcmp(file.data(), "KYCODE1\0", 8) == 0);
		uint32_t version = 0, header = 0;
		uint64_t fp = 0;
		std::memcpy(&version, file.data() + 8, 4);
		std::memcpy(&header, file.data() + 12, 4);
		std::memcpy(&fp, file.data() + 48, 8);
		CHECK(version == 1 && header == 64 && fp == 0x1122334455667788ull);
		CHECK(std::string(reinterpret_cast<const char*>(file.data() + 16)) == "PPSA00000" &&
		      std::string(reinterpret_cast<const char*>(file.data() + 32)) == "02.003.004");
		uint32_t record_magic = 0, record_bytes = 0;
		std::memcpy(&record_magic, file.data() + 64, 4);
		std::memcpy(&record_bytes, file.data() + 68, 4);
		CHECK(record_magic == 0x45444F43u && record_bytes == a.size() * 4);
	}
	// Reopen with the same stamps: everything is still there; appends continue.
	{
		PipelineCodeStore store;
		CHECK(store.Open(path, "PPSA00000", "02.003.004", 0x1122334455667788ull));
		CHECK(store.Count() == 3 && !store.DiscardedStale() && store.TruncatedBytes() == 0);
		CHECK(FindIs(store, a) && FindIs(store, b) && FindIs(store, c));
		const auto d = Code(103, 50);
		CHECK(AddCode(store, d) && !AddCode(store, b));
	}
	{
		PipelineCodeStore store;
		CHECK(store.Open(path, "PPSA00000", "02.003.004", 0x1122334455667788ull));
		CHECK(store.Count() == 4 && FindIs(store, Code(103, 50)));
	}
}

void TestStoreDamage() {
	TempDir    dir;
	const auto path = dir.path / "s.codestore";
	const auto a = Code(200, 64), b = Code(201, 65), c = Code(202, 66);
	std::vector<uint8_t> intact;
	{
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5));
		CHECK(AddCode(store, a) && AddCode(store, b));
		CHECK(store.Flush());
		intact = ReadAll(path);
		CHECK(AddCode(store, c));
	}
	const auto full = ReadAll(path);
	CHECK(full.size() > intact.size());
	// A truncated tail (the last record cut): the intact prefix stays, the damage is cut off, appends go on.
	for (const size_t cut: {size_t {1}, size_t {9}, full.size() - intact.size() - 1}) {
		auto torn = full;
		torn.resize(full.size() - cut);
		WriteAll(path, torn);
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5));
		CHECK(store.Count() == 2 && FindIs(store, a) && FindIs(store, b) && store.Size(HashOf(c)) == 0);
		CHECK(store.TruncatedBytes() == torn.size() - intact.size() && !store.DiscardedStale());
		CHECK(std::filesystem::file_size(path) == intact.size());
		CHECK(AddCode(store, c) && FindIs(store, c));
	}
	// A flipped byte in the last record's code: that record and nothing before it is lost.
	{
		auto flipped = full;
		flipped[intact.size() + 40] ^= 0x10;
		WriteAll(path, flipped);
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5));
		CHECK(store.Count() == 2 && FindIs(store, a) && FindIs(store, b));
	}
	// A flipped byte in the first record: nothing after it is trusted (the scan stops at damage).
	{
		auto flipped = full;
		flipped[64 + 40] ^= 0x10;
		WriteAll(path, flipped);
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5));
		CHECK(store.Count() == 0 && std::filesystem::file_size(path) == PipelineCodeStore::HeaderBytes);
		CHECK(AddCode(store, a) && FindIs(store, a));
	}
	// Garbage after the records (a record header that is no record).
	{
		auto junk = full;
		junk.insert(junk.end(), 40, 0xAB);
		WriteAll(path, junk);
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5));
		CHECK(store.Count() == 3 && store.TruncatedBytes() == 40);
	}
}

void TestStoreStaleHeader() {
	TempDir    dir;
	const auto path = dir.path / "s.codestore";
	const auto a = Code(300, 64);
	const auto stale = std::filesystem::path(path.string() + ".stale");
	struct Variant {
		const char* title;
		const char* version;
		uint64_t    fp;
	};
	for (const Variant& other: {Variant {"T2", "V", 5}, Variant {"T", "V2", 5}, Variant {"T", "V", 6}}) {
		std::filesystem::remove(path);
		std::filesystem::remove(stale);
		{
			PipelineCodeStore store;
			CHECK(store.Open(path, "T", "V", 5) && AddCode(store, a));
		}
		const auto before = ReadAll(path);
		PipelineCodeStore store;
		CHECK(store.Open(path, other.title, other.version, other.fp));
		CHECK(store.DiscardedStale() && store.Count() == 0 && std::filesystem::exists(stale) && ReadAll(stale) == before);
		CHECK(store.Size(HashOf(a)) == 0 && std::filesystem::file_size(path) == PipelineCodeStore::HeaderBytes);
		CHECK(AddCode(store, Code(301, 10)));
	}
	// A file that is no store at all (or too short) is set aside the same way; an empty file just starts a store.
	{
		std::filesystem::remove(stale);
		WriteAll(path, std::vector<uint8_t>(10, 1));
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5) && store.DiscardedStale() && std::filesystem::exists(stale));
	}
	{
		std::filesystem::remove(stale);
		WriteAll(path, {});
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5) && !store.DiscardedStale() && !std::filesystem::exists(stale));
	}
	// Long stamps are cut to 16 bytes on both sides, so a long title reopens.
	{
		std::filesystem::remove(path);
		const std::string title = "a-very-long-title-id-0123456789";
		{
			PipelineCodeStore store;
			CHECK(store.Open(path, title, "02.003.004", 1) && AddCode(store, a));
		}
		PipelineCodeStore store;
		CHECK(store.Open(path, title, "02.003.004", 1) && !store.DiscardedStale() && store.Count() == 1);
	}
}

void TestStoreCapAndReadOnly() {
	TempDir    dir;
	const auto path = dir.path / "s.codestore";
	{
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5));
		// 4 MiB codes: the 64 MiB cap leaves room for 15.
		size_t added = 0;
		for (uint32_t i = 0; i < 17; i++) {
			std::vector<uint32_t> code(1u << 20u);
			std::mt19937          random(1000 + i);
			for (auto& word: code) word = random();
			added += AddCode(store, code) ? 1 : 0;
		}
		CHECK(added == 15 && store.Count() == 15 && store.FileBytes() <= PipelineCodeStore::CapBytes);
		// A small one still fits in what is left.
		CHECK(AddCode(store, Code(400, 16)) && store.Count() == 16);
	}
	CHECK(std::filesystem::file_size(path) <= PipelineCodeStore::CapBytes);
	// Read-only open: any stamps, never changes the file, cannot add.
	{
		const auto before = ReadAll(path);
		PipelineCodeStore reader;
		CHECK(reader.OpenReadOnly(path) && reader.Count() == 16 && reader.DecoderFingerprint() == 5);
		CHECK(!AddCode(reader, Code(401, 16)) && FindIs(reader, Code(400, 16)));
		CHECK(ReadAll(path) == before);
	}
	// Read-only with a damaged tail leaves the damage in the file.
	{
		auto torn = ReadAll(path);
		torn.resize(torn.size() - 3);
		WriteAll(path, torn);
		PipelineCodeStore reader;
		CHECK(reader.OpenReadOnly(path) && reader.Count() == 15 && reader.TruncatedBytes() != 0);
		CHECK(std::filesystem::file_size(path) == torn.size());
	}
	// Not a store.
	{
		WriteAll(path, std::vector<uint8_t>(100, 0));
		PipelineCodeStore reader;
		CHECK(!reader.OpenReadOnly(path));
		CHECK(!reader.OpenReadOnly(dir.path / "missing"));
	}
}

// Search marks: persisted as the last record; a later Add overwrites it (Flush writes it again); a damaged mark is cut.
void TestStoreSearchMarks() {
	TempDir    dir;
	const auto path = dir.path / "m.codestore";
	using Mark      = PipelineCodeStore::SearchMark;
	{
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5) && store.Mark().state == Mark::None);
		CHECK(AddCode(store, Code(1, 64)));
		Mark partial;
		partial.state = Mark::Partial;
		partial.key   = 0xabcdef;
		partial.done  = {{0, 0, 9}, {2, 4, 5}};
		CHECK(store.WriteMark(partial));
		CHECK(store.Mark().done.size() == 2);
		Mark complete;
		complete.state = Mark::Complete;
		complete.key   = 0xabcdef;
		CHECK(store.WriteMark(complete)); // replaces the partial one in place
	}
	{
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5) && store.Count() == 1 && store.TruncatedBytes() == 0);
		const auto mark = store.Mark();
		CHECK(mark.state == Mark::Complete && mark.key == 0xabcdef && mark.done.empty());
		// a code after the mark overwrites it; the mark in memory survives and Flush writes it back
		CHECK(AddCode(store, Code(2, 64)) && store.Mark().state == Mark::Complete);
		CHECK(store.Flush());
	}
	{
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5) && store.Count() == 2 && store.Mark().state == Mark::Complete && store.TruncatedBytes() == 0);
	}
	{
		// a torn mark: the codes stay, the mark is gone
		auto bytes = ReadAll(path);
		bytes.resize(bytes.size() - 5);
		WriteAll(path, bytes);
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5) && store.Count() == 2 && store.Mark().state == Mark::None && store.TruncatedBytes() != 0);
	}
	{
		// read-only stores take no mark
		PipelineCodeStore store;
		CHECK(store.Open(path, "T", "V", 5));
		Mark complete;
		complete.state = Mark::Complete;
		complete.key   = 7;
		CHECK(store.WriteMark(complete) && store.Flush());
		PipelineCodeStore reader;
		CHECK(reader.OpenReadOnly(path) && reader.Mark().key == 7 && !reader.WriteMark(complete));
	}
	{
		// the largest mark fits in a store that is filled up to its reserve (the cap case, really filled)
		const auto       full_path = dir.path / "full.codestore";
		PipelineCodeStore store;
		CHECK(store.Open(full_path, "T", "V", 5));
		std::mt19937 random(11);
		uint32_t     seed = 0;
		const auto   add  = [&](size_t bytes) {
            std::vector<uint8_t> code(bytes);
            for (auto& b: code) b = static_cast<uint8_t>(random());
            return store.Add({++seed, 5}, seed, code);
		};
		for (int i = 0; i < 3; i++) CHECK(add(16u << 20u));
		const uint64_t room = PipelineCodeStore::CapBytes - PipelineCodeStore::MarkReserveBytes - store.FileBytes() - 40;
		CHECK(add(static_cast<size_t>(room)) && store.Full());
		CHECK(!add(64)); // the store really is at its cap: another code does not fit
		Mark partial;
		partial.state = Mark::Partial;
		partial.key   = 99;
		for (uint32_t i = 0; i < PipelineCodeStore::MaxMarkRanges; i++) partial.done.push_back({i, 2 * i, 2 * i + 1});
		CHECK(store.WriteMark(partial) && store.Flush());
		CHECK(store.FileBytes() <= PipelineCodeStore::CapBytes);
		partial.done.push_back({9999, 0, 1}); // one range too many is refused, the earlier mark stays
		CHECK(!store.WriteMark(partial) && store.Mark().done.size() == PipelineCodeStore::MaxMarkRanges);
		PipelineCodeStore reader;
		CHECK(reader.OpenReadOnly(full_path) && reader.Mark().state == Mark::Partial &&
		      reader.Mark().done.size() == PipelineCodeStore::MaxMarkRanges && reader.Mark().done[4095].library == 4095 &&
		      reader.Count() == 4);
		CHECK(std::filesystem::file_size(full_path) <= PipelineCodeStore::CapBytes);
	}
}

void TestStoreThreads() {
	TempDir    dir;
	const auto path = dir.path / "s.codestore";
	PipelineCodeStore store;
	CHECK(store.Open(path, "T", "V", 5));
	std::vector<std::vector<uint32_t>> codes;
	for (uint32_t i = 0; i < 40; i++) codes.push_back(Code(500 + i, 64 + i));
	std::vector<std::thread> threads;
	std::atomic<int>         added {0};
	for (int t = 0; t < 4; t++) {
		threads.emplace_back([&] {
			for (const auto& code: codes) added += AddCode(store, code) ? 1 : 0;
			for (const auto& code: codes) (void)FindIs(store, code);
		});
	}
	for (auto& thread: threads) thread.join();
	CHECK(added == 40 && store.Count() == 40);
	for (const auto& code: codes) CHECK(FindIs(store, code));
	CHECK(store.Flush());
	PipelineCodeStore again;
	CHECK(again.Open(path, "T", "V", 5) && again.Count() == 40);
}

// The store behind Resolve: the same flow the tool's check and the loader use.
void TestStoreWithResolve() {
	TempDir    dir;
	const auto list = MakeRuntimeList();
	const auto path = dir.path / "s.codestore";
	{
		PipelineCodeStore store;
		CHECK(store.Open(path, "PPSA00000", "01.000.000", 7));
		// Only the front half of the merged stage and the vertex shader are known so far.
		CHECK(AddCode(store, list.sources[0].code) && AddCode(store, list.sources[2].code));
		int                        reads = 0;
		PipelineList::ResolveStats stats;
		const auto resolved = PipelineList::Resolve(list.file, InputSizes, FailingRead(reads), {}, stats, store.ReadFn());
		CHECK(stats.resolved == 1 && stats.runtime_pending == 2 && resolved.usable[0]);
		CHECK(PipelineListTool::RuntimeLine(stats) == "runtime: 1 resolved, 2 pending");
		CHECK(!PipelineListTool::MeetsRuntimeRequirement(stats, 100) && !PipelineListTool::MeetsRuntimeRequirement(stats, 34) &&
		      PipelineListTool::MeetsRuntimeRequirement(stats, 33) && PipelineListTool::MeetsRuntimeRequirement(stats, 0));
		CHECK(AddCode(store, list.sources[1].code) && AddCode(store, list.sources[2].back_code));
	}
	PipelineCodeStore reader;
	CHECK(reader.OpenReadOnly(path));
	int                        reads = 0;
	PipelineList::ResolveStats stats;
	(void)PipelineList::Resolve(list.file, InputSizes, FailingRead(reads), {}, stats, reader.ReadFn());
	CHECK(stats.resolved == 3 && stats.runtime_pending == 0 && PipelineListTool::MeetsRuntimeRequirement(stats, 100));
	CHECK(PipelineListTool::RuntimeLine(stats) == "runtime: 3 resolved, 0 pending");
	// The skip rule's input: the store holds every listed hash.
	std::vector<Hash128> listed;
	for (const auto& shader: list.file.shaders) listed.push_back({shader.code_hash_low, shader.code_hash_high});
	CHECK(reader.ContainsAll(listed));
	listed.push_back({1, 1});
	CHECK(!reader.ContainsAll(listed));
}

// retarget carries `<runtime>` shaders over unchanged, pending ones included; a game-file shader that is gone is dropped.
void TestRetargetRuntime() {
	// One executable shader (gone in the other game version) and the runtime list's shaders.
	auto list = MakeRuntimeList();
	PipelineList::Shader exe;
	exe.file = PipelineList::CodeFile::Executable;
	exe.offset = 64;
	exe.code_size = 8;
	exe.input_info.assign(InputSizes[0], 0);
	exe.kind = ShaderJournal::Kind::Vertex;
	list.file.shaders.push_back(exe);
	list.file.permutations.push_back({3, 0, {}});
	MemoryStore store;
	Fill(store, list);
	for (const bool with_store: {false, true}) {
		int                        reads = 0;
		PipelineList::ResolveStats stats;
		const auto resolved = PipelineList::Resolve(list.file, InputSizes, FailingRead(reads), {}, stats, with_store ? store.Fn() : PipelineList::ReadHashFn {});
		PipelineList::RetargetStats rs;
		const auto                  out = PipelineList::Retarget(list.file, resolved, rs);
		CHECK(out.shaders.size() == 3 && out.permutations.size() == 3 && rs.missing == 1 && rs.same_offset == 3 && rs.moved == 0);
		for (const auto& shader: out.shaders) CHECK(shader.file == PipelineList::RuntimeFileOf(list.file) && shader.offset == 0);
		CHECK(out.shaders[0].code_hash_low == list.file.shaders[0].code_hash_low && out.shaders[2].back_code_size == 96);
		CHECK(PipelineList::RuntimeFileOf(out) == PipelineList::RuntimeFileOf(list.file));
		// export --base: the retargeted base joins a new export, runtime shaders included, and the same codes seen again
		// by a journal are the same shaders (no duplicates).
		if (!with_store) {
			PipelineList::Builder               builder(TestInfo());
			PipelineList::ExportStats           stats2;
			builder.AddList(out, stats2);
			std::set<PipelineListTool::HashKey> kept;
			builder.AddJournals(list.sources, list.entries, {},
			                    PipelineListTool::KeepRuntimeCode([](std::span<const uint32_t>) { return std::nullopt; }, builder, kept),
			                    stats2);
			const auto joined = builder.Finish();
			CHECK(joined.shaders.size() == 3 && joined.permutations.size() == 3 && joined.files == out.files);
		}
		PipelineList::File decoded; // the retargeted list is a valid list
		CHECK(PipelineList::Decode(PipelineList::Encode(out), decoded) && decoded.shaders.size() == 3);
	}
}

// A merged stage with one runtime half and one half in the game files is counted (it is dropped by the export).
void TestMixedMergedCount() {
	auto merged = SourceOf(Code(700, 64), ShaderJournal::Kind::Vertex, 5, InputSizes[0]);
	merged.back_code = Code(701, 32);
	const auto plain = SourceOf(Code(702, 64), ShaderJournal::Kind::Vertex, 5, InputSizes[0]);
	const auto h = PipelineList::CodeHash(Bytes(merged.back_code));
	std::set<PipelineListTool::HashKey> kept {{h.low, h.high}}; // only the back half is runtime
	CHECK(PipelineListTool::CountMixedMergedStages({merged, plain}, kept) == 1);
	const auto f = PipelineList::CodeHash(Bytes(merged.code));
	kept.insert({f.low, f.high}); // both runtime: not mixed
	CHECK(PipelineListTool::CountMixedMergedStages({merged, plain}, kept) == 0);
	CHECK(PipelineListTool::CountMixedMergedStages({merged}, {}) == 0);
}

// ---- the group split ------------------------------------------------------------------------------------------

struct GroupShader {
	std::vector<uint8_t> code;
	uint32_t             declared_size = 0xFFFFFFFFu; // the size at marker-0x0c; 0xFFFFFFFF: code.size()
	int64_t              code_offset_delta = 0;       // added to the true offset (out of bounds tests)
	bool                 header = true;               // false: the shader has no marker
	uint32_t             header_size   = 0xFFFFFFFFu; // the header's shader_size (marker+0x44); 0xFFFFFFFF: code.size()
};

// One shader: preamble (with the code offset 0x14 before the marker), the header, padding, the ISA, a footer.
void AppendShader(std::vector<uint8_t>& group, const GroupShader& shader, std::mt19937& random) {
	const auto put32 = [&](size_t at, uint32_t value) { std::memcpy(group.data() + at, &value, 4); };
	for (int i = 0; i < 24; i++) group.push_back(static_cast<uint8_t>(random() | 0x80)); // preamble, no marker bytes
	if (!shader.header) {
		group.insert(group.end(), shader.code.begin(), shader.code.end());
		return;
	}
	const size_t marker = group.size();
	const uint32_t header_to_code = 0x100;
	group.resize(marker + 0x100 + shader.code.size() + 48, 0);
	const uint8_t marker_bytes[8] = {'1', '2', '3', '4', 0x18, 0, 0, 0};
	std::memcpy(group.data() + marker, marker_bytes, 8);
	put32(marker - 0x14, static_cast<uint32_t>(static_cast<int64_t>(header_to_code) + shader.code_offset_delta));
	put32(marker - 0x0c, shader.declared_size == 0xFFFFFFFFu ? static_cast<uint32_t>(shader.code.size()) : shader.declared_size);
	put32(marker + 0x44, shader.header_size == 0xFFFFFFFFu ? static_cast<uint32_t>(shader.code.size()) : shader.header_size);
	std::memcpy(group.data() + marker + header_to_code, shader.code.data(), shader.code.size());
	const char footer[] = "sl00barefoot";
	std::memcpy(group.data() + marker + header_to_code + shader.code.size(), footer, sizeof(footer) - 1);
}

std::vector<uint8_t> RandomCode(uint32_t seed, size_t bytes) {
	std::mt19937         random(seed);
	std::vector<uint8_t> code(bytes);
	for (auto& byte: code) byte = static_cast<uint8_t>(random() | 0x01u); // never the all-zero or marker lead byte pattern
	for (auto& byte: code) {
		if (byte == '1') byte = '2';
	}
	return code;
}

void TestGroupSplit() {
	std::mt19937         random(1);
	const auto           c0 = RandomCode(1, 400), c1 = RandomCode(2, 40), c2 = RandomCode(3, 256);
	std::vector<uint8_t> group;
	AppendShader(group, {c0}, random);
	AppendShader(group, {c1}, random);
	AppendShader(group, {c2}, random);
	const std::vector<uint32_t> frequencies = {0x1A5, 0x3, 0x2};
	Ue5Group::Stats             stats;
	const auto                  shaders = Ue5Group::Split(group, frequencies, &stats);
	CHECK(shaders.size() == 3 && stats.markers == 3 && stats.shaders == 3 && stats.matched && stats.out_of_bounds == 0 && stats.bad_size == 0);
	const std::vector<uint8_t>* want[] = {&c0, &c1, &c2};
	for (size_t i = 0; i < shaders.size() && i < 3; i++) {
		CHECK(shaders[i].size == want[i]->size() && std::equal(shaders[i].code.begin(), shaders[i].code.end(), want[i]->begin()));
		const auto hash = PipelineList::CodeHash(*want[i]);
		CHECK(shaders[i].hash == hash && shaders[i].prefix == PipelineList::CodePrefixHash(*want[i]));
		// The spans point into the group.
		CHECK(shaders[i].code.data() >= group.data() && shaders[i].code.data() + shaders[i].code.size() <= group.data() + group.size());
	}
	CHECK(shaders.size() == 3 && shaders[0].stage == 5 && shaders[1].stage == 3 && shaders[2].stage == 2); // low 4 bits
	// The hash is the key a KYPLST shader uses: a 40-byte code (under the 64-byte prefix) keys the same way.
	CHECK(shaders.size() == 3 && shaders[1].prefix == XXH3_64bits(c1.data(), c1.size()));

	// Both size fields set the same: no mismatch. They differ: the marker-0x0c size wins and the shader is counted.
	CHECK(stats.size_mismatch == 0);
	{
		std::vector<uint8_t> g;
		AppendShader(g, {c0, 0xFFFFFFFFu, 0, true, 0xFFFFFFFFu}, random);
		AppendShader(g, {c1, 0xFFFFFFFFu, 0, true, 128}, random);
		Ue5Group::Stats s;
		const auto      r = Ue5Group::Split(g, {}, &s);
		CHECK(r.size() == 2 && s.size_mismatch == 1 && r[1].size == c1.size() && r[1].hash == PipelineList::CodeHash(c1));
	}
	// No frequencies: the shaders are split, the stage is unknown.
	{
		Ue5Group::Stats s;
		const auto      all = Ue5Group::Split(group, {}, &s);
		CHECK(all.size() == 3 && !s.matched && all[0].stage == Ue5Group::UnknownStage);
	}
	// More library entries than headers (an RT-style layout): still split, stages unknown, flagged.
	{
		const std::vector<uint32_t> too_many = {1, 2, 3, 4};
		Ue5Group::Stats             s;
		const auto                  all = Ue5Group::Split(group, too_many, &s);
		CHECK(all.size() == 3 && !s.matched && s.markers == 3 && all[1].stage == Ue5Group::UnknownStage);
	}
	// No marker at all: nothing, nothing dropped.
	{
		std::vector<uint8_t> none;
		AppendShader(none, {c0, 0xFFFFFFFFu, 0, false}, random);
		Ue5Group::Stats s;
		CHECK(Ue5Group::Split(none, frequencies, &s).empty() && s.markers == 0 && s.shaders == 0);
		CHECK(Ue5Group::Split({}, {}, &s).empty());
	}
	// A code outside the group is dropped and counted; its neighbours stay, and keep their library entries.
	{
		std::vector<uint8_t> g;
		AppendShader(g, {c0}, random);
		AppendShader(g, {c1, 0xFFFFFFFFu, 100000}, random);
		AppendShader(g, {c2}, random);
		Ue5Group::Stats s;
		const auto      r = Ue5Group::Split(g, frequencies, &s);
		CHECK(r.size() == 2 && s.out_of_bounds == 1 && s.markers == 3 && s.matched);
		CHECK(r.size() == 2 && r[0].stage == 5 && r[1].stage == 2 && r[1].hash == PipelineList::CodeHash(c2));
	}
	// A size running past the end is out of bounds too; size 0 and a size that is not a multiple of 4 are bad.
	{
		std::vector<uint8_t> g;
		AppendShader(g, {c0, 1u << 30u}, random);
		AppendShader(g, {c1, 0}, random);
		AppendShader(g, {c2, 255}, random);
		AppendShader(g, {c2}, random);
		Ue5Group::Stats s;
		const auto      r = Ue5Group::Split(g, {}, &s);
		CHECK(r.size() == 1 && s.out_of_bounds == 1 && s.bad_size == 2 && s.markers == 4);
	}
	// The marker bytes inside a shader's ISA are not a header.
	{
		auto inside = c0;
		const uint8_t marker_bytes[8] = {'1', '2', '3', '4', 0x18, 0, 0, 0};
		std::memcpy(inside.data() + 100, marker_bytes, 8);
		std::vector<uint8_t> g;
		AppendShader(g, {inside}, random);
		AppendShader(g, {c1}, random);
		Ue5Group::Stats s;
		const auto      r = Ue5Group::Split(g, {}, &s);
		CHECK(r.size() == 2 && s.markers == 2 && r[0].size == inside.size() && r[0].hash == PipelineList::CodeHash(inside));
	}
	// A header cut off at the end of the group (no room for its fields) and a marker too close to the start: no crash.
	{
		std::vector<uint8_t> g(8, 0);
		const uint8_t        marker_bytes[8] = {'1', '2', '3', '4', 0x18, 0, 0, 0};
		std::memcpy(g.data(), marker_bytes, 8);
		Ue5Group::Stats s;
		CHECK(Ue5Group::Split(g, {}, &s).empty() && s.out_of_bounds == 1);
		g.assign(40, 0);
		std::memcpy(g.data() + 30, marker_bytes, 8);
		g.resize(37);
		CHECK(Ue5Group::Split(g, {}, &s).empty());
	}
}

// ---- the tool's helpers ---------------------------------------------------------------------------------------

void TestToolRuntimeExport() {
	const auto found   = SourceOf(Code(600, 80), ShaderJournal::Kind::Compute, 4, InputSizes[2]);
	const auto missing = SourceOf(Code(601, 90), ShaderJournal::Kind::Compute, 4, InputSizes[2]);
	const std::vector<ShaderJournal::Entry> entries = {{0, 0, {}}, {1, 0, {}}, {1, 4, {}}};
	const PipelineList::Builder::LocateFn   inner   = [&](std::span<const uint32_t> code) -> std::optional<PipelineList::Builder::Location> {
        if (std::equal(code.begin(), code.end(), found.code.begin(), found.code.end())) {
            return PipelineList::Builder::Location {PipelineList::CodeFile::Executable, 77};
        }
        return std::nullopt;
	};
	// Without --runtime-code the code found nowhere is dropped, with its permutations (as before).
	{
		PipelineList::Builder     builder(TestInfo());
		PipelineList::ExportStats stats;
		builder.AddJournals({found, missing}, entries, {}, inner, stats);
		const auto file = builder.Finish();
		CHECK(file.shaders.size() == 1 && file.permutations.size() == 1 && file.files.empty() && stats.sources_unlocated == 1);
	}
	// With it, the code is kept as `<runtime>`: counted once per distinct code, offset 0, the file added to the list.
	{
		PipelineList::Builder               builder(TestInfo());
		std::set<PipelineListTool::HashKey> kept;
		PipelineList::ExportStats           stats;
		builder.AddJournals({found, missing}, entries, {}, PipelineListTool::KeepRuntimeCode(inner, builder, kept), stats);
		const auto file = builder.Finish();
		CHECK(kept.size() == 1 && file.shaders.size() == 2 && file.permutations.size() == 3 && stats.sources_unlocated == 0);
		CHECK(file.files.size() == 1 && file.files[0] == "<runtime>");
		CHECK(file.shaders[0].file == PipelineList::CodeFile::Executable && file.shaders[0].offset == 77);
		CHECK(file.shaders[1].file == PipelineList::RuntimeFileOf(file) && file.shaders[1].offset == 0 &&
		      file.shaders[1].code_size == missing.code.size());
		const auto hash = PipelineList::CodeHash(Bytes(missing.code));
		CHECK(kept.contains({hash.low, hash.high}) && file.shaders[1].code_hash_low == hash.low &&
		      file.shaders[1].prefix_hash == PipelineList::CodePrefixHash(Bytes(missing.code)));
	}
	// Everything found: --runtime-code adds no `<runtime>` file and keeps nothing.
	{
		PipelineList::Builder               builder(TestInfo());
		std::set<PipelineListTool::HashKey> kept;
		PipelineList::ExportStats           stats;
		builder.AddJournals({found}, {{0, 0, {}}}, {}, PipelineListTool::KeepRuntimeCode(inner, builder, kept), stats);
		CHECK(kept.empty() && builder.Finish().files.empty());
	}
	// verify's rule: a runtime shader must be covered by a --shaders journal.
	{
		PipelineList::Builder               builder(TestInfo());
		std::set<PipelineListTool::HashKey> kept;
		PipelineList::ExportStats           stats;
		builder.AddJournals({found, missing}, entries, {}, PipelineListTool::KeepRuntimeCode(inner, builder, kept), stats);
		const auto file = builder.Finish();
		std::set<PipelineListTool::HashKey> journaled;
		uint64_t                            total = 0;
		CHECK(PipelineListTool::RuntimeShadersNotCovered(file, journaled, &total) == 1 && total == 1); // no journal: 1
		const auto hash = PipelineList::CodeHash(Bytes(missing.code));
		journaled.insert({hash.low, hash.high});
		CHECK(PipelineListTool::RuntimeShadersNotCovered(file, journaled, &total) == 0 && total == 1); // with the journal: 0
		// A merged stage needs both halves covered.
		const auto list = MakeRuntimeList();
		std::set<PipelineListTool::HashKey> some;
		for (const auto& source: list.sources) {
			const auto h = PipelineList::CodeHash(Bytes(source.code));
			some.insert({h.low, h.high});
		}
		CHECK(PipelineListTool::RuntimeShadersNotCovered(list.file, some, &total) == 1 && total == 3);
		const auto back = PipelineList::CodeHash(Bytes(list.sources[2].back_code));
		some.insert({back.low, back.high});
		CHECK(PipelineListTool::RuntimeShadersNotCovered(list.file, some) == 0);
		// A list without runtime shaders: nothing to cover.
		PipelineList::File plain;
		CHECK(PipelineListTool::RuntimeShadersNotCovered(plain, {}, &total) == 0 && total == 0);
	}
	// The pending-is-no-error / gate helpers on an empty runtime set.
	{
		PipelineList::ResolveStats stats;
		CHECK(PipelineListTool::RuntimeLine(stats).empty() && PipelineListTool::MeetsRuntimeRequirement(stats, 100));
	}
}

// drop: programs the recompiler refuses are removed with their permutations and pipelines (harvest compile probe).
void TestDropShaders() {
	auto list = MakeRuntimeList();
	// A pipeline per shader pair so one dropped program takes its pipelines with it.
	PipelineList::Pipeline a, b;
	a.stages = {{0, false, 1}, {1, false, 1}};
	b.stages = {{2, false, 1}};
	list.file.pipelines = {a, b};
	char       hash[64];
	std::snprintf(hash, sizeof(hash), "0x%016llx", static_cast<unsigned long long>(list.file.shaders[1].hash));
	const std::string log = std::string("noise 0x12 short\nSPIR-V validation failed: hash=") + hash + " stage=4 reason=writable FLAT/GLOBAL\nKYTY_RT_STUB: CS shader 0xffffffffffffffff has 1 BVH\n";
	const auto        parsed = PipelineListTool::ParseDropHashes(log);
	CHECK(parsed.size() == 2 && parsed.contains(list.file.shaders[1].hash) && parsed.contains(0xffffffffffffffffull));
	CHECK(parsed.at(list.file.shaders[1].hash).find("SPIR-V validation failed") == 0);
	PipelineListTool::DropStats stats;
	std::string                 tsv;
	const auto                  out = PipelineListTool::DropShaders(list.file, parsed, stats, &tsv);
	CHECK(stats.asked == 2 && stats.shaders == 1 && stats.permutations == 1 && stats.pipelines == 2 && stats.pipelines_dropped == 1);
	CHECK(out.shaders.size() == 2 && out.permutations.size() == 2 && out.pipelines.size() == 1 && out.pipelines[0].stages.size() == 1);
	CHECK(out.shaders[0].hash == list.file.shaders[0].hash && out.shaders[1].hash == list.file.shaders[2].hash);
	CHECK(out.files == list.file.files && PipelineList::RuntimeFileOf(out) == PipelineList::RuntimeFileOf(list.file));
	CHECK(tsv.find(hash) == 0 && tsv.find("\tpixel\truntime\t") != std::string::npos);
	// Nothing named: the list is unchanged. A tsv written by drop is accepted as a hash file again.
	PipelineListTool::DropStats none;
	CHECK(PipelineListTool::DropShaders(list.file, {}, none).shaders.size() == 3 && none.shaders == 0 && none.pipelines_dropped == 0);
	CHECK(PipelineListTool::ParseDropHashes(tsv).contains(list.file.shaders[1].hash));
	auto bare = out; // the synthetic pipelines carry no snapshot words: round-trip the shaders alone
	bare.pipelines.clear();
	PipelineList::File decoded;
	CHECK(PipelineList::Decode(PipelineList::Encode(bare), decoded) && decoded.shaders.size() == 2);
}

} // namespace

// ---- edges of the extraction gates (the harvest's --require-runtime, verify's
// coverage rule, drop's hash file) ----------------

void TestExtractionGateEdges() {
  using PipelineListTool::MeetsRuntimeRequirement;
  using PipelineListTool::ParseRequirePercent;
  // --require-runtime value: digits only, 0..100; a failed parse never touches
  // the output (a typo must not pass a gate).
  {
    int out = -7;
    for (const char *bad : {"", " 5", "5 ", "-1", "+5", "5%", "1e2", "101",
                            "1000", "0x10", "five", "\xd9\xa3"})
      CHECK(!ParseRequirePercent(bad, out) && out == -7);
    for (const auto &[text, want] :
         std::vector<std::pair<const char *, int>>{{"0", 0},
                                                   {"7", 7},
                                                   {"007", 7},
                                                   {"99", 99},
                                                   {"100", 100},
                                                   {"000", 0}}) {
      out = -1;
      CHECK(ParseRequirePercent(text, out) && out == want);
    }
  }
  // The gate: shares round against the percentage exactly (no float), a list
  // without runtime shaders passes, nothing resolved fails any > 0.
  {
    PipelineList::ResolveStats st;
    st.runtime_shaders = 0;
    for (uint32_t pct : {0u, 1u, 100u})
      CHECK(MeetsRuntimeRequirement(st, pct));
    st.runtime_shaders = 3;
    st.runtime_resolved = 0;
    st.runtime_pending = 3;
    CHECK(MeetsRuntimeRequirement(st, 0) && !MeetsRuntimeRequirement(st, 1));
    st.runtime_resolved = 2;
    st.runtime_pending = 1;
    CHECK(MeetsRuntimeRequirement(st, 66) && !MeetsRuntimeRequirement(st, 67) &&
          !MeetsRuntimeRequirement(st, 100));
    st.runtime_resolved = 3;
    st.runtime_pending = 0;
    CHECK(MeetsRuntimeRequirement(st, 100));
    st.runtime_shaders = 100000;
    st.runtime_resolved =
        99999; // large lists: no overflow, 99.999 % is not 100 %
    CHECK(MeetsRuntimeRequirement(st, 99) && !MeetsRuntimeRequirement(st, 100));
    st.runtime_shaders = 1000000;
    st.runtime_resolved = 1000000;
    CHECK(MeetsRuntimeRequirement(st, 100));
    CHECK(PipelineListTool::RuntimeLine(st) ==
          "runtime: 1000000 resolved, 0 pending");
  }
  // drop's hash file: any 0x + 16 hex digits, either case; shorter / non-hex
  // runs are not hashes; the first line naming a hash is its reason.
  {
    const std::string text =
        "x 0xABCDEF0123456789 first\n0xabcdef0123456789 "
        "again\n0x123456789abcde short\n0xZZZZZZZZZZZZZZZZ nothex\n"
        "two 0x0000000000000001,0x0000000000000002\ntab\there "
        "0x00000000000000ff\rcr\n0x";
    const auto got = PipelineListTool::ParseDropHashes(text);
    CHECK(got.size() == 4 && got.contains(0xABCDEF0123456789ull) &&
          got.contains(1) && got.contains(2) && got.contains(0xff));
    CHECK(got.at(0xABCDEF0123456789ull).find("first") != std::string::npos);
    CHECK(got.at(0xff).find('\t') == std::string::npos &&
          got.at(0xff).find('\r') == std::string::npos);
    CHECK(PipelineListTool::ParseDropHashes("").empty() &&
          PipelineListTool::ParseDropHashes("no hashes here\n").empty());
    const std::string longline = "0x00000000000000aa " + std::string(400, 'z');
    CHECK(PipelineListTool::ParseDropHashes(longline).at(0xaa).size() == 160);
  }
  // drop: nothing named, an unknown hash, everything named.
  {
    auto list = MakeRuntimeList();
    PipelineList::Pipeline a;
    a.stages = {{0, false, 1}, {1, false, 1}};
    list.file.pipelines = {a};
    PipelineListTool::DropStats stats;
    auto none = PipelineListTool::DropShaders(list.file, {}, stats);
    CHECK(none.shaders.size() == list.file.shaders.size() &&
          stats.shaders == 0 && stats.pipelines_dropped == 0);
    std::map<uint64_t, std::string> unknown = {{0x1234, "x"}};
    PipelineListTool::DropStats s2;
    auto same = PipelineListTool::DropShaders(list.file, unknown, s2);
    CHECK(same.shaders.size() == list.file.shaders.size() && s2.asked == 1 &&
          s2.shaders == 0);
    std::map<uint64_t, std::string> all;
    for (const auto &shader : list.file.shaders)
      all[shader.hash] = "all";
    PipelineListTool::DropStats s3;
    std::string tsv;
    auto empty = PipelineListTool::DropShaders(list.file, all, s3, &tsv);
    CHECK(empty.shaders.empty() && empty.pipelines.empty() &&
          s3.shaders == list.file.shaders.size() && s3.pipelines_dropped == 1);
    CHECK(std::count(tsv.begin(), tsv.end(), '\n') ==
              static_cast<long>(list.file.shaders.size()) &&
          tsv.find("\truntime\t") != std::string::npos);
  }
  // KeepRuntimeCode: a code found by the locator is never runtime; an empty
  // code is not kept; the same code twice is one entry and one file name.
  {
    PipelineList::Builder builder(TestInfo());
    std::set<PipelineListTool::HashKey> kept;
    const auto fn = PipelineListTool::KeepRuntimeCode(
        [](std::span<const uint32_t>) { return std::nullopt; }, builder, kept);
    CHECK(!fn({}).has_value() && kept.empty());
    const auto code = Code(900, 40);
    const auto a = fn(code), b = fn(code);
    CHECK(a.has_value() && b.has_value() && kept.size() == 1 &&
          a->offset == 0 && a->file == b->file);
    CHECK(fn(Code(901, 40)).has_value() && kept.size() == 2 &&
          builder.Finish().files.size() == 1);
  }
  // Coverage: executable-file shaders never count as runtime shaders, whatever
  // the journals hold.
  {
    PipelineList::File plain;
    plain.shaders.resize(2);
    uint64_t total = 99;
    CHECK(PipelineListTool::RuntimeShadersNotCovered(plain, {}, &total) == 0 &&
          total == 0);
  }
}

// The code store as the extractor meets it: a directory or a missing parent
// where the file should be, an over-long code, a second writer.
void TestExtractionStoreEdges() {
  TempDir dir;
  const auto good = Code(950, 32);
  {
    PipelineCodeStore store;
    CHECK(store.Open(dir.path / "no" / "such" / "dir" / "s.codestore", "T", "V",
                     1)); // missing parents are created
    CHECK(std::filesystem::is_directory(dir.path / "no" / "such" / "dir"));
  }
  const auto path = dir.path / "s.codestore";
  {
    PipelineCodeStore store;
    CHECK(store.Open(path, "T", "V", 1));
    CHECK(!store.Add(Hash128{1, 1}, 0, std::span<const uint8_t>())); // empty
    const std::vector<uint8_t> huge(PipelineCodeStore::MaxCodeBytes + 4, 1);
    CHECK(!store.Add(Hash128{2, 2}, 0, huge) &&
          store.Count() ==
              0); // over the per-code limit: refused, nothing written
    CHECK(store.FileBytes() == PipelineCodeStore::HeaderBytes);
    CHECK(AddCode(store, good) && store.Count() == 1);
    // a store of another decoder fingerprint is set aside and read-only open
    // never renames it
  }
  {
    PipelineCodeStore ro;
    CHECK(ro.OpenReadOnly(path) && ro.Count() == 1 && !ro.DiscardedStale() &&
          ro.DecoderFingerprint() == 1);
    CHECK(!std::filesystem::exists(path.string() + ".stale"));
  }
}

int main() {
	TestRuntimeListFormat();
	TestResolveRuntime();
	TestMostlyMatches();
	TestNoRuntimeUnchanged();
	TestMixedList();
	TestStoreBasics();
	TestStoreDamage();
	TestStoreStaleHeader();
	TestStoreCapAndReadOnly();
	TestStoreThreads();
	TestStoreSearchMarks();
	TestRuntimeShadersCountMatchingLayoutOnly();
	TestStoreWithResolve();
	TestRetargetRuntime();
	TestMixedMergedCount();
	TestGroupSplit();
	TestToolRuntimeExport();
	TestDropShaders();
        TestExtractionGateEdges();
        TestExtractionStoreEdges();
        if (g_failed != 0) {
		std::printf("PipelineUe5WarmTests: %d check(s) failed\n", g_failed);
		return 1;
	}
	std::printf("PipelineUe5WarmTests: all passed\n");
	return 0;
}
