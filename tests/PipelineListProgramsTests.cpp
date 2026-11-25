// kyty_pipeline_list export --programs (tools/pipeline-list/programsSource.h): the persistent program cache as a source of list shaders and
// permutations. Compute input infos are rebuilt from the static key (inverse of BuildStageStaticKey); graphics sources take their input info from
// a journal source of the same identity and are left out without one; specializations are brought to this build's layout; variant keys and
// orphan permutations are counted, not listed.
#include "programsSource.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace Libs::Graphics;
using namespace PipelineListTool;

static int g_failed = 0;
#define CHECK(expr)                                                                          \
	do {                                                                                     \
		if (!(expr)) {                                                                       \
			std::printf("PipelineListProgramsTests: FAILED: %s (line %d)\n", #expr, __LINE__); \
			++g_failed;                                                                      \
		}                                                                                    \
	} while (0)

namespace {

template <class T>
void Put(std::vector<uint8_t>& o, T v) {
	const auto* b = reinterpret_cast<const uint8_t*>(&v);
	o.insert(o.end(), b, b + sizeof(T));
}

// BuildStageStaticKey(ShaderComputeInputInfo) of a 128x1x1, wave 64, lds 4096 dwords in the device buffer, async compute shader.
std::vector<uint32_t> ComputeKey() {
	return {5, 64u | (0xc0u << 8u), 32, 1, 4096u | 0x80000000u, 8, 2u, 128, 1, 1, 0, 1, 1, 1};
}

std::vector<uint8_t> SourceKey(uint32_t stage, uint64_t hash, const std::vector<uint32_t>& statics, const std::vector<uint32_t>& code, uint8_t plain,
                               bool variant, uint64_t var) {
	std::vector<uint8_t> k;
	Put<uint32_t>(k, 1);
	Put<uint32_t>(k, stage);
	Put<uint64_t>(k, hash);
	Put<uint32_t>(k, 16);
	Put<uint32_t>(k, static_cast<uint32_t>(code.size()));
	Put<uint32_t>(k, static_cast<uint32_t>(statics.size()));
	for (auto w: statics) Put<uint32_t>(k, w);
	Put<uint32_t>(k, 64);
	Put<uint32_t>(k, 0);
	Put<uint8_t>(k, plain);
	if (variant) Put<uint64_t>(k, var);
	Put<uint32_t>(k, static_cast<uint32_t>(code.size()));
	for (auto w: code) Put<uint32_t>(k, w);
	Put<uint32_t>(k, 0);
	return k;
}

void Record(std::vector<uint8_t>& f, uint32_t kind, const std::vector<uint8_t>& payload) {
	Put<uint32_t>(f, Kprc::kMagic);
	Put<uint32_t>(f, kind);
	Put<uint64_t>(f, payload.size());
	Put<uint64_t>(f, 0);
	f.insert(f.end(), payload.begin(), payload.end());
}

void AddSource(std::vector<uint8_t>& f, uint8_t digest, const std::vector<uint8_t>& key) {
	std::vector<uint8_t> p(16, digest);
	Put<uint32_t>(p, static_cast<uint32_t>(key.size()));
	p.insert(p.end(), key.begin(), key.end());
	Put<uint8_t>(p, 0);
	Put<uint32_t>(p, 0);
	Record(f, 1, p);
}

void AddPerm(std::vector<uint8_t>& f, uint8_t digest, uint32_t cursor, const std::vector<uint8_t>& spec) {
	std::vector<uint8_t> p(16, digest);
	Put<uint32_t>(p, cursor);
	Put<uint32_t>(p, static_cast<uint32_t>(spec.size()));
	p.insert(p.end(), spec.begin(), spec.end());
	for (int i = 0; i < 3; i++) Put<uint32_t>(p, 0);
	Record(f, 2, p);
}

std::vector<uint8_t> Header() {
	std::vector<uint8_t> f(16, 0);
	Put<uint64_t>(f, 3);
	f.insert(f.end(), {'h', 'd', 'r'});
	Put<uint64_t>(f, 0);
	return f;
}

std::vector<uint8_t> EmptySpec() {
	std::vector<uint8_t> s;
	Put<uint32_t>(s, 0);
	Put<uint32_t>(s, 0);
	return s;
}

// One image record of the int21 release: 8 words + cube + fmask.
std::vector<uint8_t> Int21Spec() {
	std::vector<uint8_t> s;
	Put<uint32_t>(s, 0);
	Put<uint32_t>(s, 1);
	for (int i = 0; i < 8; i++) Put<uint32_t>(s, i);
	Put<uint8_t>(s, 0);
	Put<uint8_t>(s, 1);
	return s;
}

void TestComputeInfo() {
	const auto key = ComputeKey();
	ShaderComputeInputInfo info;
	CHECK(ComputeInfoFromStaticKey(key, info));
	CHECK(info.workgroup_register == 5 && info.wave_size == 64 && info.float_mode == 0xc0 && info.host_subgroup_size == 32);
	CHECK(info.thread_ids_num == 1 && info.lds_size_dwords == 4096 && info.lds_storage && info.scratch_size_dwords == 8);
	CHECK(!info.dispatch_thread_dimensions && info.async_compute && !info.dispatch_dimensions_indirect);
	CHECK(info.threads_num[0] == 128 && info.threads_num[1] == 1 && info.threads_num[2] == 1);
	CHECK(info.group_id[0] && !info.group_id[1] && info.group_id[2] && info.tg_size_en);
	const auto bytes = StoredComputeInfo(key);
	CHECK(bytes.size() == 96);
	// Layout the journals store (offsets verified against 1,623 recorded compute sources): threads_num@0, lds@12, scratch@16, host@20, wave@24, float@28,
	// group_id@56, dispatch_thread_dimensions@59, lds_storage@60, thread_ids_num@64, workgroup_register@68, tg_size_en@72, async@73.
	uint32_t w = 0;
	std::memcpy(&w, bytes.data() + 0, 4);
	CHECK(w == 128);
	std::memcpy(&w, bytes.data() + 12, 4);
	CHECK(w == 4096);
	std::memcpy(&w, bytes.data() + 24, 4);
	CHECK(w == 64);
	CHECK(bytes[28] == 0xc0 && bytes[56] == 1 && bytes[58] == 1 && bytes[60] == 1 && bytes[72] == 1 && bytes[73] == 1 && bytes[74] == 0);
	CHECK(bytes[80] == 0 && bytes[88] == 0); // runtime pointers cleared
	{ const std::vector<uint32_t> bad {1, 2, 3}; CHECK(StoredComputeInfo(bad).empty()); } // not a compute key
}

void TestLoadPrograms() {
	auto             f       = Header();
	const auto       cs_key  = ComputeKey();
	AddSource(f, 1, SourceKey(4, 0xaa, cs_key, {1, 2, 3}, 0, false, 0));                  // compute: info from the key
	AddSource(f, 2, SourceKey(1, 0xbb, {7, 8}, {4, 5}, 0, false, 0));                     // vertex, no journal knows it: left out
	AddSource(f, 3, SourceKey(2, 0xcc, {9}, {6, 7}, 1, false, 0));                        // pixel, plain variant byte, a journal knows it
	AddSource(f, 4, SourceKey(4, 0xdd, cs_key, {8}, 0, true, 0x1122334455667788ull));    // SAROS variant: left out
	AddPerm(f, 1, 0, EmptySpec());
	AddPerm(f, 1, 16, Int21Spec());                                                       // int21 specialization: upgraded
	AddPerm(f, 2, 0, EmptySpec());                                                        // its source is left out
	AddPerm(f, 3, 4, EmptySpec());
	AddPerm(f, 9, 0, EmptySpec());                                                        // orphan
	AddPerm(f, 1, 8, std::vector<uint8_t> {0xff, 0xff, 0xff, 0xff});                      // unreadable
	const std::vector<uint8_t> pixel_info(232, 0x5a);
	ShaderJournal::Source      known;
	known.stage = 2; known.hash = 0xcc; known.user_data_count = 16; known.wave_size = 64; known.user_data_base = 0;
	known.plain_mip_stats_variant = true; known.static_state = {9}; known.code = {6, 7};
	std::vector<ShaderJournal::Source> sources;
	std::vector<ShaderJournal::Entry>  entries;
	std::string                        error;
	const auto                         stats = LoadPrograms(
        f, [&](const ShaderJournal::Source& s) { return IdentityKeyOf(s) == IdentityKeyOf(known) ? pixel_info : std::vector<uint8_t> {}; }, sources, entries, &error);
	CHECK(error.empty());
	CHECK(stats.sources == 4 && stats.permutations == 6);
	CHECK(stats.sources_no_info == 1 && stats.sources_with_variant == 1 && stats.permutations_orphan == 1 && stats.permutations_unreadable == 1);
	CHECK(sources.size() == 2 && sources[0].kind == ShaderJournal::Kind::Compute && sources[1].kind == ShaderJournal::Kind::Pixel);
	CHECK(sources[0].input_info.size() == 96 && sources[1].input_info == pixel_info && sources[1].plain_mip_stats_variant);
	CHECK(sources[0].hash == 0xaa && sources[0].code == std::vector<uint32_t>({1, 2, 3}) && sources[0].static_state == cs_key);
	CHECK(entries.size() == 3);
	CHECK(entries[0].source == 0 && entries[0].push_data_cursor == 0 && entries[0].specialization == EmptySpec());
	CHECK(entries[1].source == 0 && entries[1].push_data_cursor == 16);
	size_t at = 0; uint32_t n = 0;
	CHECK(ReadsAsSpecialization(entries[1].specialization, false, &at, &n) && n == 1);   // upgraded to this build's image record
	CHECK(entries[2].source == 1 && entries[2].push_data_cursor == 4);
}

void TestDamagedFile() {
	auto f = Header();
	AddSource(f, 1, SourceKey(4, 0xaa, ComputeKey(), {1, 2, 3}, 0, false, 0));
	AddPerm(f, 1, 0, EmptySpec());
	f.resize(f.size() - 7); // truncated last record: the records before it stay, the damage is reported
	std::vector<ShaderJournal::Source> sources;
	std::vector<ShaderJournal::Entry>  entries;
	std::string                        error;
	const auto stats = LoadPrograms(f, {}, sources, entries, &error);
	CHECK(!error.empty() && stats.sources == 1 && sources.size() == 1 && entries.empty());
}

} // namespace

int main() {
	TestComputeInfo();
	TestLoadPrograms();
	TestDamagedFile();
	if (g_failed != 0) {
		std::printf("PipelineListProgramsTests: %d check(s) failed\n", g_failed);
		return 1;
	}
	std::printf("PipelineListProgramsTests: all passed\n");
	return 0;
}
