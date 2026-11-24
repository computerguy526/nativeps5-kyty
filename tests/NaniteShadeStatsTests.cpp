// naniteshade: unit test of the Nanite shading-bin counters (naniteShadeStats.h). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/NaniteShadeStatsTests.cpp -o /tmp/nst && /tmp/nst
// CMake target: nanite_shade_stats_tests.
#include "graphics/guest_gpu/command_processor/naniteShadeStats.h"

#include <cstdio>
#include <string>

using namespace Libs::Graphics::NaniteShadeStats;

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                                                   \
	do {                                                                                           \
		g_checks++;                                                                                \
		if (!(c)) {                                                                                \
			g_failed++;                                                                            \
			std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, #c);                              \
		}                                                                                          \
	} while (0)

int main() {
	// Env parse: off by default, "1" = 5 s, "10" = 10 s.
	CHECK(ParseIntervalMs(nullptr) == 0);
	CHECK(ParseIntervalMs("") == 0);
	CHECK(ParseIntervalMs("0") == 0);
	CHECK(ParseIntervalMs("1") == 5000);
	CHECK(ParseIntervalMs("10") == 10000);
	CHECK(ParseIntervalMs("2.5") == 2500);

	State st;
	// Before any compaction nothing is a bin.
	CHECK(!st.IsBinArgs(0x1ec2000000ull));
	st.NoteIndirect(0x1ec2000010ull, Outcome::Executed, 0x1111);
	// SAROS frame: 51 records, args pool 0x1ec2000000, stride 16, 11 kept.
	st.NoteCompaction(0x1ec2000000ull, 51u * 16u, 51, 11);
	CHECK(st.IsBinArgs(0x1ec2000000ull));
	CHECK(st.IsBinArgs(0x1ec2000000ull + 50u * 16u));
	CHECK(!st.IsBinArgs(0x1ec2000000ull + 51u * 16u));
	CHECK(!st.IsBinArgs(0x1ec1ffffffull));
	// Same base again with a larger extent widens the range, does not take a new slot.
	st.NoteCompaction(0x1ec2000000ull, 54u * 16u, 54, 12);
	CHECK(st.IsBinArgs(0x1ec2000000ull + 53u * 16u));
	CHECK(st.range_next.load() == 1);
	// Bin outcomes, distinct bin shaders, others.
	for (uint32_t i = 0; i < 11; i++) {
		st.NoteIndirect(0x1ec2000000ull + i * 16u, Outcome::Executed, 0x100 + (i % 7));
	}
	st.NoteIndirect(0x1ec2000000ull + 20u * 16u, Outcome::NotReady, 0);
	st.NoteIndirect(0x1ec2000000ull + 21u * 16u, Outcome::NoProgram, 0);
	st.NoteIndirect(0x1a40af0000ull, Outcome::SkipEmpty, 0);
	st.NoteIndirect(0x1a40af0010ull, Outcome::Executed, 0x999);
	auto s = st.Take();
	CHECK(s.compactions == 2 && s.records_in == 105 && s.records_kept == 23);
	CHECK(s.bin[0] == 11 && s.bin[3] == 1 && s.bin[2] == 1 && s.bin[1] == 0);
	CHECK(s.bin_shaders == 7);
	CHECK(s.other[0] == 2 && s.other[1] == 1); // the pre-compaction one counted as other
	const std::string line = Format(s, 5.0);
	CHECK(line.find("[naniteshade] 5.0s: compact 2 (in 105 kept 23 = 11.5/c)") == 0);
	CHECK(line.find("bin exec 11 empty 0 noprog 1 notready 1 nocs 0 (5.5/c, 7 cs)") != std::string::npos);
	CHECK(line.find("other exec 2 empty 1") != std::string::npos);
	// Take resets the counts, keeps the ranges.
	auto z = st.Take();
	CHECK(z.compactions == 0 && z.bin[0] == 0 && z.bin_shaders == 0 && z.other[0] == 0);
	CHECK(st.IsBinArgs(0x1ec2000000ull));
	CHECK(Format(z, 5.0).find("= 0.0/c") != std::string::npos);
	// Ring: 9 distinct bases evict the oldest.
	for (uint32_t i = 1; i <= kRanges; i++) {
		st.NoteCompaction(0x1ec2000000ull + i * 0x10000ull, 0x400, 64, 1);
	}
	CHECK(!st.IsBinArgs(0x1ec2000000ull));
	CHECK(st.IsBinArgs(0x1ec2000000ull + kRanges * 0x10000ull));
	// Interval: first call arms, then once per interval with the real elapsed time.
	int64_t e = 0;
	CHECK(!st.Due(1000, 5000, &e));
	CHECK(!st.Due(5999, 5000, &e));
	CHECK(st.Due(6100, 5000, &e) && e == 5100);
	CHECK(!st.Due(6200, 5000, &e));

	std::printf("nanite_shade_stats_tests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
