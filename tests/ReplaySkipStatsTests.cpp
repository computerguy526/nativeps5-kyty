#include "graphics/host_gpu/renderer/pipeline/replaySkipStats.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Libs::Graphics;

static void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		std::exit(1);
	}
}

int main() {
	const std::vector<uint32_t> a {1, 2, 3, 4}, b {1, 2, 9, 4}, c {1, 2, 3}, d {1, 2, 3, 4};
	Check(ReplayFirstKeyDiff(a, b) == 2, "diff in the middle");
	Check(ReplayFirstKeyDiff(a, c) == 3, "prefix: shorter length");
	Check(ReplayFirstKeyDiff(a, d) == 4, "equal: size");
	Check(ReplayFirstKeyDiff({}, a) == 0, "empty vs non-empty");

	const auto text = ReplaySkipStats::DescribeStaticKey("ps", a, b);
	Check(text.find("stored 4 built 4") != std::string::npos && text.find("@2 stored 0x3 built 0x9") != std::string::npos,
	      "describe names the first differing word");
	const auto prefix = ReplaySkipStats::DescribeStaticKey("vs", a, c);
	Check(prefix.find("@3 stored 0x4 built -0") != std::string::npos, "describe: built shorter");

	ReplaySkipStats stats;
	Check(stats.Total() == 0 && stats.Summary().empty(), "empty stats");
	for (uint64_t i = 0; i < 20; ++i) stats.Note(ReplaySkipReason::KeyStatic, 0xabc00 + i, "d");
	stats.Note(ReplaySkipReason::InputSize, 0x1);
	stats.Note(ReplaySkipReason::NotPublished, 0x2);
	stats.Note(ReplaySkipReason::NotPublished, 0x3);
	Check(stats.Count(ReplaySkipReason::KeyStatic) == 20, "key-static counted");
	Check(stats.Count(ReplaySkipReason::NotPublished) == 2, "not-published counted");
	Check(stats.Count(ReplaySkipReason::SkipDispatch) == 0, "unused reason is zero");
	Check(stats.Total() == 23, "total");
	const auto summary = stats.Summary();
	Check(summary.find("key-static 20 [") != std::string::npos, "summary names key-static");
	Check(summary.find("00000000000abc08 d") == std::string::npos && summary.find("00000000000abc00 d") != std::string::npos,
	      "only the first 8 samples are kept");
	Check(summary.find("skip-dispatch") == std::string::npos, "zero reasons omitted");
	Check(summary.find("input-size 1 [0000000000000001]") != std::string::npos, "sample without detail");
	std::printf("replay skip stats tests passed\n");
	return 0;
}
