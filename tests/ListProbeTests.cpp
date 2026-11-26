// ListProbe (tools/pipeline-list/listProbe.h): each program is translated in a child process; a fatal exit, an abort, a signal or a
// hang kills the child only. Synthetic data: a fake translator that fails for chosen hashes.
#include "listProbe.h"

#include <cstdio>
#include <unistd.h>

static int g_failed = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); g_failed++; } } while (0)

using namespace ListProbe;

namespace {

constexpr uint64_t kGood = 0x1111111111111111ull, kExit = 0x2222222222222222ull, kAbort = 0x3333333333333333ull,
                   kHang = 0x4444444444444444ull, kSegv = 0x5555555555555555ull, kRt = 0x6666666666666666ull;

int Fake(const Unit& u) {
	switch (u.hash) {
		case kExit:
			std::fprintf(stderr, "noise\nSPIR-V validation failed: hash=0x%016llx stage=4 reason=writable FLAT/GLOBAL addresses\nmore\n",
			             static_cast<unsigned long long>(u.hash));
			_exit(65);
		case kAbort:
			std::fprintf(stderr, "Assertion failed: source.indirect_root >= images.size() (ResourceMaterialization.cpp:2661)\n");
			std::abort();
		case kHang:
			for (;;) pause();
		case kSegv: {
			volatile int* p = nullptr;
			*p = 1;
			return 0;
		}
		default:
			// permutation 1 of the good program is the one that fails: the program is refused when any permutation is
			if (u.hash == kRt && u.perm == 1) { std::fprintf(stderr, "EXIT bad perm\n"); return 3; }
			return 0;
	}
}

void TestRun() {
	std::vector<Unit> units = {{kGood, 2, 0}, {kExit, 2, 0}, {kAbort, 2, 0}, {kSegv, 0, 0}, {kHang, 1, 0}, {kGood, 2, 1}, {kRt, 2, 0}, {kRt, 2, 1}};
	Options o;
	o.jobs = 3;
	o.timeout_ms = 700;
	const auto out = RunUnits(units, Fake, o);
	CHECK(out.size() == units.size());
	CHECK(out[0].verdict == Verdict::Ok && out[5].verdict == Verdict::Ok && out[6].verdict == Verdict::Ok);
	CHECK(out[1].verdict == Verdict::Refused && out[1].code == 65 && out[1].reason.find("SPIR-V validation failed") == 0);
	CHECK(out[1].reason.find("0x") == std::string::npos || out[1].reason.find("hash=") != std::string::npos);
	CHECK(out[2].verdict == Verdict::Crashed && out[2].reason.find("ResourceMaterialization.cpp:2661") != std::string::npos);
	CHECK(out[3].verdict == Verdict::Crashed);
	CHECK(out[4].verdict == Verdict::TimedOut && out[4].reason.find("timed out") == 0);
	CHECK(out[7].verdict == Verdict::Refused && out[7].reason == "EXIT bad perm");
	const auto rows = Refused(units, out);
	CHECK(rows.size() == 5 && !rows.contains(kGood) && rows.contains(kExit) && rows.contains(kAbort) && rows.contains(kHang) && rows.contains(kSegv) && rows.contains(kRt));
	CHECK(rows.at(kExit).rfind("refused: ", 0) == 0 && rows.at(kAbort).rfind("crashed: ", 0) == 0 && rows.at(kHang).rfind("timeout: ", 0) == 0);
	// The rows are a hash file: only the first column names a hash (the reason's own hash is stripped of its `0x`).
	const auto text = FormatRows(rows);
	size_t     n = 0;
	for (size_t at = text.find("0x"); at != std::string::npos; at = text.find("0x", at + 2)) n++;
	CHECK(n == 5);
	CHECK(text.find("0x1111111111111111") == std::string::npos && text.find("0x2222222222222222\trefused: ") == 0);
}

void TestEmptyAndCleaning() {
	CHECK(RunUnits({}, Fake, Options {}).empty());
	CHECK(CleanReason("a\tb 0x0123456789abcdef c").find("0x") == std::string::npos);
	CHECK(CleanReason("0x12 short") == "0x12 short");
	CHECK(PickReason("one\ntwo\n") == "two" && PickReason("one\nEXIT bad\nlast\n") == "EXIT bad" && PickReason("").empty());
}

void TestRt() {
	std::map<uint64_t, std::string> rows = {{kExit, "refused: x"}};
	AddRayTracing(rows, {kExit, kRt}, "ray tracing (BVH), the profile turns ray tracing off");
	CHECK(rows.size() == 2 && rows.at(kExit) == "refused: x" && rows.at(kRt).find("ray tracing") == 0);
}

} // namespace

int main() {
	TestRun();
	TestEmptyAndCleaning();
	TestRt();
	if (g_failed != 0) { std::printf("ListProbeTests: %d check(s) failed\n", g_failed); return 1; }
	std::printf("ListProbeTests: all passed\n");
	return 0;
}
