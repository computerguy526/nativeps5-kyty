// KYTY_INPUT_PROFILE unit tests (standalone, no GPU):
//   clang++ -std=c++20 -O1 [-fsanitize=address,undefined] -I src tests/InputProfileTests.cpp src/common/inputProfile.cpp -o <out>/ipt
//   <out>/ipt <scratch dir>     (writes <scratch dir>/ipt.jsonl; check it with python3 -c 'import json;[json.loads(l) for l in open(...)]')
#include "common/inputProfile.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int g_checks = 0, g_fail = 0;
#define CHECK(c)                                                                 \
	do {                                                                         \
		g_checks++;                                                              \
		if (!(c)) {                                                              \
			g_fail++;                                                            \
			std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c);     \
		}                                                                        \
	} while (0)

using namespace InputProfile;

static uint32_t Bits(float f) {
	uint32_t v;
	std::memcpy(&v, &f, 4);
	return v;
}

static size_t Count(const std::string& s, const std::string& sub) {
	size_t n = 0;
	for (size_t p = s.find(sub); p != std::string::npos; p = s.find(sub, p + 1)) {
		n++;
	}
	return n;
}

static void TestClassify() {
	CHECK(Classify(0) == kZero);
	CHECK(Classify(0x80000000u) == kNegZero);
	CHECK(Classify(7) == kSmallInt);
	CHECK(Classify(255) == kSmallInt);
	CHECK(Classify(256) == kInt16);
	CHECK(Classify(65535) == kInt16);
	CHECK(Classify(0xffffffffu) == kNegInt);
	CHECK(Classify(Bits(1.0f)) == kFloatPos);
	CHECK(Classify(Bits(-0.5f)) == kFloatNeg);
	CHECK(Classify(Bits(1e30f)) == kFloatBig);
	CHECK(Classify(Bits(INFINITY)) == kInf);
	CHECK(Classify(Bits(-INFINITY)) == kInf);
	CHECK(Classify(0x7fc00000u) == kNan);
	CHECK(Classify(0x00010001u) == kDenorm); // > 65535 and exponent 0
	CHECK(Classify(0x12345678u) == kFloatBig);  // bitmask-like dword lands in big float/other
}

static void TestDwordStat() {
	DwordStat d;
	d.Add(Bits(0.25f));
	d.Add(Bits(-2.0f));
	d.Add(Bits(-2.0f));
	d.Add(3);
	CHECK(d.n == 4);
	CHECK(d.flags == (kFloatPos | kFloatNeg | kSmallInt));
	CHECK(d.fmin == -2.0f && d.fmax == 0.25f);
	CHECK(d.umin == 3 && d.umax == Bits(-2.0f));
	CHECK(d.changes == 2);
	DwordStat c;
	c.Add(5);
	c.Add(5);
	CHECK(c.changes == 0 && c.umin == 5 && c.umax == 5 && c.flags == kSmallInt);
}

static Config TestCfg() {
	Config c;
	c.first       = 2;
	c.max_samples = 4;
	c.interval_ms = 100;
	c.raw         = 2;
	c.raw_budget  = 1 << 20;
	c.max_slots   = 8;
	return c;
}

static void TestGate() {
	Registry r(TestCfg());
	CHECK(r.BeginSample(1, 2, 0, 1000));
	CHECK(r.BeginSample(1, 2, 0, 1000));
	CHECK(!r.BeginSample(1, 2, 0, 1050)); // past first, inside the interval
	CHECK(r.BeginSample(1, 2, 0, 1100));
	CHECK(!r.BeginSample(1, 2, 0, 1150));
	CHECK(r.BeginSample(1, 2, 0, 1300));
	CHECK(!r.BeginSample(1, 2, 0, 5000)); // max 4 reached
	CHECK(r.BeginSample(1, 2, 1, 5000));  // another gate of the same shader
	CHECK(r.BeginSample(9, 2, 0, 5000));  // another shader
	const std::string s = r.Serialize();
	CHECK(s.find("\"k\":\"P\",\"gate\":0,\"draws\":7,\"n\":4") != std::string::npos);
}

static void TestData() {
	Registry r(TestCfg());
	uint32_t a[4] = {3, Bits(1.0f), 0, 0xdeadbeefu};
	uint32_t b[4] = {5, Bits(2.0f), 0, 0xdeadbeefu};
	uint32_t meta[2] = {11, 22};
	r.AddData(0xabc, 2, 'B', 1, a, sizeof(a), 4096, meta, 2, 1);
	r.AddData(0xabc, 2, 'B', 1, b, sizeof(b), 8192, meta, 2, 2);
	r.AddData(0xabc, 2, 'B', 1, a, sizeof(a), 4096, meta, 2, 0); // same bytes as a: no 3rd raw
	r.AddData(0xabc, 2, 'B', 1, nullptr, 0, 4096, meta, 2, 32);   // unreadable sample
	const std::string s = r.Serialize();
	CHECK(s.find("\"h\":\"0000000000000abc\",\"st\":2,\"k\":\"B\",\"s\":1,\"n\":4,\"unread\":1,\"sz\":8192,\"info\":35") !=
	      std::string::npos);
	// position 0: small ints 3..5, 2 changes (3->5->3), n 3
	CHECK(s.find("[2,3,5,") != std::string::npos);
	CHECK(Count(s, "\"raw\":[\"") == 1);
	CHECK(r.RawBytes() == 32);
	CHECK(s.find("\"meta\":[[11,22]]") != std::string::npos);
	// strided (vertex) data: 2 vertices of stride 8 fold onto 2 positions
	uint32_t v[4] = {Bits(-1.0f), 7, Bits(1.0f), 9};
	r.AddData(0xdef, 1, 'V', 0, v, sizeof(v), 16, nullptr, 0, 0, 8);
	const std::string s2 = r.Serialize();
	const auto        p  = s2.find("\"k\":\"V\"");
	CHECK(p != std::string::npos);
	CHECK(s2.find("\"stride\":8", p > 80 ? p - 80 : 0) != std::string::npos);
	CHECK(s2.find(std::string("[") + std::to_string(kFloatPos | kFloatNeg) + ",", p) != std::string::npos);
	CHECK(s2.find(std::string("[") + std::to_string(kSmallInt) + ",7,9,", p) != std::string::npos);
}

static void TestDescriptorsAndLimits() {
	Config c = TestCfg();
	c.raw_budget = 16;
	Registry r(c);
	uint32_t t1[8] = {1, 2, 3, 4, 5, 6, 7, 8}, t2[8] = {1, 2, 3, 4, 5, 6, 7, 9};
	for (int i = 0; i < 3; i++) {
		r.AddDescriptor(5, 2, 'T', 0, t1, 8);
	}
	r.AddDescriptor(5, 2, 'T', 0, t2, 8);
	for (uint32_t k = 0; k < 6; k++) {
		uint32_t t[8] = {k};
		r.AddDescriptor(5, 2, 'T', 1, t, 8);
	}
	std::string s = r.Serialize();
	CHECK(s.find("\"k\":\"T\",\"s\":0,\"n\":4") != std::string::npos);
	CHECK(s.find("[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,9]]") != std::string::npos);
	CHECK(Count(s.substr(s.find("\"k\":\"T\",\"s\":1")), "],[") == 3); // capped at 4 distinct
	// raw budget 16 bytes: a 32-byte slot gets stats but no raw snapshot
	uint32_t big[8] = {};
	r.AddData(6, 2, 'B', 0, big, sizeof(big), 32, nullptr, 0, 0);
	CHECK(r.RawBytes() == 0);
	// slot cap 8
	for (uint32_t k = 0; k < 20; k++) {
		r.AddDescriptor(7, 2, 'S', k, t1, 4);
	}
	CHECK(r.SlotCount() == 8);
	s = r.Serialize();
	CHECK(s.find("\"dropped\":") != std::string::npos && s.find("\"dropped\":0") == std::string::npos);
}

static void TestBlocksAndChunks() {
	Config c     = TestCfg();
	c.stat_bytes = 16;   // 4 positions
	c.read_bytes = 8192; // the rest in 64-dword blocks
	c.raw_bytes  = 64;
	Registry              r(c);
	std::vector<uint32_t> d(4096, 0); // 16 KiB: only 8 KiB is read
	for (uint32_t i = 0; i < 4; i++) {
		d[i] = Bits(0.5f * static_cast<float>(i + 1));
	}
	for (uint32_t i = 4; i < 68; i++) {
		d[i] = i % 9 + 1; // block 0: small ints 1..9 (loop-count-like)
	}
	for (uint32_t i = 1024; i < 2048; i++) {
		d[i] = 0x7fc00000u; // chunk 1: NaNs, marked GPU-modified below
	}
	r.AddData(1, 2, 'B', 0, d.data(), d.size() * 4, d.size() * 4, nullptr, 0, 0, 0, 0x1u);
	std::string s = r.Serialize();
	const auto  p = s.find("\"k\":\"B\"");
	CHECK(p != std::string::npos);
	CHECK(s.find("\"blk\":[[" + std::to_string(kSmallInt) + ",1,9,", p) != std::string::npos);
	CHECK(s.find(std::to_string(kNan) + ",", s.find("\"blk\"", p)) == std::string::npos); // chunk 1 skipped
	// blocks: (1024 - 4) dwords of chunk 0 -> 16 blocks; chunk 1 never seen -> trimmed
	const auto b0 = s.find("\"blk\":[", p), b1 = s.find("],\"raw\"", b0);
	CHECK(Count(s.substr(b0, b1 - b0), "],[") == 15);
	CHECK(r.RawBytes() == 64);
	// a raw window that crosses an unreadable chunk is not snapshotted
	Config c2     = c;
	c2.raw_bytes  = 8192;
	Registry r2(c2);
	r2.AddData(1, 2, 'B', 0, d.data(), 8192, 8192, nullptr, 0, 0, 0, 0x1u);
	CHECK(r2.RawBytes() == 0);
	r2.AddData(1, 2, 'B', 0, d.data(), 8192, 8192, nullptr, 0, 0, 0, 0x3u);
	CHECK(r2.RawBytes() == 8192);
}

static void TestBase64() {
	CHECK(Base64("", 0).empty());
	CHECK(Base64("f", 1) == "Zg==");
	CHECK(Base64("fo", 2) == "Zm8=");
	CHECK(Base64("foo", 3) == "Zm9v");
	CHECK(Base64("foobar", 6) == "Zm9vYmFy");
}

static void TestThreadsAndFile(const std::string& dir) {
	Config c = TestCfg();
	c.max_slots = 1000;
	c.max_samples = 1000000;
	c.first       = 1000000;
	Registry r(c);
	std::vector<std::thread> ts;
	for (int t = 0; t < 4; t++) {
		ts.emplace_back([&r, t] {
			std::vector<uint32_t> d(1024);
			for (int i = 0; i < 2000; i++) {
				d[i % 1024] = static_cast<uint32_t>(i * (t + 1));
				if (r.BeginSample(100 + (i % 7), 1, 0, i)) {
					r.AddData(100 + (i % 7), 1, 'B', i % 3, d.data(), d.size() * 4, 4096, nullptr, 0, 0);
				}
				if (i % 50 == 0) {
					(void)r.Serialize();
				}
			}
		});
	}
	for (auto& t: ts) {
		t.join();
	}
	CHECK(r.SlotCount() == 21);
	const std::string path = dir + "/ipt.jsonl";
	CHECK(r.WriteFile(path));
	FILE* f = std::fopen(path.c_str(), "rb");
	CHECK(f != nullptr);
	if (f != nullptr) {
		std::fseek(f, 0, SEEK_END);
		const long n = std::ftell(f);
		std::fclose(f);
		CHECK(n > 1000);
	}
	const std::string s = r.Serialize();
	CHECK(Count(s, "\n") == 1 + 7 + 21); // header + 7 gates + 21 slots
	CHECK(Count(s, "\"n\":") == 7 + 21);
}

static void TestOffByDefault() {
	// The test binary runs without KYTY_INPUT_PROFILE: the capture is off.
	if (std::getenv("KYTY_INPUT_PROFILE") == nullptr) {
		CHECK(!On());
		CHECK(!Config::FromEnv().on);
	}
}

int main(int argc, char** argv) {
	const std::string dir = argc > 1 ? argv[1] : ".";
	TestOffByDefault();
	TestClassify();
	TestDwordStat();
	TestGate();
	TestData();
	TestDescriptorsAndLimits();
	TestBlocksAndChunks();
	TestBase64();
	TestThreadsAndFile(dir);
	{
		// Cost of one sampled 64 KiB buffer (statistics + raw dedup compare), for the capture budget.
		Registry              r(Config {});
		std::vector<uint32_t> d(16384);
		for (uint32_t i = 0; i < d.size(); i++) {
			d[i] = i * 2654435761u;
		}
		const auto t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < 200; i++) {
			d[0] = static_cast<uint32_t>(i);
			r.AddData(1, 2, 'B', 0, d.data(), d.size() * 4, d.size() * 4, nullptr, 0, 0);
		}
		const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
		std::printf("bench: %.1f us per sampled 64 KiB buffer\n", us);
	}
	std::printf("InputProfileTests: %d checks, %d failed\n", g_checks, g_fail);
	return g_fail == 0 ? 0 : 1;
}
