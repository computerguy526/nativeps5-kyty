// KYTY_FLIGHTREC unit tests + micro-benchmark (standalone, no GPU):
//   clang++ -std=c++20 -O2 [-fsanitize=address,undefined] -I src tests/FlightRecTests.cpp src/common/flightRec.cpp -o /tmp/frt
//   /tmp/frt <out dir>          (then: python3 work/tools/flightrec/decode.py <out dir>/flightrec_*.bin --summary)
#include "common/flightRec.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
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

static std::vector<std::string> List(const std::string& dir, const char* part) {
	std::vector<std::string> out;
	if (DIR* d = opendir(dir.c_str())) {
		while (dirent* e = readdir(d)) {
			if (std::strstr(e->d_name, part) != nullptr) {
				out.push_back(dir + "/" + e->d_name);
			}
		}
		closedir(d);
	}
	return out;
}

struct Parsed {
	uint32_t rings = 0, flip = 0, version = 0, level = 0;
	char     why[32] {};
	uint64_t records = 0, bad = 0, window_ns = 0;
	std::vector<uint64_t>             heads;
	std::vector<FlightRec::Record>    recs;
};
// Minimal reader (decode.py is the real one): checks the per-ring sequence of a0 counters.
static Parsed Parse(const std::string& path) {
	Parsed p;
	FILE*  f = std::fopen(path.c_str(), "rb");
	if (f == nullptr) {
		return p;
	}
	char     magic[8];
	uint32_t v[10];
	uint64_t t, window;
	uint32_t images, level;
	std::fread(magic, 1, 8, f);
	std::fread(v, 4, 10, f);
	std::fread(&t, 8, 1, f);
	std::fread(p.why, 1, 32, f);
	std::fread(&window, 8, 1, f);
	std::fread(&images, 4, 1, f);
	std::fread(&level, 4, 1, f);
	p.version   = v[0];
	p.rings     = v[2];
	p.flip      = v[5];
	p.window_ns = window;
	p.level     = level;
	for (uint32_t i = 0; i < p.rings; i++) {
		uint32_t a[2];
		uint64_t b[4];
		char     name[32];
		std::fread(a, 4, 2, f);
		std::fread(b, 8, 4, f);
		std::fread(name, 1, 32, f);
		p.heads.push_back(b[2]);
		uint64_t prev = 0;
		for (uint64_t k = 0; k < b[3]; k++) {
			FlightRec::Record r;
			std::fread(&r, sizeof(r), 1, f);
			p.records++;
			p.recs.push_back(r);
			if (r.ev == 1000 && prev != 0 && r.a[0] != prev + 1) {
				p.bad++;
			}
			if (r.ev == 1000) {
				prev = r.a[0];
			}
		}
		uint64_t after;
		std::fread(&after, 8, 1, f);
	}
	std::fclose(f);
	return p;
}

static uint64_t Count(const Parsed& p, uint16_t ev) {
	uint64_t n = 0;
	for (const auto& r: p.recs) {
		n += r.ev == ev ? 1 : 0;
	}
	return n;
}

// Child modes (re-exec'd so each gets a fresh recorder with its own env).
static int ChildMain(const std::string& mode, const std::string& dir) {
	setenv("KYTY_FLIGHTREC_DIR", dir.c_str(), 1);
	setenv("KYTY_FLIGHTREC_TAG", mode.c_str(), 1);
	setenv("KYTY_FLIGHTREC_EXIT", "0", 1);
	if (mode == "window") {
		setenv("KYTY_FLIGHTREC_WINDOW_S", "1", 1);
		for (uint64_t k = 1; k <= 5; k++) {
			FlightRec::Rec(1000, k);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1700));
		for (uint64_t k = 11; k <= 15; k++) {
			FlightRec::Rec(1000, k);
		}
		FlightRec::DumpNow("win", 0);
	} else if (mode == "stall") {
		for (int i = 0; i < 3; i++) {
			FlightRec::NotePresent();
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(2600)); // monitor: present gap > 2 s -> dump
	} else if (mode == "mainhang") {
		FlightRec::Heartbeat();
		std::this_thread::sleep_for(std::chrono::milliseconds(2600)); // monitor: heartbeat gap > 2 s -> dump + backtrace
	} else if (mode == "sigterm") {
		FlightRec::Rec(1000, 7);
		raise(SIGTERM);
		_exit(3); // not reached: the dump handler re-raises with the default action
	} else if (mode == "light") {
		for (int i = 0; i < 1000; i++) {
			FlightRec::Rec(FlightRec::EvDraw, static_cast<uint64_t>(i));
			FlightRec::Rec(FlightRec::EvUpload, 1);
		}
		FlightRec::Rec(FlightRec::EvCpWrite, FlightRec::CpWriteData, 0x1000, 4, 9);
		std::this_thread::sleep_for(std::chrono::milliseconds(1300)); // one 1 Hz sample: EvMem + EvCounters
		FlightRec::DumpNow("light", 0);
	} else if (mode == "verbose") {
		setenv("KYTY_FLIGHTREC", "2", 1);
		for (int i = 0; i < 100; i++) {
			FlightRec::Rec(FlightRec::EvDraw, static_cast<uint64_t>(i));
		}
		FlightRec::DumpNow("verbose", 0);
	}
	return 0;
}

static int RunChild(const char* self, const char* mode, const std::string& dir) {
	pid_t c = fork();
	if (c == 0) {
		execl(self, self, "--child", mode, dir.c_str(), nullptr);
		_exit(127);
	}
	int st = 0;
	waitpid(c, &st, 0);
	return st;
}

int main(int argc, char** argv) {
	if (argc > 3 && std::strcmp(argv[1], "--child") == 0) {
		return ChildMain(argv[2], argv[3]);
	}
	const std::string dir = argc > 1 ? argv[1] : "/tmp/frt_out";
	std::string       cmd = "rm -rf '" + dir + "' && mkdir -p '" + dir + "'";
	(void)!std::system(cmd.c_str());
	setenv("KYTY_FLIGHTREC_DIR", dir.c_str(), 1);
	setenv("KYTY_FLIGHTREC_N", "4096", 1);
	setenv("KYTY_FLIGHTREC_TRIGGER", "5,p3", 1);
	const std::string ctl = dir + "/ctl";
	setenv("KYTY_FLIGHTREC_CTL", ctl.c_str(), 1);
	setenv("KYTY_FLIGHTREC_EXIT", "0", 1);

	// 1. off: nothing recorded, no dump (child process with KYTY_FLIGHTREC=0).
	if (pid_t c = fork(); c == 0) {
		setenv("KYTY_FLIGHTREC", "0", 1);
		FlightRec::Rec(1000, 1);
		FlightRec::DumpNow("off", 0);
		_exit(FlightRec::On() ? 1 : 0);
	} else {
		int st = 0;
		waitpid(c, &st, 0);
		CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0);
		CHECK(List(dir, "_off.").empty());
	}
	setenv("KYTY_FLIGHTREC", "1", 1);
	CHECK(FlightRec::On());
	CHECK(!FlightRec::Verbose());

	// 2. concurrency: 4 threads x 10000 records into their own rings (wraps 4096).
	std::vector<std::thread> ts;
	for (int i = 0; i < 4; i++) {
		ts.emplace_back([i] {
			for (uint64_t k = 1; k <= 10000; k++) {
				FlightRec::Rec(1000, k, static_cast<uint64_t>(i));
			}
		});
	}
	for (auto& t: ts) {
		t.join();
	}
	FlightRec::RecShared(FlightRec::EvFault, 0x1234, 1);
	FlightRec::DumpNow("conc", 0);
	auto files = List(dir, ".conc.");
	CHECK(files.size() == 1);
	if (!files.empty()) {
		const auto p = Parse(files[0]);
		CHECK(p.version == 2 && p.level == 1);
		CHECK(p.rings >= 6); // shared + 4 workers + this thread (EvTrigger) + the monitor thread
		CHECK(p.bad == 0);
		CHECK(Count(p, 1000) == 4 * 4096);
		CHECK(std::string(p.why) == "conc");
	}

	// 3. flip / present triggers and the control file (async dumps on helper threads).
	for (int i = 0; i < 6; i++) {
		FlightRec::NoteFlip(0, 0, 0);
		FlightRec::NotePresent();
	}
	FILE* cf = std::fopen(ctl.c_str(), "w");
	std::fclose(cf);
	FlightRec::NoteFlip(0, 0, 0);
	std::this_thread::sleep_for(std::chrono::milliseconds(500));
	CHECK(List(dir, ".flip.").size() == 1);
	CHECK(List(dir, ".present.").size() == 1);
	CHECK(List(dir, ".ctl.").size() == 1);
	CHECK(access(ctl.c_str(), F_OK) != 0);
	CHECK(FlightRec::Detail::g_flip.load() == 7);

	// 4. crash path: SIGABRT in a child dumps through the signal-safe writer, then dies by SIGABRT.
	if (pid_t c = fork(); c == 0) {
		FlightRec::Rec(1000, 42);
		std::abort();
	} else {
		int st = 0;
		waitpid(c, &st, 0);
		CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT);
		CHECK(List(dir, ".abort.").size() == 1);
	}
	// SIGINFO dump (kill -INFO), process continues.
	raise(SIGINFO);
	CHECK(List(dir, ".siginfo.").size() == 1);

	// 5. max dumps (8): further dumps are refused.
	for (int i = 0; i < 10; i++) {
		FlightRec::DumpNow("many", 0);
	}
	CHECK(List(dir, ".flightrec").size() <= 13); // 12 in this process + the child's abort dump

	// 6. hash: content fingerprint.
	char a[100], b[100];
	std::memset(a, 1, sizeof(a));
	std::memcpy(b, a, sizeof(b));
	CHECK(FlightRec::Hash(a, 100) == FlightRec::Hash(b, 100));
	b[57] = 2;
	CHECK(FlightRec::Hash(a, 100) != FlightRec::Hash(b, 100));
	CHECK(FlightRec::Hash(a, 99) != FlightRec::Hash(a, 100));

	// 7. light mode: per-draw events are counted, not recorded; low-rate events and the 1 Hz samples are.
	{
		const char* self = argv[0];
		CHECK(RunChild(self, "light", dir) == 0);
		auto f = List(dir, "light.0.light");
		CHECK(f.size() == 1);
		if (!f.empty()) {
			const auto p = Parse(f[0]);
			CHECK(Count(p, FlightRec::EvDraw) == 0 && Count(p, FlightRec::EvUpload) == 0);
			CHECK(Count(p, FlightRec::EvCpWrite) == 1);
			CHECK(Count(p, FlightRec::EvMem) >= 1);
			bool counted = false;
			for (const auto& r: p.recs) {
				counted = counted || (r.ev == FlightRec::EvCounters && r.a[0] >= 900);
			}
			CHECK(counted);
		}
		CHECK(RunChild(self, "verbose", dir) == 0);
		f = List(dir, "verbose.0.verbose");
		CHECK(f.size() == 1);
		if (!f.empty()) {
			const auto p = Parse(f[0]);
			CHECK(p.level == 2 && Count(p, FlightRec::EvDraw) == 100);
		}
		// window: only the records of the last second survive
		CHECK(RunChild(self, "window", dir) == 0);
		f = List(dir, "window.0.win");
		CHECK(f.size() == 1);
		if (!f.empty()) {
			const auto p = Parse(f[0]);
			uint64_t   old_n = 0, new_n = 0;
			for (const auto& r: p.recs) {
				old_n += r.ev == 1000 && r.a[0] < 10 ? 1 : 0;
				new_n += r.ev == 1000 && r.a[0] >= 10 ? 1 : 0;
			}
			CHECK(old_n == 0 && new_n == 5);
			CHECK(p.window_ns == 1000000000ull);
		}
		// monitor: present gap > 2 s, heartbeat gap > 2 s (+ backtrace on arm64), SIGTERM
		CHECK(RunChild(self, "stall", dir) == 0);
		f = List(dir, "stall.1.stall");
		CHECK(f.size() == 1);
		if (!f.empty()) {
			CHECK(Count(Parse(f[0]), FlightRec::EvStall) >= 1);
		}
		CHECK(RunChild(self, "mainhang", dir) == 0);
		f = List(dir, "mainhang.0.mainhang");
		CHECK(f.size() == 1);
		if (!f.empty()) {
			const auto p = Parse(f[0]);
			CHECK(Count(p, FlightRec::EvStall) >= 1);
#if defined(__arm64__)
			CHECK(Count(p, FlightRec::EvBacktrace) >= 2);
#endif
		}
		int st = RunChild(self, "sigterm", dir);
		CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGTERM);
		CHECK(List(dir, "sigterm.0.sigterm").size() == 1);
	}

	// Bench: Rec on this thread's ring, and Hash.
	constexpr int N = 10000000;
	auto          t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < N; i++) {
		FlightRec::Rec(FlightRec::EvDraw, static_cast<uint64_t>(i), 2, 3, 4);
	}
	auto   t1 = std::chrono::steady_clock::now();
	double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
	std::vector<uint8_t> buf(64 * 1024, 7);
	uint64_t             sink = 0;
	t0                        = std::chrono::steady_clock::now();
	for (int i = 0; i < 2000; i++) {
		buf[0] = static_cast<uint8_t>(i);
		sink ^= FlightRec::Hash(buf.data(), buf.size());
	}
	t1        = std::chrono::steady_clock::now();
	double gb = 2000.0 * buf.size() / std::chrono::duration<double>(t1 - t0).count() / 1e9;
	std::printf("bench: Rec %.1f ns/record; Hash %.2f GB/s (%llx)\n", ns, gb, static_cast<unsigned long long>(sink & 1));
	std::printf("%d checks, %d failed\n", g_checks, g_fail);
	return g_fail == 0 ? 0 : 1;
}
