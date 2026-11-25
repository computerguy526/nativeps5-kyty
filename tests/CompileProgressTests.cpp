#include "graphics/compileProgress.h"

#include <cstdio>
#include <thread>
#include <vector>

using namespace Libs::Graphics::CompileProgress;

static int g_failed = 0;
#define CHECK(cond)                                                                      \
	do {                                                                                 \
		if (!(cond)) {                                                                   \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
			g_failed++;                                                                  \
		}                                                                                \
	} while (0)

int main() {
	CHECK(FormatStatus({}).empty());
	CHECK(FormatStatus({.replay_done = 312, .replay_total = 510}) == "compiling shaders 312/510");
	CHECK(FormatStatus({.live = 3}) == "compiling shader (live) 3");
	CHECK(FormatStatus({.replay_done = 1, .replay_total = 2, .live = 3}) == "compiling shaders 1/2 + live 3");
	CHECK(FormatStatus({.live = 1, .waiting = 1}) == "compiling shader (live) 1 (draw waiting)");
	CHECK(FormatStatus({.waiting = 2}) == "compiling shader (live) 2 (draw waiting)");
	CHECK(FormatStatus({.replay_done = 5, .replay_total = 5}).empty()); // a finished replay shows nothing
	CHECK(FormatStatus({.live = -1}).empty());

	// Window title: always valid UTF-8 (a nil NSString from invalid bytes crashed -[NSWindow setTitle:]).
	CHECK(ValidUtf8Title("Game") == "Game");
	CHECK(ValidUtf8Title("Game \xE2\x80\x94 x\xE2\x80\xA6") == "Game \xE2\x80\x94 x\xE2\x80\xA6");
	CHECK(ValidUtf8Title("A\xE2\x80") == "A\xEF\xBF\xBD\xEF\xBF\xBD");   // truncated sequence
	CHECK(ValidUtf8Title("A\xFF" "B") == "A\xEF\xBF\xBD" "B");
	CHECK(ValidUtf8Title("A\xC0\xAF") == "A\xEF\xBF\xBD\xEF\xBF\xBD");  // overlong
	CHECK(ValidUtf8Title("A\xED\xA0\x80") == "A\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD"); // surrogate
	CHECK(ValidUtf8Title(std::string("ab\0cd", 5)) == "ab");
	CHECK(ValidUtf8Title("").empty());
	CHECK(ComposeTitle("Game", {}, true) == "Game");
	CHECK(ComposeTitle("Game", {}, false) == "Game \xE2\x80\x94 loading\xE2\x80\xA6");
	CHECK(ComposeTitle("Game", {.replay_done = 3, .replay_total = 9}, false) ==
	      "Game \xE2\x80\x94 compiling shaders 3/9");
	CHECK(ComposeTitle("Game", {.live = 2}, true) == "Game \xE2\x80\x94 compiling shader (live) 2");

	TitleThrottle t;
	CHECK(t.ShouldSet(1000, "a"));
	CHECK(!t.ShouldSet(1010, "a"));  // unchanged
	CHECK(!t.ShouldSet(1100, "b"));  // changed but within 250 ms
	CHECK(t.ShouldSet(1250, "b"));   // interval passed
	CHECK(!t.ShouldSet(5000, "b"));  // unchanged, never repeated
	CHECK(t.ShouldSet(5000, "a"));

	// Counters and scopes.
	{
		LiveScope a;
		WaitScope w;
		const auto s = Read();
		CHECK(s.live == 1 && s.waiting == 1);
		MarkReplayThread();
		{
			LiveScope b; // a replay thread does not count
			CHECK(Read().live == 1);
		}
		t_replay_thread = false;
	}
	CHECK(Read().live == 0 && Read().waiting == 0);
	Get().shader_total = 10;
	Get().shader_done  = 4;
	Get().pipeline_total = 5;
	CHECK(FormatStatus(Read()) == "compiling shaders 4/15");
	ClearShaderReplay();
	ClearPipelineReplay();
	CHECK(FormatStatus(Read()).empty());

	{
		// card3 unit: every compile path feeds one total/done pair; total never shrinks, done never above total,
		// monotonic reads while several threads add events; a finished session restarts at 0.
		namespace CP = Libs::Graphics::CompileProgress;
		auto& c = CP::Get();
		c.shader_total.store(819); c.shader_done.store(0);
		CP::UnitAddTotal(819);               // library replay starts
		CP::UnitAddTotal(1396);              // pipeline list known (Start)
		c.pipeline_total.store(1396);
		auto snap = CP::Read();
		CHECK(snap.list_total == 2215 && snap.list_done == 0);
		std::atomic<bool> stop {false};
		uint64_t last_done = 0, last_total = 0; bool mono = true;
		std::thread reader([&] {
			while (!stop.load()) {
				const auto s2 = CP::Read();
				if (s2.list_done < last_done || s2.list_total < last_total || s2.list_done > s2.list_total) mono = false;
				last_done = s2.list_done; last_total = s2.list_total;
			}
		});
		std::vector<std::thread> workers;
		for (int t = 0; t < 4; t++) workers.emplace_back([&] { for (int i = 0; i < 819 / 4 + 1 && i < 205; i++) { c.shader_done.fetch_add(1); CP::UnitAddDone(1); } });
		for (int t = 0; t < 3; t++) workers.emplace_back([&] { for (int i = 0; i < 400; i++) { c.pipeline_done.fetch_add(1); CP::UnitAddDone(1); } });
		for (int i = 0; i < 100; i++) CP::PrewarmRequested(); // discovered extra work grows the total
		for (int t = 0; t < 2; t++) workers.emplace_back([&] { for (int i = 0; i < 50; i++) CP::PrewarmFinished(); });
		for (auto& w: workers) w.join();
		stop = true; reader.join();
		CHECK(mono);
		snap = CP::Read();
		CHECK(snap.list_total == 2315);
		CHECK(snap.list_done == 820 + 1200 + 100); // 4*205 shader + 3*400 pipelines + 100 prewarm
		CHECK(snap.replay_total == 819 + 1396 + 100 && snap.replay_done == 820 + 1200 + 100);
		// Stopped replays: the unit completes through the Clear functions.
		CP::ClearPipelineReplay();
		CP::ClearShaderReplay();
		snap = CP::Read();
		CHECK(snap.list_total == 2315 && snap.list_done == 2315); // 196 + 0 never-ran entries added as done
		CP::UnitAddTotal(10); // new session: restarts from 0
		snap = CP::Read();
		CHECK(snap.list_total == 10 && snap.list_done == 0);
		CP::UnitAddDone(10);
		c.prewarm_total.store(0); c.prewarm_done.store(0);
	}
	if (g_failed == 0) std::printf("compile_progress_tests: all passed\n");
	return g_failed == 0 ? 0 : 1;
}
