#include "graphics/compileProgress.h"

#include <cstdio>

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

	if (g_failed == 0) std::printf("compile_progress_tests: all passed\n");
	return g_failed == 0 ? 0 : 1;
}
