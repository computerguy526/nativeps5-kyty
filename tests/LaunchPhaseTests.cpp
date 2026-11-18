#include "graphics/host_gpu/renderer/pipeline/launchPhase.h"
#include "graphics/loadCard.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using namespace Libs::Graphics::LaunchPhase;

static int g_failed = 0;
#define CHECK(cond)                                                          \
	do {                                                                     \
		if (!(cond)) {                                                       \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
			g_failed++;                                                      \
		}                                                                    \
	} while (0)

int main() {
	// Key files (prepare delta).
	{
		const auto keys = ParseKeyFile("0000000000000010\n# comment\n\n  00000000000000ff \r\nxyz\n12345678901234567\n0\n");
		CHECK(keys.size() == 2 && keys.contains(0x10) && keys.contains(0xff)); // malformed, too long and 0 dropped
		CHECK(ParseKeyFile("").empty());
		const auto text = FormatKeyFile({0x10, 0xdeadbeefcafe0001ull});
		CHECK(text == "0000000000000010\ndeadbeefcafe0001\n");
		const auto back = ParseKeyFile(text);
		CHECK(back.size() == 2 && back.contains(0xdeadbeefcafe0001ull));
	}
	// Scene tagging and selection.
	{
		const auto always = ParsePatterns("worldmap, hub_*,ui_*");
		CHECK(always.size() == 3 && always[1] == "hub_*");
		CHECK(MatchesAny("worldmap", always) && MatchesAny("hub_crashsite", always) && MatchesAny("ui_title", always));
		CHECK(!MatchesAny("worldmap2", always) && !MatchesAny("hoover_beach", always) && !MatchesAny("", always));
		CHECK(InSceneHold({}, "hoover_beach", always));                      // untagged: before any scene (title, menus)
		CHECK(InSceneHold({""}, "hoover_beach", always));                    // journal record with an empty marker
		CHECK(InSceneHold({"hoover_beach"}, "hoover_beach", always));        // the target scene
		CHECK(InSceneHold({"garden", "hoover_beach"}, "hoover_beach", always)); // a list record seen in several scenes
		CHECK(InSceneHold({"hub_crashsite"}, "hoover_beach", always));       // always set
		CHECK(!InSceneHold({"garden"}, "hoover_beach", always));             // another scene: background
		CHECK(!InSceneHold({"garden"}, "", always));                         // no target known: only untagged + always
		CHECK(!InSceneHold({"garden"}, "hoover_beach", {}));
	}
	// Last-scene memory.
	{
		const auto always = ParsePatterns("worldmap,ui_*");
		CHECK(NextLastScene("", "hoover_beach", always) == "hoover_beach");
		CHECK(NextLastScene("hoover_beach", "worldmap", always) == "hoover_beach"); // an always scene never replaces it
		CHECK(NextLastScene("hoover_beach", "", always) == "hoover_beach");
		CHECK(NextLastScene("hoover_beach", "garden", always) == "garden");
		CHECK(ParseLastScene("hoover_beach\n") == "hoover_beach");
		CHECK(ParseLastScene("").empty() && ParseLastScene("a b\n").empty() && ParseLastScene(std::string(200, 'a')).empty());
		CHECK(ParseLastScene("bad\x01name").empty());
	}
	// Compile fan-out: logical cores - 4, at least 2; the override wins (1..16); unknown core count = 8.
	{
		CHECK(CompileFanoutFor(14, 0) == 10 && CompileFanoutFor(8, 0) == 4 && CompileFanoutFor(6, 0) == 2);
		CHECK(CompileFanoutFor(4, 0) == 2 && CompileFanoutFor(1, 0) == 2 && CompileFanoutFor(0, 0) == 8);
		CHECK(CompileFanoutFor(64, 0) == 16 && CompileFanoutFor(14, 3) == 3 && CompileFanoutFor(14, 99) == 16);
	}
        // Compile gate: capacity limits concurrent slots, 0 = unlimited, the
        // slot count returns to 0.
        {
          auto &gate = CompileGate::Get();
          gate.Configure(2);
          {
            CompileGate::Slot a;
            CompileGate::Slot b;
            CHECK(gate.Running() == 2);
            std::atomic<bool> got{false};
            std::thread t([&] {
              CompileGate::Slot c; // blocks until a slot frees
              got = true;
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            CHECK(!got.load() && gate.Running() == 2);
            // a and b leave at the end of the scope; c then runs
            gate.Configure(0); // unlimited: c passes at once
            t.join();
            CHECK(got.load());
          }
          CHECK(gate.Running() == 0);
        }
        // Background continuation: the card follows the hold, not the unit
        // (scene mode ends the hold before the unit is done).
        {
          using namespace Libs::Graphics;
          auto &c = CompileProgress::Get();
          c.unit_total.store(200), c.unit_done.store(50);
          c.launch_hold.store(1);
          CHECK(LoadCard::LaunchPhasePending()); // held: the card stays
          c.launch_hold.store(2);
          CHECK(
              !LoadCard::LaunchPhasePending()); // hold over, 150 still building
                                                // behind the game: no full card
          c.launch_hold.store(0);
          CHECK(
              LoadCard::LaunchPhasePending()); // no hold this session
                                               // (KYTY_PIPELINE_REPLAY_ALL=0):
                                               // card3's rule
          c.unit_done.store(200);
          CHECK(!LoadCard::LaunchPhasePending());
        }
        std::printf(g_failed ? "LaunchPhaseTests: %d FAILED\n"
                             : "LaunchPhaseTests: ok\n",
                    g_failed);
        return g_failed != 0;
}
