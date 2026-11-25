#include "graphics/host_gpu/renderer/pipeline/cacheUpgrade.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace Libs::Graphics::CacheUpgrade;
namespace fs = std::filesystem;

static int g_failed = 0;
static int g_checks = 0;
#define CHECK(cond)                                                       \
	do {                                                                  \
		g_checks++;                                                       \
		if (!(cond)) {                                                    \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			g_failed++;                                                   \
		}                                                                 \
	} while (0)

static void Put(const fs::path& p, const std::string& text) {
	std::ofstream(p, std::ios::binary) << text;
}

int main() {
	const std::string v13 = "v1;codegen=aaaa;fp=1111;drv=dddd";
	const std::string v14 = "v1;codegen=bbbb;fp=2222;drv=dddd";

	// Decide: the table
	CHECK(Decide(std::nullopt, v14, false, false) == Action::Fresh);
	CHECK(Decide(std::nullopt, v14, true, false) == Action::Upgrade);                 // a v1.3.x cache has no stamp
	CHECK(Decide(Stamp {v13, true}, v14, true, false) == Action::Upgrade);           // old stamp -> prepare
	CHECK(Decide(Stamp {v13, false}, v14, true, false) == Action::Upgrade);          // interrupted under yet another stack
	CHECK(Decide(Stamp {v14, true}, v14, true, false) == Action::Warm);              // same -> skip
	CHECK(Decide(Stamp {v14, false}, v14, true, false) == Action::Resume);           // interrupted -> resumes
	CHECK(Decide(Stamp {v14, false}, v14, false, false) == Action::Resume);          // cache file missing but stamp preparing
	CHECK(Decide(std::nullopt, v14, true, true) == Action::Warm);                    // escape env
	CHECK(Decide(Stamp {v13, true}, v14, true, true) == Action::Warm);
	CHECK(NeedsFullReplay(Action::Upgrade) && NeedsFullReplay(Action::Resume) && NeedsFullReplay(Action::Fresh) && !NeedsFullReplay(Action::Warm));

	// Format / Parse round trip and torn input
	CHECK(Parse(Format(Stamp {v14, true})) == std::optional<Stamp>(Stamp {v14, true}));
	CHECK(Parse(Format(Stamp {v14, false})) == std::optional<Stamp>(Stamp {v14, false}));
	CHECK(!Parse(""));
	CHECK(!Parse("identity=x\nstate=prepared"));       // no final newline: torn write
	CHECK(!Parse("identity=\nstate=prepared\n"));      // empty identity
	CHECK(!Parse("identity=x\n"));                     // no state
	CHECK(!Parse("identity=x\nstate=maybe\n"));
	CHECK(Hash16("abc") == Hash16("abc") && Hash16("abc") != Hash16("abd") && Hash16("").size() == 16);

	// Files: the full upgrade lifecycle in a scratch dir
	const fs::path dir = fs::temp_directory_path() / ("cacheupgrade_test_" + std::to_string(std::rand()));
	fs::create_directories(dir);
	const auto cache = dir / "T.bin";
	const auto stamp = dir / "T.cacheprep";
	Put(cache, std::string(1000, 'x'));                                    // a v1.3.x driver cache, no stamp
	CHECK(!Read(stamp));
	auto act = Decide(Read(stamp), v14, fs::exists(cache), false);
	CHECK(act == Action::Upgrade);
	CHECK(SetAside(cache));
	CHECK(!fs::exists(cache) && fs::exists(AsidePath(cache)));             // set aside, not deleted
	CHECK(Write(stamp, Stamp {v14, false}));
	// quit midway: the next start resumes and keeps the aside copy and whatever the partial run saved
	Put(cache, std::string(200, 'y'));
	act = Decide(Read(stamp), v14, fs::exists(cache), false);
	CHECK(act == Action::Resume);
	CHECK(fs::exists(AsidePath(cache)) && fs::file_size(cache) == 200);
	// finished: prepared, aside removed (one generation), next start warm
	CHECK(ReplayComplete(false, 0));
	CHECK(RemoveAside(cache) == 1000);
	CHECK(Write(stamp, Stamp {v14, true}));
	CHECK(!fs::exists(AsidePath(cache)));
	CHECK(Decide(Read(stamp), v14, true, false) == Action::Warm);
	CHECK(RemoveAside(cache) == 0);                                         // nothing left, no error
	// a second upgrade replaces any older aside copy (one generation only)
	Put(AsidePath(cache), std::string(5, 'z'));
	CHECK(Decide(Read(stamp), "v1;codegen=cccc;fp=3333;drv=dddd", true, false) == Action::Upgrade);
	CHECK(SetAside(cache));
	CHECK(fs::file_size(AsidePath(cache)) == 200);
	CHECK(!SetAside(cache));                                                // no cache file: nothing moved

	// an incomplete replay never earns "prepared"
	CHECK(!ReplayComplete(true, 0));
	CHECK(!ReplayComplete(false, 4772));

	// the replay limit follows the cap (7/8)
	CHECK(DefaultReplayLimitMb(512) == 448);
	CHECK(DefaultReplayLimitMb(1024) == 896);

	// a torn stamp on disk reads as no stamp (cache data present -> Upgrade, i.e. redo, never trust it)
	Put(stamp, "identity=v1;codegen=bbbb");
	CHECK(Decide(Read(stamp), v14, true, false) == Action::Upgrade);

	fs::remove_all(dir);
	std::printf("cache_upgrade_tests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
