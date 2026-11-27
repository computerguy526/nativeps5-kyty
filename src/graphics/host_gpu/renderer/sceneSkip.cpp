#include "graphics/host_gpu/renderer/sceneSkip.h"

#include "graphics/host_gpu/renderer/pipeline/launchPhase.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#endif

namespace Libs::Graphics::SceneSkip {

namespace {

struct Runtime {
	bool                  keys = false;
	bool                  table_loaded = false;
	Table                 table;
	Controller            ctl;
	std::string           state_path;
	std::string           scene;
	std::mutex            mu;
	std::mutex            toast_mu;
	Toast                 toast;
	std::atomic<uint64_t> toast_until_ms {0};
	std::atomic<uint32_t> toast_revision {0};
};

uint64_t NowMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool EnvOn(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && v[0] == '1' && v[1] == '\0';
}

Runtime& R() {
	static Runtime* rt = [] {
		auto* r   = new Runtime;
		r->keys   = EnvOn("KYTY_SCENE_KEYS");
		if (const char* f = std::getenv("KYTY_SCENE_SKIP_FILE"); f != nullptr && f[0] != '\0') {
			std::ifstream in(f);
			if (in) {
				std::stringstream ss;
				ss << in.rdbuf();
				r->table        = ParseTable(ss.str());
				r->table_loaded = true;
				std::fprintf(stderr, "[scenery] table %s: %zu presets, %zu skip rows, %u bad lines, %u over the limit\n", f,
				             r->table.presets.size(), r->table.skips.size(), r->table.bad_lines, r->table.dropped);
			} else {
				std::fprintf(stderr, "[scenery] table %s not readable: instant skip off\n", f);
			}
		}
		if (const char* s = std::getenv("KYTY_SCENE_STATE_FILE"); s != nullptr) r->state_path = s;
		r->ctl.Init(r->table);
		return r;
	}();
	return *rt;
}

// Whether anything of this feature is on (decided once; the flip hook returns at once when not).
bool FeatureOn() {
	static const bool on = std::getenv("KYTY_SCENE_KEYS") != nullptr || std::getenv("KYTY_SCENE_SKIP_FILE") != nullptr;
	return on;
}

std::string CurrentScene() {
#if defined(__APPLE__) || defined(__linux__)
	using Fn       = const char* (*)();
	static Fn func = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "nps5_vfs_current_scene"));
	if (func != nullptr) {
		if (const char* name = func(); name != nullptr) return LaunchPhase::ParseLastScene(name);
	}
#endif
	return {};
}

// Under r.mu: publish the pairs of the current scene for the current mask.
uint32_t Republish(Runtime& r) {
	const auto rows = SelectActive(r.table, r.scene, r.ctl.Current());
	g_active.Publish(rows);
	return static_cast<uint32_t>(rows.size());
}

void WriteState(const Runtime& r) {
	if (r.state_path.empty()) return;
	const auto tmp = r.state_path + ".tmp";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) return;
		out << r.ctl.StateText();
	}
	std::rename(tmp.c_str(), r.state_path.c_str());
}

} // namespace

bool KeysEnabled() noexcept {
	static const bool on = EnvOn("KYTY_SCENE_KEYS");
	return on;
}

bool HandleKey(Key key, bool shift, bool pressed) {
	if (!KeysEnabled()) return false;
	auto&                       r = R();
	std::lock_guard<std::mutex> lock(r.mu);
	const auto                  res = r.ctl.OnKey(true, key, shift, pressed);
	if (!res.consumed || !pressed) return res.consumed;
	uint32_t pairs = 0;
	if (res.changed) {
		pairs = Republish(r);
		WriteState(r);
		std::fprintf(stderr, "%s\n", r.ctl.LogLine(res.cat, pairs, r.scene).c_str());
	} else {
		pairs = static_cast<uint32_t>(SelectActive(r.table, r.scene, r.ctl.Current()).size());
	}
	{
		std::lock_guard<std::mutex> t(r.toast_mu);
		r.toast = r.ctl.MakeToast(res.headline, pairs, res.show);
	}
	r.toast_until_ms.store(NowMs() + kToastMs, std::memory_order_release);
	r.toast_revision.fetch_add(1, std::memory_order_relaxed);
	return true;
}

void OnCpFlip() {
	if (!FeatureOn()) return;
	auto&                       r = R();
	auto                        scene = CurrentScene();
	std::lock_guard<std::mutex> lock(r.mu);
	if (scene != r.scene) {
		r.scene = std::move(scene);
		r.ctl.MarkLoaded(); // a level was entered: its data is the one the launcher staged
		const auto pairs = Republish(r);
		std::fprintf(stderr, "[scenery] scene %s: %u shader pairs active (%s)\n", r.scene.empty() ? "-" : r.scene.c_str(), pairs,
		             r.ctl.RowText().c_str());
	}
}

bool ToastWanted() noexcept {
	if (!KeysEnabled()) return false;
	return NowMs() < R().toast_until_ms.load(std::memory_order_acquire);
}

uint32_t ToastRevision() noexcept {
	if (!KeysEnabled()) return 0;
	return R().toast_revision.load(std::memory_order_relaxed);
}

bool ToastView(char* line1, size_t cap1, char* line2, size_t cap2, float& alpha) {
	if (!KeysEnabled()) return false;
	auto&      r     = R();
	const auto now   = NowMs();
	const auto until = r.toast_until_ms.load(std::memory_order_acquire);
	if (now >= until) return false;
	alpha = until - now >= kToastFadeMs ? 1.0f : static_cast<float>(until - now) / static_cast<float>(kToastFadeMs);
	std::lock_guard<std::mutex> t(r.toast_mu);
	std::snprintf(line1, cap1, "%s", r.toast.line1.c_str());
	std::snprintf(line2, cap2, "%s", r.toast.line2.c_str());
	return true;
}

std::string OverlayRow() {
	if (!KeysEnabled()) return {};
	auto&                       r = R();
	std::lock_guard<std::mutex> lock(r.mu);
	return r.ctl.RowText();
}

} // namespace Libs::Graphics::SceneSkip
