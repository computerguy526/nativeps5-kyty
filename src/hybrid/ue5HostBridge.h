#ifndef EMULATOR_SRC_HYBRID_UE5HOSTBRIDGE_H_
#define EMULATOR_SRC_HYBRID_UE5HOSTBRIDGE_H_

// The ABI 1.8 glue between the host's callback table (kytyGpuApi.h) and the UE5 warm list's seam
// (graphics/host_gpu/renderer/pipeline/ue5CodeWarm.h): the extract trampolines, the checked guest
// reader and the main-thread hop. Header only and GPU-free: kytyGpuBridge.cpp installs it, and
// tests/Ue5BridgeTests.cpp drives the very same code with a fake host (kyty_gpu.dylib itself cannot
// be loaded without a GPU, so this is the closest buildable equivalent).

#include "graphics/host_gpu/renderer/pipeline/ue5CodeWarm.h"
#include "hybrid/kytyGpuApi.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

namespace Hybrid::Ue5Bridge {

namespace Ue5 = Libs::Graphics::Ue5CodeWarm;

// What the bridge knows about the process, as callbacks so a test can stand in for it.
struct Env {
	const KytyGpuHostCallbacks* host = nullptr;
	const KytyGpuHostInfo*      info = nullptr;
	std::function<bool()>       on_main_thread; // the calling thread is the process's main thread
	std::function<bool()> main_queue_drains;    // the host's main queue runs between window steps
	std::function<void(int level, const std::string&)> log;
};

// extract_ue5_shader_groups: the query's C callbacks forward to the std::function seam. Built
// without exceptions: nothing here throws.
[[nodiscard]] inline Ue5::HostResult Extract(const Env& env, uint64_t decoder, uint32_t abi,
                                             const Ue5::GroupFn&    group,
                                             const Ue5::ProgressFn& progress) {
	struct Context {
		const Ue5::GroupFn*    group;
		const Ue5::ProgressFn* progress;
	} context {&group, &progress};
	KytyGpuUe5Query query {};
	query.struct_size   = sizeof(query);
	query.abi           = abi;
	query.decoder       = decoder;
	query.max_raw_bytes = 64u << 20u;
	query.threads       = 0;
	query.group         = [](void* arg, const KytyGpuUe5Group* g) -> int {
		if (g == nullptr) return 0;
		Ue5::GroupView view;
		view.library      = g->library;
		view.group        = g->group;
		view.shader_count = g->shader_count;
		// A host that hands a null pointer with a size is wrong; an empty view splits to nothing.
		if (g->frequencies != nullptr) view.frequencies = {g->frequencies, g->shader_count};
		if (g->bytes != nullptr) view.bytes = {g->bytes, g->raw_bytes};
		return (*static_cast<Context*>(arg)->group)(view) ? 1 : 0;
	};
	query.progress = [](void* arg, uint32_t done, uint32_t total) {
		(*static_cast<Context*>(arg)->progress)(done, total);
	};
	query.arg = &context;
	KytyGpuUe5Result result {};
	Ue5::HostResult  out;
	out.code             = env.host->extract_ue5_shader_groups(env.host->user, &query, &result);
	out.groups_total     = result.groups_total;
	out.decoded          = result.groups_decoded;
	out.failed           = result.groups_failed;
	out.skipped          = result.groups_skipped;
	out.compressed_bytes = result.compressed_bytes;
	out.raw_bytes        = result.raw_bytes;
	out.ms               = result.ms;
	return out;
}

// Never a raw read of an address taken from a profile: inside the guest window, and through the
// kernel's checked read.
[[nodiscard]] inline bool ReadGuest(const KytyGpuHostInfo& info, uint64_t address,
                                    std::span<uint8_t> out) {
	const uint64_t end = address + out.size();
	if (info.guest_hi <= info.guest_lo || address < info.guest_lo || end < address ||
	    end > info.guest_hi) {
		return false;
	}
#if defined(__APPLE__)
	mach_vm_size_t got = 0;
	return mach_vm_read_overwrite(mach_task_self(), address + info.guest_bias, out.size(),
	                              reinterpret_cast<mach_vm_address_t>(out.data()),
	                              &got) == KERN_SUCCESS &&
	       got == out.size();
#else
	return false;
#endif
}

// Runs `fn` on the host's main thread and returns when it is done. Never blocks the caller in the
// host's queue: the block is queued without waiting and this thread polls, so a window close
// (Ue5::Cancelled) can give the job up (it then never runs), and a main queue that does not drain
// (KYTY_WINDOW_STEPPED=0: one never-returning block) is not waited for at all: `fn` runs on the
// calling thread then.
inline void RunOnMain(const Env& env, std::function<void()> fn) {
	const bool can_hop = env.host->run_on_main_thread != nullptr &&
	                     (env.info->flags & KYTY_GPU_HOST_MAIN_QUEUE) != 0 &&
	                     !(env.on_main_thread && env.on_main_thread());
	if (can_hop && env.main_queue_drains && !env.main_queue_drains()) {
		if (env.log)
			env.log(1, "UE5 code: the window loop is not stepped (KYTY_WINDOW_STEPPED=0): the list "
			           "start runs on the launch thread, not the main thread");
		fn();
		return;
	}
	if (!can_hop) {
		fn();
		return;
	}
	struct Job {
		std::mutex              mutex;
		std::condition_variable cv;
		bool                    started = false, done = false, abandoned = false;
		std::function<void()>   fn;
	};
	auto job         = std::make_shared<Job>();
	job->fn          = std::move(fn);
	auto*     arg    = new std::shared_ptr<Job>(job);
	const int queued = env.host->run_on_main_thread(
	    env.host->user,
	    [](void* a) {
		    auto* holder = static_cast<std::shared_ptr<Job>*>(a);
		    auto  j      = *holder;
		    delete holder;
		    {
			    std::scoped_lock lock(j->mutex);
			    if (j->abandoned) return;
			    j->started = true;
		    }
		    j->fn();
		    {
			    std::scoped_lock lock(j->mutex);
			    j->done = true;
		    }
		    j->cv.notify_all();
	    },
	    arg, 0);
	if (queued != 0) {
		delete arg;
		job->fn(); // the host would not queue it: inline, as before
		return;
	}
	std::unique_lock lock(job->mutex);
	while (!job->done) {
		if (job->cv.wait_for(lock, std::chrono::milliseconds(20), [&] { return job->done; })) break;
		if (Ue5::Cancelled() && !job->started) {
			job->abandoned = true; // the process is ending; the main thread may never get to it
			return;
		}
	}
}

// Sets the three seam entries from a 1.8 host (extract_ue5_shader_groups set). `env` must outlive
// the process (the bridge's globals).
inline void Install(const Env* env) {
	Ue5::SetHostExtract([env](uint64_t decoder, uint32_t abi, const Ue5::GroupFn& group,
	                          const Ue5::ProgressFn& progress) {
		return Extract(*env, decoder, abi, group, progress);
	});
	Ue5::SetRunOnMain([env](std::function<void()> fn) { RunOnMain(*env, std::move(fn)); });
	Ue5::SetGuestReader([env](uint64_t address, std::span<uint8_t> out) {
		return ReadGuest(*env->info, address, out);
	});
}

} // namespace Hybrid::Ue5Bridge

#endif // EMULATOR_SRC_HYBRID_UE5HOSTBRIDGE_H_
