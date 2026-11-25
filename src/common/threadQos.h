#ifndef KYTY_COMMON_THREADQOS_H
#define KYTY_COMMON_THREADQOS_H

// Frame-critical host threads: QoS class and a Mach time-constraint policy (nxnative 4c71083c: a thread that
// declares the frame deadline and the CPU time it needs is clocked and placed so that it fits; one left on the
// default policy sits on an efficiency core).  macOS only; everywhere else every call is a no-op.
//
//   NPS5_THREAD_QOS=0          kill switch for everything here
//   NPS5_QOS_GPU=qos|rt        (default off) qos: the threads take QOS_CLASS_USER_INTERACTIVE.
//                              rt: also a governor thread declares each one a time-constraint policy,
//                              period = the measured frame time, computation = measured CPU time per frame
//   NPS5_QOS_GPU_<GROUP>=0|1   per group, default 1 once NPS5_QOS_GPU is on: COMMIT (draw-commit, draw-finish),
//                              QWORKER (queue workers), PRESENT (the presenter), PACER (VideoOut vblank)
//   NPS5_QOS_LOG=1             log registrations and declarations
//   NPS5_QOS_AB_PERIOD=<s>     in-run A/B of the rt mode: declarations on for one period, off for the next (absolute
//                              monotonic clock, the same rule the guest runtime's governor uses)
//
// Header-only; Kyty (nps5): ported from AnyPS5 nps5-qos b4708323ff/b17c2255e0. Groups here: COMMIT = the command-processor
// thread (RaiseCurrentThreadPriority), QWORKER = draw-prep workers, PRESENT = the queue-submission thread; frames are
// counted at the guest flip (HangTrace::RecordFlip). Default off, no effect unless NPS5_QOS_GPU is set.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <time.h>
#include <unistd.h>
#endif

namespace ThreadQos {

enum class Group { Commit, Qworker, Present, Pacer };

inline const char* GroupName(Group group) {
    switch (group) {
    case Group::Commit: return "COMMIT";
    case Group::Qworker: return "QWORKER";
    case Group::Present: return "PRESENT";
    case Group::Pacer: return "PACER";
    }
    return "?";
}

inline bool Log() {
    static const bool on = [] { const char* e = std::getenv("NPS5_QOS_LOG"); return e && *e && *e != '0'; }();
    return on;
}

// 0 = off, 1 = qos, 2 = rt
inline int Mode() {
    static const int mode = [] {
        const char* master = std::getenv("NPS5_THREAD_QOS");
        if (master && std::strcmp(master, "0") == 0) return 0;
        const char* e = std::getenv("NPS5_QOS_GPU");
        if (!e || !*e || std::strcmp(e, "0") == 0) return 0;
        if (std::strcmp(e, "rt") == 0) return 2;
        return 1;
    }();
    return mode;
}

inline bool GroupOn(Group group) {
    const std::string name = std::string("NPS5_QOS_GPU_") + GroupName(group);
    const char* e = std::getenv(name.c_str());
    return !(e && std::strcmp(e, "0") == 0);
}

#if defined(__APPLE__)
namespace Detail {

struct Entry {
    std::string name;
    mach_port_t port = 0;
    std::uint64_t cpu0 = 0;
    std::uint64_t comp = 0;
    bool dead = false;
    pthread_t thread = nullptr;
    pthread_override_t boost = nullptr;  // qos mode under NPS5_QOS_AB_PERIOD: the toggled class override
};

inline std::mutex& Mutex() { static std::mutex m; return m; }
inline std::vector<Entry>& Entries() { static std::vector<Entry> e; return e; }
inline std::atomic<std::uint64_t>& Frames() { static std::atomic<std::uint64_t> f{0}; return f; }

inline std::uint32_t ToAbs(std::uint64_t ns) {
    static const mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t t; mach_timebase_info(&t); return t; }();
    return static_cast<std::uint32_t>(ns * tb.denom / tb.numer);
}

inline bool Declare(Entry& entry, std::uint64_t period, std::uint64_t comp) {
    thread_time_constraint_policy_data_t policy;
    policy.period = ToAbs(period);
    policy.computation = ToAbs(comp);
    policy.constraint = ToAbs(period);
    policy.preemptible = 1;
    const kern_return_t result = thread_policy_set(entry.port, THREAD_TIME_CONSTRAINT_POLICY, reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    if (Log()) std::fprintf(stderr, "[qos] time-constraint '%s' period %.2f ms computation %.2f ms: %s\n", entry.name.c_str(), period / 1e6, comp / 1e6, result == KERN_SUCCESS ? "set" : "FAILED");
    if (result == KERN_SUCCESS) entry.comp = comp;
    return result == KERN_SUCCESS;
}

inline bool AbPhaseOn() {
    static const double period = [] { const char* e = std::getenv("NPS5_QOS_AB_PERIOD"); return e && *e ? std::atof(e) : 0.0; }();
    if (period <= 0) return true;
    const double now = static_cast<double>(clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW));
    return ((static_cast<std::uint64_t>(now / (period * 1e9))) & 1) == 0;
}

inline bool AbEnabled() {
    const char* e = std::getenv("NPS5_QOS_AB_PERIOD");
    return e && *e && std::atof(e) > 0;
}

inline void SetBoost(bool on) {
    std::lock_guard lock(Mutex());
    for (auto& entry : Entries()) {
        if (entry.dead) continue;
        if (on && !entry.boost) entry.boost = pthread_override_qos_class_start_np(entry.thread, QOS_CLASS_USER_INTERACTIVE, 0);
        if (!on && entry.boost) { pthread_override_qos_class_end_np(entry.boost); entry.boost = nullptr; }
    }
}

inline void Governor() {
    pthread_setname_np("nps5-qos");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    std::uint64_t frames0 = Frames().load(), time0 = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
    bool lastPhase = true;
    int ticks = 0;
    for (;;) {
        usleep(100 * 1000);
        const bool phase = AbPhaseOn();
        if (phase != lastPhase) {
            lastPhase = phase;
            if (Log()) std::fprintf(stderr, "[qos-ab] gpu governor phase %s\n", phase ? "on" : "off");
            if (Mode() == 1) SetBoost(phase);
            if (!phase) {
                std::lock_guard lock(Mutex());
                for (auto& entry : Entries()) {
                    if (entry.dead || entry.comp == 0) continue;
                    thread_standard_policy_data_t standard = {};
                    thread_policy_set(entry.port, THREAD_STANDARD_POLICY, reinterpret_cast<thread_policy_t>(&standard), THREAD_STANDARD_POLICY_COUNT);
                    entry.comp = 0;
                }
            }
        }
        if (Mode() == 1 && phase) SetBoost(true);  // threads registered since the last tick
        if (++ticks < 10) continue;
        ticks = 0;
        if (Mode() == 1) continue;
        if (!phase) { frames0 = Frames().load(); time0 = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW); continue; }
        const std::uint64_t frames1 = Frames().load(), time1 = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
        const std::uint64_t dframes = frames1 - frames0, dtime = time1 - time0;
        frames0 = frames1;
        time0 = time1;
        if (dframes < 8) continue;  // loading or paused: no frame rate to declare
        std::uint64_t period = dtime / dframes;
        if (period < 8'000'000ull) period = 8'000'000ull;
        if (period > 100'000'000ull) period = 100'000'000ull;
        std::lock_guard lock(Mutex());
        for (auto& entry : Entries()) {
            if (entry.dead) continue;
            thread_basic_info_data_t info;
            mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
            if (thread_info(entry.port, THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&info), &count) != KERN_SUCCESS) { entry.dead = true; continue; }
            const std::uint64_t cpu = static_cast<std::uint64_t>(info.user_time.seconds + info.system_time.seconds) * 1'000'000'000ull + static_cast<std::uint64_t>(info.user_time.microseconds + info.system_time.microseconds) * 1000ull;
            const std::uint64_t previous = entry.cpu0;
            entry.cpu0 = cpu;
            if (previous == 0) continue;
            std::uint64_t comp = (cpu - previous) / dframes;
            comp += comp / 8;  // 12% headroom
            if (comp < 1'000'000ull) comp = 1'000'000ull;
            if (comp > period * 9 / 10) comp = period * 9 / 10;
            const std::uint64_t delta = comp > entry.comp ? comp - entry.comp : entry.comp - comp;
            if (entry.comp != 0 && delta * 4 < entry.comp) continue;  // within 25%: keep what is declared
            Declare(entry, period, comp);
        }
    }
}

}  // namespace Detail
#endif

// Call on the thread itself, at its start.
inline void Register(Group group, const char* name) {
#if defined(__APPLE__)
    const int mode = Mode();
    if (mode == 0 || !GroupOn(group)) return;
    const bool toggled = Detail::AbEnabled();
    if (mode == 1 && !toggled) pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // toggled runs boost by override instead; rt leaves the class alone (its off phase is the baseline)
    if (Log()) std::fprintf(stderr, "[qos] %s '%s': %s\n", GroupName(group), name, mode == 2 ? "time-constraint governor" : "user-interactive");
    if (mode != 2 && !toggled) return;
    {
        std::lock_guard lock(Detail::Mutex());
        Detail::Entry entry;
        entry.name = name;
        entry.port = pthread_mach_thread_np(pthread_self());
        entry.thread = pthread_self();
        Detail::Entries().push_back(std::move(entry));
    }
    static std::once_flag once;
    std::call_once(once, [] { std::thread(Detail::Governor).detach(); });
#else
    (void)group; (void)name;
#endif
}

// One call per presented frame (or per vblank) on any thread of the library: the governor derives the period from it.
inline void NoteFrame() {
#if defined(__APPLE__)
    if (Mode() == 2) Detail::Frames().fetch_add(1, std::memory_order_relaxed);
#endif
}

}  // namespace ThreadQos

#endif
