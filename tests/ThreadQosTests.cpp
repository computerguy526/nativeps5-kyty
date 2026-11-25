// ThreadQos (NPS5_QOS_GPU governor, default off): mode parsing and group switches. Each case runs in its own process
// (the mode is read once): argv[1] = expected Mode(), argv[2] (optional) = group that must be off.
#include "common/threadQos.h"
#include <cstdio>
#include <cstdlib>
#include <string_view>

static void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "thread qos test failed: %s\n", what);
    std::abort();
}

// The governor's pure arithmetic (v1.4 review M7), no device and no environment.
static int Math() {
    using namespace ThreadQos;
    // period = measured frame time clamped to [8 ms, 100 ms]
    Require(PolicyPeriod(16'666'666ull * 60, 60) == 16'666'666ull, "period 60 fps");
    Require(PolicyPeriod(1'000'000ull * 60, 60) == 8'000'000ull, "period clamps up to 8 ms");
    Require(PolicyPeriod(500'000'000ull * 10, 10) == 100'000'000ull, "period clamps down to 100 ms");
    Require(PolicyPeriod(8'000'000ull * 8, 8) == 8'000'000ull && PolicyPeriod(100'000'000ull * 8, 8) == 100'000'000ull, "period edges");
    Require(PolicyPeriod(123, 0) == 100'000'000ull, "no frames: the slowest period, no divide by zero");
    // computation = cpu per frame + 12.5%, in [1 ms, 0.9 period]
    Require(PolicyComputation(8'000'000ull * 10, 10, 16'000'000ull) == 9'000'000ull, "comp +12.5% (8 ms -> 9 ms)");
    Require(PolicyComputation(100'000ull * 10, 10, 16'000'000ull) == 1'000'000ull, "comp floor 1 ms");
    Require(PolicyComputation(40'000'000ull * 10, 10, 16'000'000ull) == 14'400'000ull, "comp ceiling 0.9 * period");
    Require(PolicyComputation(5, 0, 16'000'000ull) == 1'000'000ull, "no frames: floor");
    Require(PolicyComputation(0, 8, 8'000'000ull) == 1'000'000ull, "idle thread: floor");
    // hysteresis: the first declaration is always made, within 25% keeps, 25% or more changes
    Require(!PolicyKeep(0, 5'000'000ull), "first declaration is made");
    Require(PolicyKeep(4'000'000ull, 4'900'000ull) && PolicyKeep(4'000'000ull, 3'100'000ull), "within 25% keeps");
    Require(!PolicyKeep(4'000'000ull, 5'000'000ull) && !PolicyKeep(4'000'000ull, 3'000'000ull), "25% changes");
    // timebase: 1/1 (Intel) is the identity, 125/3 (Apple silicon) shrinks ns to ticks, an exact 125 ns is 3 ticks
    Require(PolicyToAbs(16'000'000ull, 1, 1) == 16'000'000u, "timebase 1/1");
    Require(PolicyToAbs(125, 125, 3) == 3u && PolicyToAbs(16'666'666ull, 125, 3) == 399'999u, "timebase 125/3");
    Require(PolicyToAbs(100'000'000ull, 125, 3) == 2'400'000u, "timebase 125/3 at the 100 ms period");
    // the mode parser: only qos/1/rt turn it on, the kill switch wins
    Require(ParseMode(nullptr, nullptr) == 0 && ParseMode(nullptr, "") == 0 && ParseMode(nullptr, "0") == 0, "unset/empty/0 off");
    Require(ParseMode(nullptr, "off") == 0 && ParseMode(nullptr, "false") == 0 && ParseMode(nullptr, "RT") == 0, "typos are off");
    Require(ParseMode(nullptr, "qos") == 1 && ParseMode(nullptr, "1") == 1 && ParseMode(nullptr, "rt") == 2, "qos/1/rt");
    Require(ParseMode("0", "rt") == 0 && ParseMode("1", "rt") == 2, "NPS5_THREAD_QOS=0 kills it");
    return 0;
}

int main(int argc, char** argv) {
    Require(argc >= 2, "expected mode argument");
    if (std::string_view(argv[1]) == "--math") return Math();
    Require(ThreadQos::Mode() == std::atoi(argv[1]), "Mode()");
    Require(std::string_view(ThreadQos::GroupName(ThreadQos::Group::Commit)) == "COMMIT", "group name");
    if (argc >= 3) {
        const std::string_view off = argv[2];
        for (auto group : {ThreadQos::Group::Commit, ThreadQos::Group::Qworker, ThreadQos::Group::Present, ThreadQos::Group::Pacer})
            Require(ThreadQos::GroupOn(group) == (std::string_view(ThreadQos::GroupName(group)) != off), "GroupOn()");
    }
    // Registering with the mode off (or with a switched-off group) must not start a governor or throw.
    ThreadQos::Register(ThreadQos::Group::Commit, "test");
    ThreadQos::NoteFrame();
    return 0;
}
