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

int main(int argc, char** argv) {
    Require(argc >= 2, "expected mode argument");
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
