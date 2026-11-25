// nps5 archfix T5: AutoCap's shared state must stay usable while the process exits.
//  1. exit test: a static destructor that runs after AutoCap::State()'s statics calls Publish/Snapshot (old code: the std::mutex was
//     already destroyed -> 'mutex lock failed: Invalid argument' -> abort, the test process dies by a signal).
//  2. crash path: a child process whose "crash handler" (a worker thread) ends in exit() while the main thread keeps calling
//     AutoCap::Publish in a loop (WindowContext::RunStep) must leave with the handler's exit code within 2 s and print no
//     'mutex lock failed' (old code: abort by SIGABRT, or the message).
#include "autoCap.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
// Constructed before main (and before the function-local statics first used in main), so destroyed after them.
struct LateCaller {
	~LateCaller() {
		AutoCap::Screen s;
		s.name = "late";
		AutoCap::Publish(s);
		(void)AutoCap::Snapshot();
	}
};
LateCaller g_late;

// Destroyed after AutoCap's statics and sleeps: the main thread of the crash test gets time to lock the (old: destroyed) mutex.
struct SlowTail {
	~SlowTail() {
		if (std::getenv("AUTOCAP_EXIT_TEST_CHILD") != nullptr) std::this_thread::sleep_for(std::chrono::milliseconds(300));
	}
};
SlowTail g_tail;

int ChildMain() {
	AutoCap::Screen s;
	s.name = "a";
	AutoCap::Publish(s);
	std::thread crash([] {
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		std::exit(37); // what the crash handler does: static destructors run on this thread, the process ends here
	});
	crash.detach();
	for (unsigned i = 0;; ++i) { // the window loop
		AutoCap::Screen n;
		n.name = (i & 1) ? "a" : "b";
		AutoCap::Publish(n);
	}
}
} // namespace

int main(int argc, char** argv) {
	if (argc > 1 && std::string(argv[1]) == "--child") return ChildMain();

	AutoCap::Screen s;
	s.name = "x";
	AutoCap::Publish(s); // constructs the function-local statics after g_late

	int pipefd[2];
	if (pipe(pipefd) != 0) return 2;
	const auto t0  = std::chrono::steady_clock::now();
	const pid_t pid = fork();
	if (pid == 0) {
		setenv("AUTOCAP_EXIT_TEST_CHILD", "1", 1);
		dup2(pipefd[1], 2);
		close(pipefd[0]);
		execl(argv[0], argv[0], "--child", static_cast<char*>(nullptr));
		_exit(127);
	}
	close(pipefd[1]);
	std::string err;
	char        buf[512];
	for (ssize_t n; (n = read(pipefd[0], buf, sizeof buf)) > 0;) err.append(buf, static_cast<size_t>(n));
	int status = 0;
	waitpid(pid, &status, 0);
	const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	if (!WIFEXITED(status) || WEXITSTATUS(status) != 37) {
		std::fprintf(stderr, "FAIL: crash-path child did not exit with the handler's code (status 0x%x)\n", status);
		return 1;
	}
	if (err.find("mutex lock failed") != std::string::npos) {
		std::fprintf(stderr, "FAIL: stderr has 'mutex lock failed'\n");
		return 1;
	}
	if (secs > 2.0) {
		std::fprintf(stderr, "FAIL: child took %.2f s\n", secs);
		return 1;
	}
	std::printf("autocap_exit_tests: all passed (child %.2f s)\n", secs);
	return 0; // g_late's destructor runs now, after AutoCap's statics: the exit test
}
