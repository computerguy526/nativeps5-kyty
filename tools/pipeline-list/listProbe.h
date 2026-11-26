// List-builder compile probe (UE5 warm lists): which listed programs does the recompiler refuse?
//
// The launch-phase compile of a list ends the whole process (EXIT / assert, exit 65) on the first program the recompiler refuses,
// so a warm list has to be made of programs that build. The probe translates every listed program through the SAME translation
// the title uses, one program (permutation) per CHILD process: a fatal EXIT, an assert, a signal or a hang kills the child only.
// The parent records the program's hash and the reason; programs with any refused permutation are dropped from the list
// (kyty_pipeline_list drop) with their pipelines, and ray-tracing programs can be dropped as well for a title that runs with ray
// tracing off. The translation itself is a callback, so tests use a fake translator that crashes for chosen hashes.
//
// Pure POSIX + std; no game data. Used by tools/spirv-dump/kytySpvDump.cpp (--probe) and tests/ListProbeTests.cpp.
#ifndef KYTY_TOOLS_PIPELINE_LIST_LISTPROBE_H_
#define KYTY_TOOLS_PIPELINE_LIST_LISTPROBE_H_

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <map>
#include <poll.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace ListProbe {

struct Unit {
	uint64_t hash = 0;  // the program (shader source hash), as the list names it
	uint32_t stage = 0; // 0 vertex, 1 pixel, 2 compute (informational)
	int      perm = 0;  // which permutation of the program (the translate callback's own index)
};

enum class Verdict { Ok, Refused, Crashed, TimedOut };

struct Outcome {
	Verdict     verdict = Verdict::Ok;
	int         code = 0;    // exit status, or the signal number for Crashed
	std::string reason;      // one line, no tabs, no `0x` + 16 hex digits
};

struct Options {
	int      jobs = 4;                  // children running at once
	int      timeout_ms = 120000;       // per child
	size_t   capture_bytes = 16 * 1024; // of the child's stderr kept
};

// The child runs this and returns its exit code (0 = the program builds). It may write to stderr (kept as the reason) and may
// terminate the process any way it likes.
using TranslateFn = std::function<int(const Unit&)>;

// One line of a child's stderr that says why: the first one naming a refusal, else the last non-empty one.
[[nodiscard]] inline std::string PickReason(const std::string& text) {
	static const char* const Marks[] = {"validation failed", "Error:", "Not implemented", "EXIT", "NOTIMPL", "assert", "Assert", "ASSERT", "requires", "Fail"};
	std::string last, first_marked;
	size_t      at = 0;
	while (at < text.size()) {
		size_t end = text.find('\n', at);
		if (end == std::string::npos) end = text.size();
		std::string line = text.substr(at, end - at);
		at = end + 1;
		while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
		if (line.empty()) continue;
		last = line;
		if (first_marked.empty())
			for (const char* m: Marks)
				if (line.find(m) != std::string::npos) { first_marked = line; break; }
	}
	return first_marked.empty() ? last : first_marked;
}

// A reason as a tsv cell: tabs and control characters become spaces, every `0x` + 16 hex digits loses its `0x` (a later
// `drop` reads each such token of a file as a program to drop; only the first column may name one), cut to 200 characters.
[[nodiscard]] inline std::string CleanReason(std::string s) {
	for (auto& c: s)
		if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) c = ' ';
	for (size_t at = s.find("0x"); at != std::string::npos; at = s.find("0x", at + 1)) {
		size_t n = 0;
		while (n < 16 && at + 2 + n < s.size() && std::strchr("0123456789abcdefABCDEF", s[at + 2 + n]) != nullptr) n++;
		if (n == 16) s.erase(at, 2);
	}
	if (s.size() > 200) s.resize(200);
	return s;
}

namespace detail {
struct Running {
	size_t                                   unit = 0;
	pid_t                                    pid = -1;
	int                                      fd = -1;
	std::string                              captured;
	std::chrono::steady_clock::time_point    deadline;
	bool                                     killed = false;
};
} // namespace detail

// Runs every unit in its own child process, `jobs` at a time. outcomes[i] belongs to units[i].
[[nodiscard]] inline std::vector<Outcome> RunUnits(const std::vector<Unit>& units, const TranslateFn& fn, const Options& options) {
	using Clock = std::chrono::steady_clock;
	std::vector<Outcome>          outcomes(units.size());
	std::vector<detail::Running>  running;
	size_t                        next = 0;
	const size_t                  jobs = options.jobs < 1 ? 1 : static_cast<size_t>(options.jobs);
	std::fflush(stdout);
	std::fflush(stderr);
	while (next < units.size() || !running.empty()) {
		while (next < units.size() && running.size() < jobs) {
			int fds[2];
			if (pipe(fds) != 0) { outcomes[next] = {Verdict::Refused, -1, "probe: pipe failed"}; next++; continue; }
			const pid_t pid = fork();
			if (pid < 0) {
				close(fds[0]); close(fds[1]);
				outcomes[next] = {Verdict::Refused, -1, "probe: fork failed"};
				next++;
				continue;
			}
			if (pid == 0) {
				close(fds[0]);
				dup2(fds[1], 2);
				dup2(fds[1], 1); // fatal reports of the title's own handlers go to the console (stdout): captured as well
				if (fds[1] > 2) close(fds[1]);
				const int rc = fn(units[next]);
				std::fflush(nullptr);
				_exit(rc & 0xff);
			}
			close(fds[1]);
			fcntl(fds[0], F_SETFL, fcntl(fds[0], F_GETFL, 0) | O_NONBLOCK);
			detail::Running r;
			r.unit = next; r.pid = pid; r.fd = fds[0];
			r.deadline = Clock::now() + std::chrono::milliseconds(options.timeout_ms);
			running.push_back(std::move(r));
			next++;
		}
		if (running.empty()) break;
		std::vector<pollfd> pfds;
		for (const auto& r: running) pfds.push_back({r.fd, POLLIN, 0});
		poll(pfds.data(), static_cast<nfds_t>(pfds.size()), 100);
		for (size_t i = 0; i < running.size();) {
			auto& r = running[i];
			char  buf[4096];
			bool  eof = false;
			for (;;) {
				const ssize_t n = read(r.fd, buf, sizeof(buf));
				if (n > 0) { if (r.captured.size() < options.capture_bytes) r.captured.append(buf, static_cast<size_t>(n)); continue; }
				if (n == 0) eof = true;
				break; // n < 0: EAGAIN or an error
			}
			if (!eof && !r.killed && Clock::now() > r.deadline) { kill(r.pid, SIGKILL); r.killed = true; }
			if (!eof && !r.killed) { i++; continue; }
			int status = 0;
			waitpid(r.pid, &status, 0);
			close(r.fd);
			Outcome o;
			const std::string why = CleanReason(PickReason(r.captured));
			if (r.killed) { o.verdict = Verdict::TimedOut; o.code = SIGKILL; o.reason = "timed out after " + std::to_string(options.timeout_ms / 1000) + " s"; }
			else if (WIFSIGNALED(status)) { o.verdict = Verdict::Crashed; o.code = WTERMSIG(status); o.reason = "signal " + std::to_string(o.code) + (why.empty() ? "" : ": " + why); }
			else if (WEXITSTATUS(status) != 0) { o.verdict = Verdict::Refused; o.code = WEXITSTATUS(status); o.reason = why.empty() ? "exit " + std::to_string(o.code) : why; }
			outcomes[r.unit] = std::move(o);
			running.erase(running.begin() + static_cast<std::ptrdiff_t>(i));
		}
	}
	return outcomes;
}

// Programs with at least one unit that did not build: hash -> the first such unit's reason (kind prefix: refused / crashed / timeout).
[[nodiscard]] inline std::map<uint64_t, std::string> Refused(const std::vector<Unit>& units, const std::vector<Outcome>& outcomes) {
	std::map<uint64_t, std::string> out;
	for (size_t i = 0; i < units.size() && i < outcomes.size(); i++) {
		if (outcomes[i].verdict == Verdict::Ok || out.count(units[i].hash) != 0) continue;
		const char* kind = outcomes[i].verdict == Verdict::Refused ? "refused" : outcomes[i].verdict == Verdict::Crashed ? "crashed" : "timeout";
		out.emplace(units[i].hash, std::string(kind) + ": " + outcomes[i].reason);
	}
	return out;
}

// A hash file `drop` reads: one `0x<hash>` TAB reason row per program, sorted by hash.
[[nodiscard]] inline std::string FormatRows(const std::map<uint64_t, std::string>& rows) {
	std::string out;
	for (const auto& kv: rows) {
		const uint64_t hash = kv.first;
		const std::string& reason = kv.second;
		char h[32];
		std::snprintf(h, sizeof(h), "0x%016llx", static_cast<unsigned long long>(hash));
		out += std::string(h) + "\t" + CleanReason(reason) + "\n";
	}
	return out;
}

// Ray-tracing programs (a title that runs with ray tracing off never runs them): rows for every hash in `rt`, with `reason`.
// A program that is both refused and ray tracing keeps its refusal reason.
inline void AddRayTracing(std::map<uint64_t, std::string>& rows, const std::vector<uint64_t>& rt, const std::string& reason) {
	for (const auto hash: rt) rows.emplace(hash, reason);
}

} // namespace ListProbe

#endif // KYTY_TOOLS_PIPELINE_LIST_LISTPROBE_H_
