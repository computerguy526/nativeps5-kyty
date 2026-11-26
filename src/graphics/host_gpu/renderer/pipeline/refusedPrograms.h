#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_REFUSEDPROGRAMS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_REFUSEDPROGRAMS_H_

// Programs the recompiler refuses (KYTY_REFUSED_PROGRAMS). A program that ends the process with a
// fatal error while the launch phase / prepare translates it (SpirV validation failure, an assert
// in the materialization, ...) made every later start die at the same program again, whenever a
// player's own journals held it. The precompile worker now notes the program it is translating
// (thread-local); when a fatal error ends the process (common/fatalHook.h) the failing thread
// appends that program's hash to a small text file next to the program cache. The next start skips
// it, counts it and logs the hash instead of translating it.
//   - Only a fatal error records a program: a kill, a crash of another kind or a normal exit leave
//   the file alone, so closing the
//     game during a prepare never blacklists a good program.
//   - The file is text: one `0x<16 hex digits>` per line, then a tab and the first line of the
//   fatal text (cut to 120 characters).
//     Lines that do not start with a hash are ignored. A program is removed by deleting its line.
//   - Nothing is skipped without a file (KYTY_REFUSED_PROGRAMS=0 turns the whole thing off), and a
//   clean list has no file at all.
// Header only: tests/RefusedProgramsTests.cpp needs no emulator.

#include "common/fatalHook.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class RefusedPrograms {
public:
	static constexpr size_t MaxReasonChars = 120;

	// Reads the file (a missing one is an empty list). Returns the number of programs read.
	size_t Open(const std::filesystem::path& path) {
		std::scoped_lock lock(m_mutex);
		m_path = path;
		m_map.clear();
		std::ifstream in(path);
		std::string   line;
		while (std::getline(in, line)) {
			uint64_t hash = 0;
			if (!ParseHash(line, hash)) continue;
			const auto tab = line.find('\t');
			m_map.emplace(hash, tab == std::string::npos ? std::string {} : line.substr(tab + 1));
		}
		return m_map.size();
	}

	[[nodiscard]] bool Contains(uint64_t hash) const {
		std::scoped_lock lock(m_mutex);
		return m_map.contains(hash);
	}
	[[nodiscard]] std::string Reason(uint64_t hash) const {
		std::scoped_lock lock(m_mutex);
		const auto       it = m_map.find(hash);
		return it == m_map.end() ? std::string {} : it->second;
	}
	[[nodiscard]] size_t Count() const {
		std::scoped_lock lock(m_mutex);
		return m_map.size();
	}
	[[nodiscard]] const std::filesystem::path& Path() const { return m_path; }

	// The thread starts translating `hash` / is done with it. A fatal error on this thread in
	// between records it.
	static void                   Begin(uint64_t hash) { Current() = hash; }
	static void                   End() { Current() = 0; }
	[[nodiscard]] static uint64_t InFlight() { return Current(); }

	// The failing thread's program (if any) goes to the file. Safe to call from the fatal hook: no
	// allocation beyond the line. Returns true when a line was written.
	bool RecordFatal(std::string_view text) {
		const auto hash = Current();
		if (hash == 0 || m_path.empty()) return false;
		std::string reason(text.substr(0, text.find('\n')));
		if (reason.size() > MaxReasonChars) reason.resize(MaxReasonChars);
		for (auto& c: reason)
			if (c == '\t' || c == '\r') c = ' ';
		char head[32];
		std::snprintf(head, sizeof(head), "0x%016llx\t", static_cast<unsigned long long>(hash));
		const std::string line = std::string(head) + reason + "\n";
		bool              ok   = false;
#if defined(_WIN32)
		if (FILE* f = std::fopen(m_path.string().c_str(), "ab")) {
			ok = std::fwrite(line.data(), 1, line.size(), f) == line.size();
			std::fclose(f);
		}
#else
		const int fd = ::open(m_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
		if (fd >= 0) {
			ok = ::write(fd, line.data(), line.size()) == static_cast<ssize_t>(line.size());
			::close(fd);
		}
#endif
		Current() = 0;
		return ok;
	}

	// The process-wide instance the fatal hook writes to (set by the pipeline cache; nullptr:
	// none).
	static std::atomic<RefusedPrograms*>& Active() {
		static std::atomic<RefusedPrograms*> active {nullptr};
		return active;
	}
	static void FatalHook(std::string_view text) {
		if (auto* self = Active().load(std::memory_order_acquire)) (void)self->RecordFatal(text);
	}
	// Makes this the recording instance and installs the fatal hook.
	void Install() {
		Active().store(this, std::memory_order_release);
		Common::SetFatalHook(&RefusedPrograms::FatalHook);
	}
	void Uninstall() {
		RefusedPrograms* self = this;
		Active().compare_exchange_strong(self, nullptr);
	}

	// One line of the file: `0x` + 16 hex digits (either case), then end of line or a tab / space.
	static bool ParseHash(const std::string& line, uint64_t& out) {
		if (line.size() < 18 || line[0] != '0' || (line[1] != 'x' && line[1] != 'X')) return false;
		uint64_t value = 0;
		for (size_t i = 0; i < 16; i++) {
			const char c = line[2 + i];
			const int  d = c >= '0' && c <= '9'   ? c - '0'
			               : c >= 'a' && c <= 'f' ? c - 'a' + 10
			               : c >= 'A' && c <= 'F' ? c - 'A' + 10
			                                      : -1;
			if (d < 0) return false;
			value = (value << 4) | static_cast<uint64_t>(d);
		}
		if (line.size() > 18 && line[18] != '\t' && line[18] != ' ') return false;
		out = value;
		return true;
	}

	// Where the file lives: KYTY_REFUSED_PROGRAMS (a path; "0" or empty = off), else `<program
	// cache path>.refused`, else none.
	[[nodiscard]] static std::filesystem::path ChoosePath(const char* env_value,
	                                                      const char* program_cache_path) {
		if (env_value != nullptr) {
			const std::string v = env_value;
			if (v.empty() || v == "0") return {};
			return v;
		}
		if (program_cache_path != nullptr && *program_cache_path != '\0')
			return std::string(program_cache_path) + ".refused";
		return {};
	}

private:
	static uint64_t& Current() {
		thread_local uint64_t hash = 0;
		return hash;
	}

	mutable std::mutex                        m_mutex;
	std::filesystem::path                     m_path;
	std::unordered_map<uint64_t, std::string> m_map;
};

} // namespace Libs::Graphics

#endif
