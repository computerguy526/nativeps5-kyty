#ifndef GRAPHICS_HOST_GPU_RENDERER_PIPELINE_REPLAYSKIPSTATS_H_
#define GRAPHICS_HOST_GPU_RENDERER_PIPELINE_REPLAYSKIPSTATS_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <span>
#include <string>
#include <vector>

// nps5 skipfix: why a shader precompile replay entry (journal or pipeline list permutation) was
// skipped, counted per reason; one summary line at the end of the replay. Pure bookkeeping, no
// renderer dependencies (unit tested on its own).

namespace Libs::Graphics {

enum class ReplaySkipReason : uint32_t {
	SourceCode,     // no code, code size differs, user data count too large, or merged hash differs
	Specialization, // the stored specialization does not decode
	InputSize,      // stored input info size differs from this build's
	KeyStage,       // rebuilt key: logical stage differs
	KeyStatic,      // rebuilt key: static state words differ from the stored ones
	KeyCodeSize,    // rebuilt key: code size differs
	SkipDispatch,   // the program was marked skip (ray tracing etc.)
	NotPublished,   // compile ran but published nothing new (raced with another publisher / failed)
	Count,
};

inline const char* ReplaySkipReasonName(ReplaySkipReason reason) {
	switch (reason) {
		case ReplaySkipReason::SourceCode: return "source";
		case ReplaySkipReason::Specialization: return "specialization";
		case ReplaySkipReason::InputSize: return "input-size";
		case ReplaySkipReason::KeyStage: return "key-stage";
		case ReplaySkipReason::KeyStatic: return "key-static";
		case ReplaySkipReason::KeyCodeSize: return "key-code-size";
		case ReplaySkipReason::SkipDispatch: return "skip-dispatch";
		case ReplaySkipReason::NotPublished: return "not-published";
		default: return "?";
	}
}

// First index where the two words differ (the shorter length when one is a prefix of the other);
// equal spans return a.size().
inline size_t ReplayFirstKeyDiff(std::span<const uint32_t> a, std::span<const uint32_t> b) {
	const size_t n = a.size() < b.size() ? a.size() : b.size();
	for (size_t i = 0; i < n; ++i) {
		if (a[i] != b[i]) return i;
	}
	return n;
}

class ReplaySkipStats {
public:
	static constexpr size_t kReasons = static_cast<size_t>(ReplaySkipReason::Count);
	static constexpr size_t kSamples = 8;

	void Note(ReplaySkipReason reason, uint64_t source_hash, std::string detail = {}) {
		const auto index = static_cast<size_t>(reason);
		if (index >= kReasons) return;
		counts[index].fetch_add(1, std::memory_order_relaxed);
		std::lock_guard lock(mutex);
		auto&           list = samples[index];
		if (list.size() < kSamples) {
			char text[40];
			std::snprintf(text, sizeof(text), "%016llx",
			              static_cast<unsigned long long>(source_hash));
			list.push_back(detail.empty() ? std::string(text) : std::string(text) + " " + detail);
		}
	}

	// Details of a static key mismatch: stage kind, word counts and the first differing word.
	static std::string DescribeStaticKey(const char* kind, std::span<const uint32_t> stored,
	                                     std::span<const uint32_t> built) {
		const auto               at = ReplayFirstKeyDiff(stored, built);
		char                     text[160];
		const unsigned long long s = at < stored.size() ? stored[at] : 0,
		                         b = at < built.size() ? built[at] : 0;
		std::snprintf(text, sizeof(text),
		              "%s words stored %zu built %zu first diff @%zu stored %s%llx built %s%llx",
		              kind, stored.size(), built.size(), at, at < stored.size() ? "0x" : "-", s,
		              at < built.size() ? "0x" : "-", b);
		return text;
	}

	uint64_t Total() const {
		uint64_t sum = 0;
		for (const auto& c: counts)
			sum += c.load(std::memory_order_relaxed);
		return sum;
	}
	uint64_t Count(ReplaySkipReason reason) const {
		return counts[static_cast<size_t>(reason)].load(std::memory_order_relaxed);
	}

	// "Shader precompile: N skipped: key-static 560 [h1 h2 ...]; ..." (reasons with count 0
	// omitted).
	std::string Summary() {
		std::string     out;
		std::lock_guard lock(mutex);
		for (size_t i = 0; i < kReasons; ++i) {
			const auto n = counts[i].load(std::memory_order_relaxed);
			if (n == 0) continue;
			if (!out.empty()) out += "; ";
			out += std::string(ReplaySkipReasonName(static_cast<ReplaySkipReason>(i))) + " " +
			       std::to_string(n) + " [";
			for (size_t s = 0; s < samples[i].size(); ++s)
				out += (s != 0 ? " | " : "") + samples[i][s];
			out += "]";
		}
		return out;
	}

private:
	std::array<std::atomic<uint64_t>, kReasons>    counts {};
	std::mutex                                     mutex;
	std::array<std::vector<std::string>, kReasons> samples;
};

} // namespace Libs::Graphics

#endif
