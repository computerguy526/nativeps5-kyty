#pragma once
// MSL text builder: indentation-aware line writer. No IR knowledge (see MslEmitter.h).
#include <cstdint>
#include <string>
#include <string_view>

namespace Libs::Graphics::ShaderRecompiler::Msl {

class MslBuilder {
public:
	void Line(std::string_view text);
	void Raw(std::string_view text) { out_.append(text); }
	void Open(std::string_view header = {}); // "header {"
	void Close(std::string_view tail = {});  // "} tail"
	void Blank() { out_.push_back('\n'); }
	void Comment(std::string_view text);

	[[nodiscard]] const std::string& Str() const { return out_; }
	[[nodiscard]] std::string        Take() { return std::move(out_); }
	[[nodiscard]] uint32_t           Depth() const { return depth_; }

private:
	void Indent();

	std::string out_;
	uint32_t    depth_ = 0;
};

// Literal formatting shared by the emitter and its tests.
[[nodiscard]] std::string MslU32(uint32_t value);
[[nodiscard]] std::string MslU64(uint64_t value);
[[nodiscard]] std::string
MslF32Bits(uint32_t bits); // exact: finite values as hex-free decimal, else as_type

} // namespace Libs::Graphics::ShaderRecompiler::Msl
