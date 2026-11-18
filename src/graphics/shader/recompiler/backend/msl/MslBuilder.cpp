#include "graphics/shader/recompiler/backend/msl/MslBuilder.h"

#include <bit>
#include <cmath>
#include <cstdio>

namespace Libs::Graphics::ShaderRecompiler::Msl {

void MslBuilder::Indent() {
	out_.append(static_cast<size_t>(depth_) * 2u, ' ');
}

void MslBuilder::Line(std::string_view text) {
	Indent();
	out_.append(text);
	out_.push_back('\n');
}

void MslBuilder::Open(std::string_view header) {
	Indent();
	out_.append(header);
	out_.append(header.empty() ? "{\n" : " {\n");
	depth_++;
}

void MslBuilder::Close(std::string_view tail) {
	if (depth_ > 0) {
		depth_--;
	}
	Indent();
	out_.push_back('}');
	out_.append(tail);
	out_.push_back('\n');
}

void MslBuilder::Comment(std::string_view text) {
	Indent();
	out_.append("// ");
	out_.append(text);
	out_.push_back('\n');
}

std::string MslU32(uint32_t value) {
	char buffer[16];
	std::snprintf(buffer, sizeof(buffer), "%uu", value);
	return buffer;
}

std::string MslU64(uint64_t value) {
	char buffer[32];
	std::snprintf(buffer, sizeof(buffer),
	              "%llu"
	              "ul",
	              static_cast<unsigned long long>(value));
	return buffer;
}

std::string MslF32Bits(uint32_t bits) {
	const float value = std::bit_cast<float>(bits);
	if (!std::isfinite(value)) {
		return "as_type<float>(" + MslU32(bits) + ")";
	}
	char buffer[48];
	// 9 significant digits round-trip a float exactly.
	std::snprintf(buffer, sizeof(buffer), "%.9g", static_cast<double>(value));
	std::string text = buffer;
	if (text.find_first_of(".eE") == std::string::npos) {
		text += ".0";
	}
	if (bits == 0x80000000u) {
		return "-0.0f";
	}
	return text + "f";
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl
