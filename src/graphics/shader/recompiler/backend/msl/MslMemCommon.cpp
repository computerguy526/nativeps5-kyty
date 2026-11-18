#include "graphics/shader/recompiler/backend/msl/MslMemCommon.h"

#include <cstdio>

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

std::string U(uint32_t v) { return std::to_string(v) + "u"; }
std::string Hex(uint32_t v) {
	char b[32];
	std::snprintf(b, sizeof b, "0x%xu", v);
	return b;
}
std::string F32Lit(float v) {
	char b[64];
	std::snprintf(b, sizeof b, "%.9g", static_cast<double>(v));
	std::string s = b;
	if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
	return s + "f";
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
