// T2 msl-mem: shared plumbing for the memory/image/wave/binding MSL emitters.
// Every emitter is pure text-in/text-out: callers pass MSL expression strings for the operands and
// get statements appended to a MemOut plus the name of the result expression. No dependency on
// MslBuilder, so T1's emitter can adopt it by forwarding its operand expressions.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

struct MemOut {
	std::string code;      // statements, '\n' terminated
	uint32_t    next_tmp = 0;
	std::string prefix   = "m";
	std::string error;     // non-empty: unsupported construct (caller should fail the shader)

	std::string Tmp() { return prefix + std::to_string(next_tmp++); }
	void        Line(const std::string& s) { code += "    " + s + "\n"; }
	// Declares `type name = expr;` and returns name.
	std::string Def(const char* type, const std::string& expr) {
		auto n = Tmp();
		Line(std::string(type) + " " + n + " = " + expr + ";");
		return n;
	}
	bool Fail(const std::string& what) {
		if (error.empty()) error = what;
		return false;
	}
};

std::string U(uint32_t v);          // "123u"
std::string Hex(uint32_t v);        // "0x7bu"
std::string F32Lit(float v);        // exact float literal

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
