#pragma once
// MSL backend: emits Metal Shading Language directly from Kyty's IR (the frontend and IR passes are
// shared with backend/spirv). Wave model: see work/mslnative/STATUS.md "wave64 -> simdgroup(32)".
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

#include <map>
#include <span>
#include <string>
#include <string_view>

namespace Libs::Graphics::ShaderRecompiler::Msl {

struct MslOptions {
	// 0: derive like the SPIR-V emitter (ShaderLanesPerInvocation with host_subgroup_size 32);
	// otherwise force 1 or 2 guest lanes per Metal thread.
	uint32_t lanes_override   = 0;
	bool     comments         = true;  // IR opcode comments in the output
	bool     force_dispatcher = false; // skip structured control flow (tests / A-B)
	// nps5 gpuguard (2026-11-18 GPU lockup reboot): >0 = every emitted loop checks a per-invocation
	// budget of this many iterations and breaks out when it is spent (KK compute loop-cap policy,
	// 2^18): a shader looping forever on bad input ends by itself instead of locking up the GPU
	// (killing the process does not stop committed GPU work). One budget shared by all loops of the
	// kernel; programs with barriers get a private counter per loop (threads of a threadgroup leave
	// a uniform barrier loop on the same iteration). 0 = off (default here; kyty_msl_dump and the
	// harness builds turn it on, KYTY_MSL_LOOP_CAP=<n> overrides, 0 disables).
	uint32_t loop_cap = 0;
	bool     kk_bda   = false; // compute: KosmicKrisp binding convention (MslKkBda.cpp); also env
	                           // KYTY_MSL_CONVENTION=kk-bda
};

struct MslResult {
	std::string source;
	bool        complete = false; // no unsupported opcode/stage/terminator
	uint32_t    lanes    = 1; // guest lanes per Metal thread (1: wave32, 2: wave64 as two halves)
	uint32_t    threadgroup[3] = {1, 1, 1}; // MTLSize threadsPerThreadgroup (compute)
	uint32_t    instructions   = 0;
	bool        structured     = false; // true: if/loop statements; false: dispatcher switch loop
	// Opcode name -> count, for what the emitter could not translate (coverage table).
	std::map<std::string, uint32_t> unsupported;
};

// Pure expression for a side-effect-free ALU opcode over already-formatted operands, or an empty
// string when the opcode is not an ALU/convert/compare/select op handled here. `lanes` is only
// used by lane-cross ops (never for plain ALU). Operand count must equal IR::NumArgsOf(opcode).
// Guest lanes per Metal thread for a program: 2 = wave64 as two halves, 1 = wave32 or wave64
// without cross-lane ops (one lane per thread).
uint32_t ChooseLanes(const IR::Program& program, ShaderStageInputInfo input_info,
                     uint32_t lanes_override = 0);
// Fused V_MAD_F32 (fma + host contraction allowed) vs exact mul+add; thread-local, set by
// EmitProgram from KYTY_MAD_MODE.
void                      SetFuseMad(bool fuse);
[[nodiscard]] std::string AluExpression(IR::ValueOpcode opcode, std::span<const std::string> args);

// MSL type name of an IR value type ("" for Void/opaque/unsupported).
[[nodiscard]] std::string MslTypeName(IR::Type type);

// The helper prelude every emitted source starts with (includes, fp contract(off), kyty_* helpers).
[[nodiscard]] const char* MslPrelude();

[[nodiscard]] MslResult EmitProgram(const IR::Program& program, ShaderStageInputInfo input_info,
                                    const MslOptions& options = {});

} // namespace Libs::Graphics::ShaderRecompiler::Msl
