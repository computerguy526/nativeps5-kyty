#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_ATOMIC64LOWERING_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_ATOMIC64LOWERING_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

// 64-bit buffer atomics on a host without shaderBufferInt64Atomics (KosmicKrisp: Metal has no
// 64-bit atomic with a result). Two guest forms need no read-modify-write and are emitted as one
// naturally aligned 64-bit access, which Apple GPUs perform as a single access (a load never
// returns the dwords of two stores; measured by AnyPS5's mesa/tests/atomics, whose design this
// follows). Any other form is named at analysis time instead of failing pipeline creation.
namespace Libs::Graphics::ShaderRecompiler::Spirv {

enum class BufferAtomic64Form {
	// A real read-modify-write; needs the device feature.
	Native,
	// Swap whose result nobody reads: an atomic 64-bit store.
	PlainStore,
	// Or/Xor/Add/Sub of zero (also a pair of zero-or-masked dwords): the word is unchanged, the
	// result is the old value: an atomic 64-bit load.
	PlainLoad,
};

namespace Detail {
// A dword that is a literal zero, or SelectU32(active, 0, x) with `active` the atomic's own
// active-lane argument (x only reaches inactive lanes, which do not execute the atomic).
[[nodiscard]] inline bool IsZeroDword(const IR::Value& value, const IR::Value& active) {
	if (value.IsImmediate()) {
		return value.GetType() == IR::Type::U32 && value.U32() == 0u;
	}
	const auto* producer = value.ResolveInstruction();
	if (producer == nullptr || producer->GetOpcode() != IR::ValueOpcode::SelectU32) {
		return false;
	}
	const auto when_true = producer->Arg(1);
	return when_true.IsImmediate() && when_true.GetType() == IR::Type::U32 && when_true.U32() == 0u &&
	       producer->Arg(0).Resolve() == active.Resolve();
}

[[nodiscard]] inline bool IsZeroU64(const IR::Value& value, const IR::Value& active) {
	if (value.IsImmediate()) {
		return value.GetType() == IR::Type::U64 && value.U64() == 0u;
	}
	const auto* producer = value.ResolveInstruction();
	if (producer == nullptr || producer->GetOpcode() != IR::ValueOpcode::CompositeConstructU64) {
		return false;
	}
	return IsZeroDword(producer->Arg(0), active) && IsZeroDword(producer->Arg(1), active);
}
} // namespace Detail

// `inst` must be a 64-bit buffer atomic (IR::BufferAccessOf == Atomic, type U64). The operands are
// laid out as the emitter reads them: [.., data, active].
[[nodiscard]] inline BufferAtomic64Form ClassifyBufferAtomic64(const IR::Inst& inst) {
	const auto opcode = inst.GetOpcode();
	const auto active = inst.Arg(inst.NumArgs() - 1);
	if (opcode == IR::ValueOpcode::BufferAtomicSwap64) {
		return inst.HasUses() ? BufferAtomic64Form::Native : BufferAtomic64Form::PlainStore;
	}
	if (opcode == IR::ValueOpcode::BufferAtomicOr64 || opcode == IR::ValueOpcode::BufferAtomicXor64 ||
	    opcode == IR::ValueOpcode::BufferAtomicIAdd64 || opcode == IR::ValueOpcode::BufferAtomicISub64) {
		return Detail::IsZeroU64(inst.Arg(inst.NumArgs() - 2), active) ? BufferAtomic64Form::PlainLoad
		                                                               : BufferAtomic64Form::Native;
	}
	return BufferAtomic64Form::Native;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif
