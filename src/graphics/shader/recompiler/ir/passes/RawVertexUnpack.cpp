#include "graphics/shader/recompiler/ir/passes/RawVertexUnpack.h"

#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

using Kind = ShaderParameterFieldKind;

// A 16-bit word of the read: BitFieldS/UExtract(read, 0 or 16, 16).
bool IsWordExtract(const Inst& inst, size_t operand, uint32_t& word, bool& sign) {
	const auto opcode = inst.GetOpcode();
	if ((opcode != ValueOpcode::BitFieldSExtract && opcode != ValueOpcode::BitFieldUExtract) ||
	    operand != 0 || !inst.Arg(1).IsImmediate() || !inst.Arg(2).IsImmediate() ||
	    inst.Arg(2).U32() != 16u || (inst.Arg(1).U32() != 0u && inst.Arg(1).U32() != 16u)) {
		return false;
	}
	word = inst.Arg(1).U32() / 16u;
	sign = opcode == ValueOpcode::BitFieldSExtract;
	return true;
}

// A use of an extracted word that is the word's number as a float: what the field carries.
struct WordFloat {
	Inst* result = nullptr; // the instruction whose value the field (negated: its negation) replaces
	Kind  kind   = Kind::None;
	bool  negate = false;
};

// The single user of `inst` when it is `opcode` (taking `inst` as its first argument).
Inst* OnlyUser(const Inst& inst, ValueOpcode opcode) {
	if (inst.UseCount() != 1) {
		return nullptr;
	}
	const auto& use = inst.Uses().front();
	return use.user->GetOpcode() == opcode && use.operand == 0 ? use.user : nullptr;
}

// The half float of a 16-bit value: ConvertU16U32 -> BitCastF16U16 -> ConvertF32F16.
Inst* HalfFloat(Inst& narrow) {
	if (narrow.GetOpcode() != ValueOpcode::ConvertU16U32) {
		return nullptr;
	}
	auto* half = OnlyUser(narrow, ValueOpcode::BitCastF16U16);
	return half == nullptr ? nullptr : OnlyUser(*half, ValueOpcode::ConvertF32F16);
}

// `user` uses an extracted word (sign: sign-extended):
//  - ConvertF32S32 / ConvertF32U32: the signed or unsigned word as a float (a zero-extended word
//    is the same number either way);
//  - the word as a half float, also with its sign bit flipped first (BitwiseXor32 0x8000).
WordFloat ClassifyWordUse(Inst& user, size_t operand, bool sign) {
	if (operand != 0) {
		return {};
	}
	switch (user.GetOpcode()) {
		case ValueOpcode::ConvertF32S32: return {&user, sign ? Kind::S16 : Kind::U16, false};
		case ValueOpcode::ConvertF32U32:
			return sign ? WordFloat {} : WordFloat {&user, Kind::U16, false};
		case ValueOpcode::ConvertU16U32:
			if (auto* result = HalfFloat(user); result != nullptr) {
				return {result, Kind::F16, false};
			}
			return {};
		case ValueOpcode::BitwiseXor32:
			if (user.Arg(1).IsImmediate() && user.Arg(1).U32() == 0x8000u) {
				if (auto* narrow = OnlyUser(user, ValueOpcode::ConvertU16U32); narrow != nullptr) {
					if (auto* result = HalfFloat(*narrow); result != nullptr) {
						return {result, Kind::F16, true};
					}
				}
			}
			return {};
		default: return {};
	}
}

// Whether anything still consumes the instruction's value (through any chain of users).
bool Consumed(const Inst& inst, std::vector<const Inst*>& visited) {
	for (const auto& use: inst.Uses()) {
		if (std::ranges::find(visited, use.user) != visited.end()) {
			continue;
		}
		visited.push_back(use.user);
		if (use.user->MayHaveSideEffects() || Consumed(*use.user, visited)) {
			return true;
		}
	}
	return false;
}

} // namespace

namespace {
thread_local RawVertexUnpackStats g_last_stats;
} // namespace

const RawVertexUnpackStats& LastRawVertexUnpackStats() {
	return g_last_stats;
}

RawVertexUnpackStats UnpackRawVertexParameters(Program& program, const ShaderPixelInputInfo& pixel) {
	auto& stats = g_last_stats;
	stats       = {};
	if (program.stage != ShaderType::Pixel) {
		return stats;
	}
	// Parameters whose P10 or P20 is read: the shader interpolates them itself.
	std::array<bool, 32> interpolated {};
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::GetInterpolationParameter && inst.Arg(2).U32() < 2u) {
				interpolated[inst.Arg(0).U32() & 31u] = true;
			}
		}
	}
	const auto field = [&](Block& block, Block::iterator at, uint32_t attr, Kind kind, uint32_t chan,
	                       uint32_t word) {
		return Value(&*block.PrependNewInst(
		    at, ValueOpcode::GetParameterField,
		    {Value(attr), Value(ShaderParameterFieldCode(kind, chan, word))}));
	};
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (inst.GetOpcode() == ValueOpcode::GetAttributeF16) {
				const auto attr = inst.Arg(0).U32();
				if (!ShaderPixelParameterIsFlat(pixel, attr)) {
					const auto value =
					    field(*block, it, attr, Kind::F16, inst.Arg(1).U32(), inst.Arg(2).U32() & 1u);
					inst.ReplaceUsesWith(value);
					stats.half_reads++;
				}
				continue;
			}
			if (inst.GetOpcode() != ValueOpcode::GetInterpolationParameter) {
				continue;
			}
			const auto attr = inst.Arg(0).U32();
			const auto chan = inst.Arg(1).U32();
			const auto mode = inst.Arg(2).U32();
			if (!ShaderPixelParameterIsCustom(pixel, attr)) {
				// The differences of a host-interpolated (or flat) parameter are zero; P0 stays (the
				// emitter reads the input).
				if (mode < 2u) {
					inst.ReplaceUsesWith(Value(uint32_t {0}));
					stats.zero_deltas++;
				}
				continue;
			}
			stats.raw_reads++;
			if (!interpolated[attr & 31u]) {
				const auto raw  = field(*block, it, attr, Kind::Raw, chan, 0);
				const auto bits = Value(&*block->PrependNewInst(it, ValueOpcode::BitCastU32F32, {raw}));
				inst.ReplaceUsesWith(bits);
				stats.flat_reads++;
				continue;
			}
			// Words used as floats become field reads.
			std::array<Kind, 2> word_kind {Kind::None, Kind::None};
			bool                integer_use = false;
			const auto          uses = inst.Uses(); // copy: the lists change below
			for (const auto& use: uses) {
				uint32_t word = 0;
				bool     sign = false;
				if (!IsWordExtract(*use.user, use.operand, word, sign)) {
					continue;
				}
				const auto extract_uses = use.user->Uses();
				for (const auto& extract_use: extract_uses) {
					const auto found = ClassifyWordUse(*extract_use.user, extract_use.operand, sign);
					if (found.kind == Kind::None) {
						// The word is used as an integer: rebuilt below, in this signedness.
						integer_use = true;
						if (word_kind[word] == Kind::None) {
							word_kind[word] = sign ? Kind::S16 : Kind::U16;
						}
						continue;
					}
					auto value = field(*block, it, attr, found.kind, chan, word);
					if (found.negate) {
						value = Value(&*block->PrependNewInst(it, ValueOpcode::FPNeg32, {value}));
					}
					found.result->ReplaceUsesWith(value);
					stats.float_fields++;
					(found.kind == Kind::S16   ? stats.signed_fields
					 : found.kind == Kind::U16 ? stats.unsigned_fields
					                           : stats.half_fields)++;
					if (word_kind[word] == Kind::None) {
						word_kind[word] = found.kind;
					}
				}
			}
			// What still uses the read (or a word of it as an integer) gets the dword put together
			// from the fields, rounded: the same for the three vertices. A read that is only kept as
			// the old value of an overwritten register (the false side of a SelectU32) takes the
			// words whose kind is known; any other use takes both (signed by default).
			std::vector<const Inst*> visited;
			if (!Consumed(inst, visited)) {
				continue;
			}
			// Uses of the dword itself: as the old value behind a select, or anything else.
			bool other_use = false;
			for (const auto& use: inst.Uses()) {
				uint32_t word = 0;
				bool     sign = false;
				visited.clear();
				if (IsWordExtract(*use.user, use.operand, word, sign) ||
				    (!use.user->MayHaveSideEffects() && !Consumed(*use.user, visited))) {
					continue;
				}
				other_use = other_use || use.user->GetOpcode() != ValueOpcode::SelectU32 || use.operand != 2;
			}
			const bool stale_only = !other_use;
			Value bits(uint32_t {0});
			for (uint32_t word = 0; word < 2u; word++) {
				auto kind = word_kind[word];
				if (kind == Kind::None) {
					if (stale_only) {
						continue;
					}
					kind = Kind::S16;
				}
				const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
					return Value(&*block->PrependNewInst(it, opcode, args));
				};
				const auto value = field(*block, it, attr, kind, chan, word);
				Value      low;
				if (kind == Kind::F16) {
					low = emit(ValueOpcode::ConvertU32U16,
					           {emit(ValueOpcode::BitCastU16F16, {emit(ValueOpcode::ConvertF16F32, {value})})});
				} else {
					low = emit(ValueOpcode::BitwiseAnd32,
					           {emit(ValueOpcode::ConvertS32F32, {emit(ValueOpcode::FPRoundEven32, {value})}),
					            Value(uint32_t {0xffffu})});
				}
				if (word == 1u) {
					low = emit(ValueOpcode::ShiftLeftLogical32, {low, Value(uint32_t {16})});
				}
				bits = bits.IsImmediate() ? low : emit(ValueOpcode::BitwiseOr32, {bits, low});
			}
			(other_use || integer_use ? stats.rounded_reads : stats.stale_reads)++;
			inst.ReplaceUsesWith(bits);
		}
	}
	return stats;
}

Value EmitParameterFieldExport(Block& block, Block::iterator at, Value data,
                               const ShaderParameterFieldSlot& slot) {
	const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
		return Value(&*block.PrependNewInst(at, opcode, args));
	};
	std::array<Value, 4> components {Value(uint32_t {0}), Value(uint32_t {0}), Value(uint32_t {0}),
	                                 Value(uint32_t {0})};
	for (uint32_t component = 0; component < 4u; component++) {
		const auto code = slot.codes[component];
		const auto kind = ShaderParameterFieldCodeKind(code);
		if (kind == Kind::None) {
			continue;
		}
		const auto dword = emit(ValueOpcode::CompositeExtractU32x4,
		                        {data, Value(ShaderParameterFieldCodeChannel(code))});
		const auto shift = Value(ShaderParameterFieldCodeWord(code) * 16u);
		switch (kind) {
			case Kind::S16:
				components[component] = emit(
				    ValueOpcode::BitCastU32F32,
				    {emit(ValueOpcode::ConvertF32S32,
				          {emit(ValueOpcode::BitFieldSExtract, {dword, shift, Value(uint32_t {16})})})});
				break;
			case Kind::U16:
				components[component] = emit(
				    ValueOpcode::BitCastU32F32,
				    {emit(ValueOpcode::ConvertF32U32,
				          {emit(ValueOpcode::BitFieldUExtract, {dword, shift, Value(uint32_t {16})})})});
				break;
			case Kind::F16: {
				const auto word = emit(ValueOpcode::BitFieldUExtract, {dword, shift, Value(uint32_t {16})});
				const auto half = emit(ValueOpcode::BitCastF16U16, {emit(ValueOpcode::ConvertU16U32, {word})});
				components[component] =
				    emit(ValueOpcode::BitCastU32F32, {emit(ValueOpcode::ConvertF32F16, {half})});
				break;
			}
			case Kind::Raw: components[component] = dword; break;
			default: break;
		}
	}
	return emit(ValueOpcode::CompositeConstructU32x4,
	            {components[0], components[1], components[2], components[3]});
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
