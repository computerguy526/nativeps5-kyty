#include "graphics/shader/recompiler/ir/passes/PrecisionAnalysis.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

bool PrecisionGenerator(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::FPRecip32:
		case ValueOpcode::FPRecip64:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPRecipSqrt32:
		case ValueOpcode::FPSqrt:
		case ValueOpcode::FPLog2:
		case ValueOpcode::FPExp2:
		case ValueOpcode::FPLdexp: return true;
		default: return false;
	}
}

bool PrecisionFloatCompare(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::FPOrdEqual32:
		case ValueOpcode::FPOrdEqual64:
		case ValueOpcode::FPOrdNotEqual32:
		case ValueOpcode::FPOrdNotEqual64:
		case ValueOpcode::FPOrdLessThan32:
		case ValueOpcode::FPOrdLessThan64:
		case ValueOpcode::FPOrdGreaterThan32:
		case ValueOpcode::FPOrdGreaterThan64:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdLessThanEqual64:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual64: return true;
		default: return false;
	}
}

// The guest asks for NaN/Inf semantics explicitly.
bool PrecisionNanTest(const Inst& inst) {
	switch (inst.GetOpcode()) {
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPIsNan64:
		case ValueOpcode::FPCmpClass32:
		case ValueOpcode::FPCmpClass16:
		case ValueOpcode::FPUnordEqual32:
		case ValueOpcode::FPUnordEqual64:
		case ValueOpcode::FPUnordNotEqual32:
		case ValueOpcode::FPUnordNotEqual64:
		case ValueOpcode::FPUnordLessThan32:
		case ValueOpcode::FPUnordLessThan64:
		case ValueOpcode::FPUnordGreaterThan32:
		case ValueOpcode::FPUnordGreaterThan64:
		case ValueOpcode::FPUnordLessThanEqual32:
		case ValueOpcode::FPUnordLessThanEqual64:
		case ValueOpcode::FPUnordGreaterThanEqual32:
		case ValueOpcode::FPUnordGreaterThanEqual64: return true;
		case ValueOpcode::FPOrdEqual32:
		case ValueOpcode::FPOrdEqual64:
		case ValueOpcode::FPOrdNotEqual32:
		case ValueOpcode::FPOrdNotEqual64: {
			const auto* a = inst.Arg(0).Resolve().TryInstruction();
			return a != nullptr && a == inst.Arg(1).Resolve().TryInstruction();
		}
		default: return false;
	}
}

bool PrecisionStore(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::StoreBufferU8:
		case ValueOpcode::StoreBufferU16:
		case ValueOpcode::StoreBufferU32:
		case ValueOpcode::StoreBufferU32x2:
		case ValueOpcode::StoreBufferU32x3:
		case ValueOpcode::StoreBufferU32x4:
		case ValueOpcode::StoreAddressU8:
		case ValueOpcode::StoreAddressU16:
		case ValueOpcode::StoreAddressU32:
		case ValueOpcode::ImageWrite: return true;
		default: return false;
	}
}

// Whether a tainted (maybe infinite/NaN) operand set leaves the result tainted.
bool PrecisionCarries(const Inst& inst, const std::unordered_set<const Inst*>& tainted) {
	size_t clean = 0;
	size_t total = inst.NumArgs();
	for (size_t i = 0; i < total; i++) {
		const auto* producer = inst.Arg(i).Resolve().TryInstruction();
		if (producer == nullptr || !tainted.contains(producer)) {
			clean++;
		}
	}
	switch (inst.GetOpcode()) {
		case ValueOpcode::FPMin32:
		case ValueOpcode::FPMax32:
		case ValueOpcode::FPMin64:
		case ValueOpcode::FPMax64:
		case ValueOpcode::FPMinTri32:
		case ValueOpcode::FPMaxTri32: return clean == 0;
		case ValueOpcode::FPMedTri32: return clean < 2;
		case ValueOpcode::FPSaturate32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::PackSnorm2x16:
		case ValueOpcode::PackUnorm2x16: return false;
		default: break;
	}
	if (inst.GetType() == Type::U1 || inst.GetType() == Type::Void) {
		return false; // compares, NaN tests, stores: the value stops here
	}
	return clean < total;
}

} // namespace

void CollectPreciseSet(const Program& program, uint8_t extra,
                       std::unordered_set<const Inst*>& precise_set) {
	std::unordered_set<const Inst*> tainted;
	// Forward: to a fixed point (loop phis).
	for (bool changed = true; changed;) {
		changed = false;
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				if (tainted.contains(&inst)) {
					continue;
				}
				bool any = PrecisionGenerator(inst.GetOpcode());
				for (size_t i = 0; !any && i < inst.NumArgs(); i++) {
					const auto* producer = inst.Arg(i).Resolve().TryInstruction();
					any                  = producer != nullptr && tainted.contains(producer);
				}
				if (!any) {
					continue;
				}
				precise_set.insert(&inst); // a generator, or a consumer of a tainted value
				if (PrecisionGenerator(inst.GetOpcode()) || PrecisionCarries(inst, tainted)) {
					tainted.insert(&inst);
					changed = true;
				}
			}
		}
	}
	// Backward: the guest's own NaN/Inf tests (+ KYTY_GPU_ACCURACY_EXTRA roots).
	std::vector<const Inst*> pending;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (PrecisionNanTest(inst) || ((extra & 1u) != 0 && PrecisionFloatCompare(op)) ||
			    ((extra & 2u) != 0 && PrecisionStore(op))) {
				precise_set.insert(&inst);
				pending.push_back(&inst);
			}
		}
	}
	std::unordered_set<const Inst*> walked;
	while (!pending.empty()) {
		const auto* inst = pending.back();
		pending.pop_back();
		if (!walked.insert(inst).second) {
			continue;
		}
		for (size_t i = 0; i < inst->NumArgs(); i++) {
			const auto* producer = inst->Arg(i).Resolve().TryInstruction();
			if (producer != nullptr) {
				precise_set.insert(producer);
				pending.push_back(producer);
			}
		}
	}
	// KYTY_FAST_MATH_LOG: one line per guest shader (float instruction counts).
	if (std::getenv("KYTY_FAST_MATH_LOG") != nullptr) {
		static std::mutex         mutex;
		static std::set<uint64_t> seen;
		size_t                    fp = 0, precise = 0;
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				const auto type = inst.GetType();
				if (type == Type::F32 || type == Type::F64 || type == Type::F16) {
					fp++;
					precise += precise_set.contains(&inst) ? 1u : 0u;
				}
			}
		}
		std::lock_guard lock(mutex);
		if (seen.insert(program.shader_hash).second) {
			std::fprintf(stderr,
			             "[precision] shader 0x%016" PRIx64
			             " %s float=%zu precise=%zu tainted=%zu\n",
			             program.shader_hash, program.stage == ShaderType::Pixel ? "fs" : "cs", fp,
			             precise, tainted.size());
		}
	}
}

void CollectPositionSlice(const Program& program, bool through_shared,
                          std::unordered_set<const Inst*>& slice) {
	std::vector<const Inst*> pending;
	std::vector<const Inst*> shared_writes;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::SetAttribute) {
				const auto index = inst.Flags<ExportFlags>().index;
				if (index < program.export_info.size() &&
				    program.export_info[index].kind == ExportTargetKind::Position) {
					pending.push_back(&inst);
				}
				continue;
			}
			if (!through_shared) {
				continue;
			}
			const auto access = SharedAccessOf(inst.GetOpcode());
			if (access == SharedAccess::Write || access == SharedAccess::Atomic ||
			    access == SharedAccess::Append) {
				shared_writes.push_back(&inst);
			}
		}
	}
	bool shared_joined = !through_shared;
	while (!pending.empty()) {
		const auto* inst = pending.back();
		pending.pop_back();
		if (!shared_joined) {
			const auto access = SharedAccessOf(inst->GetOpcode());
			if (access == SharedAccess::Read || access == SharedAccess::Atomic ||
			    access == SharedAccess::Consume) {
				// What the position reads back from shared memory is what the program stored there.
				shared_joined = true;
				for (const auto* write: shared_writes) {
					if (slice.insert(write).second) {
						pending.push_back(write);
					}
				}
			}
		}
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			const auto* producer = inst->Arg(index).Resolve().TryInstruction();
			if (producer != nullptr && slice.insert(producer).second) {
				pending.push_back(producer);
			}
		}
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
