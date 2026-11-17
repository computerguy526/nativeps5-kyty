// nps5 kshaders: offline tests of the KosmicKrisp fallbacks that need no GPU:
//  - 64-bit buffer atomic classification (atomic64Lowering.h)
//  - minimum-LOD view range without VK_EXT_image_view_min_lod (imageViewMinLod.h)
//  - host flag of the codegen options
#include "graphics/host_gpu/renderer/image/imageViewMinLod.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/atomic64Lowering.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

using namespace Libs::Graphics::ShaderRecompiler;
using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderRecompiler::Spirv::BufferAtomic64Form;
using Libs::Graphics::ShaderRecompiler::Spirv::ClassifyBufferAtomic64;

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

struct Fixture {
	Program program;
	Fixture() {
		program.block_storage.push_back(std::make_unique<Block>());
		program.blocks.push_back(program.block_storage.back().get());
		program.block_info.push_back({.id = 0});
	}
	Value Emit(ValueOpcode opcode, std::initializer_list<Value> args) {
		return Value(&program.blocks[0]->AppendNewInst(opcode, args, 0));
	}
	// BufferAtomic*64 (resource, three address dwords, data, active); only data and active matter here.
	Value Atomic(ValueOpcode opcode, Value data, Value active) {
		const Value dword(uint32_t {0x10});
		return Emit(opcode, {dword, dword, dword, dword, data, active});
	}
};

void AtomicClassification() {
	Fixture f;
	const Value active(true);

	// Swap64 with an unread result is a store; with a reader it is a real exchange.
	const auto store = f.Atomic(ValueOpcode::BufferAtomicSwap64, Value(uint64_t {7}), active);
	Check(ClassifyBufferAtomic64(*store.Instruction()) == BufferAtomic64Form::PlainStore,
	      "swap64, result unused -> PlainStore");
	const auto xchg = f.Atomic(ValueOpcode::BufferAtomicSwap64, Value(uint64_t {7}), active);
	f.Emit(ValueOpcode::IAdd64, {xchg, Value(uint64_t {1})});
	Check(ClassifyBufferAtomic64(*xchg.Instruction()) == BufferAtomic64Form::Native,
	      "swap64, result read -> Native (refused without the feature)");

	// Or/Xor/Add/Sub of a literal zero reads the word.
	for (auto opcode: {ValueOpcode::BufferAtomicOr64, ValueOpcode::BufferAtomicXor64,
	                   ValueOpcode::BufferAtomicIAdd64, ValueOpcode::BufferAtomicISub64}) {
		const auto load = f.Atomic(opcode, Value(uint64_t {0}), active);
		Check(ClassifyBufferAtomic64(*load.Instruction()) == BufferAtomic64Form::PlainLoad,
		      "zero operand -> PlainLoad");
		const auto rmw = f.Atomic(opcode, Value(uint64_t {1}), active);
		Check(ClassifyBufferAtomic64(*rmw.Instruction()) == BufferAtomic64Form::Native,
		      "non-zero operand -> Native");
	}

	// The title shape: or64 with CompositeConstructU64(SelectU32(active, 0, x), SelectU32(active, 0, y)).
	const auto x     = f.Emit(ValueOpcode::ReadClockRealtime64, {});
	const auto lo    = f.Emit(ValueOpcode::SelectU32, {active, Value(uint32_t {0}), Value(uint32_t {5})});
	const auto hi    = f.Emit(ValueOpcode::SelectU32, {active, Value(uint32_t {0}), Value(uint32_t {6})});
	const auto pair  = f.Emit(ValueOpcode::CompositeConstructU64, {lo, hi});
	const auto shape = f.Atomic(ValueOpcode::BufferAtomicOr64, pair, active);
	(void)x;
	Check(ClassifyBufferAtomic64(*shape.Instruction()) == BufferAtomic64Form::PlainLoad,
	      "or64(CompositeConstruct(Select(active,0,_),Select(active,0,_))) -> PlainLoad");
	// A select on another condition does not prove the operand is zero.
	const auto other = f.Emit(ValueOpcode::SelectU32, {Value(false), Value(uint32_t {0}), Value(uint32_t {5})});
	const auto pair2 = f.Emit(ValueOpcode::CompositeConstructU64, {other, hi});
	const auto wrong = f.Atomic(ValueOpcode::BufferAtomicOr64, pair2, active);
	Check(ClassifyBufferAtomic64(*wrong.Instruction()) == BufferAtomic64Form::Native,
	      "select on a different condition -> Native");

	// And, min/max, cmpswap are never lowered.
	for (auto opcode: {ValueOpcode::BufferAtomicAnd64, ValueOpcode::BufferAtomicUMax64,
	                   ValueOpcode::BufferAtomicSMin64}) {
		const auto v = f.Atomic(opcode, Value(uint64_t {0}), active);
		Check(ClassifyBufferAtomic64(*v.Instruction()) == BufferAtomic64Form::Native,
		      "and/min/max -> Native");
	}
}

struct View {
	uint32_t base_level  = 0;
	uint32_t level_count = 1;
	uint32_t min_lod     = 0;
};

void MinLodView() {
	using Libs::Graphics::ImageViewOps::ApplyMinLodWithoutExtension;
	View v{.base_level = 1, .level_count = 6, .min_lod = 0};
	Check(!ApplyMinLodWithoutExtension(v, false) && v.base_level == 1 && v.level_count == 6,
	      "no clamp leaves the view alone");
	v = {1, 6, 2u * 256u};
	Check(ApplyMinLodWithoutExtension(v, false) && v.min_lod == 0 && v.base_level == 3 &&
	          v.level_count == 4,
	      "clamp of 2 levels moves the base up by 2");
	v = {0, 6, 2u * 256u + 128u};
	ApplyMinLodWithoutExtension(v, false);
	Check(v.base_level == 2 && v.level_count == 4 && v.min_lod == 0,
	      "the fraction of the clamp is dropped");
	v = {0, 3, 9u * 256u};
	ApplyMinLodWithoutExtension(v, false);
	Check(v.base_level == 2 && v.level_count == 1, "a clamp past the range keeps the last level");
	v = {0, 6, 128u};
	ApplyMinLodWithoutExtension(v, false);
	Check(v.base_level == 0 && v.level_count == 6 && v.min_lod == 0, "a sub-level clamp is dropped");
	v = {0, 6, 3u * 256u};
	ApplyMinLodWithoutExtension(v, true);
	Check(v.base_level == 0 && v.level_count == 6 && v.min_lod == 0,
	      "3D slice views keep their range");
}

void HostFlag() {
	Check(GetHostBufferInt64Atomics(), "host int64 buffer atomics default on");
	SetHostBufferInt64Atomics(false);
	Check(!GetHostBufferInt64Atomics(), "host flag can be cleared");
	SetHostBufferInt64Atomics(true);
}

} // namespace

int main() {
	AtomicClassification();
	MinLodView();
	HostFlag();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("kk_lowering_tests: all passed");
	return 0;
}
