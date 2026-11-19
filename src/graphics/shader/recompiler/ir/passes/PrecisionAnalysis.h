#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_PRECISIONANALYSIS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_PRECISIONANALYSIS_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

// KYTY_GPU_ACCURACY=normal (fmsel6): selective precision, the per-instruction analogue of
// yuzu/Eden honouring Maxwell's per-instruction precise flags (GCN/RDNA has none, so the precise
// set is derived from the data flow). Fast math (Metal's no-NaN/no-Inf/no-signed-zero assumption
// plus fast:: functions) changes results only where a value is infinite or NaN; such values come
// from a few instructions (or from memory a previous pass wrote, which this set keeps finite in
// the passes that produce them). The precise set holds:
//  - forward: every instruction that may produce or carry an infinite/NaN value, starting at the
//    RCP/RSQ/SQRT/LOG/EXP/LDEXP results, until a sanitizer (min/max/min3/max3 with a finite
//    operand, med3 with two, saturate, compare, float->int, packed snorm/unorm) consumes it; the
//    sanitizer itself is precise (its NaN/Inf rule is what the guest relies on).
//  - backward: everything computing an operand of the guest's explicit NaN/Inf tests
//    (V_CMP_CLASS, V_CMP_U/O-type unordered compares, x == x / x != x), whatever the source.
//  - KYTY_GPU_ACCURACY_EXTRA=cmp / stores: also the backward slices of every float compare and of
//    the data of every buffer/address/image store.
// extra: KYTY_GPU_ACCURACY_EXTRA bits (1 cmp, 2 stores). KYTY_FAST_MATH_LOG prints one
// "[precision] shader" line per guest hash.
void CollectPreciseSet(const Program& program, uint8_t extra, std::unordered_set<const Inst*>& precise_set);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif
