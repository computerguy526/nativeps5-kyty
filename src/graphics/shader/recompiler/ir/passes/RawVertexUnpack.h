#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>

namespace Libs::Graphics {
struct ShaderPixelInputInfo;
struct ShaderParameterFieldSlot;
} // namespace Libs::Graphics

namespace Libs::Graphics::ShaderRecompiler::IR {

// KYTY_PS_PER_VERTEX=unpack (nps5 kart2): raw-vertex reads of a pixel shader without raw vertices.
//
// A V_INTERP_MOV_F32 read (GetInterpolationParameter) hands the shader one vertex's value of a
// parameter (P0), or the other two (P10, P20: raw values of a custom-interpolation parameter,
// differences to P0 otherwise). Shaders use it to interpolate packed data themselves: they take a
// 16-bit word of each vertex's dword, convert it to float and compute
//     f(P0) + i * (f(P1) - f(P0)) + j * (f(P2) - f(P0))            (f: convert, scale, clamp)
// with the hardware's I/J. Without raw vertices on the host (Metal), this pass gives the three
// reads one value each, such that the shader's formula yields the host-interpolated field:
//   - a word converted to float (ConvertF32S32/U32 of a BitFieldS/UExtract of the read), or widened
//     as a half float (also with its sign flipped), becomes a read of that field, interpolated by
//     the host as a float (GetParameterField): exact, the three vertices agree and the shader's i
//     and j terms vanish;
//   - any other use of the read gets the dword put together from the rounded fields;
//   - a custom parameter that is only read at P0 is a flat copy of the dword (Raw field);
//   - P10/P20 of an ordinary parameter are zero (P0 is the interpolated value);
//   - a packed half (GetAttributeF16) of a smooth parameter is its field.
// The fields come from the vertex shader through spare locations (ShaderPixelFieldSlots,
// ShaderLinkVertexPixelParameters); ShaderInfoCollection lays them out.
struct RawVertexUnpackStats {
	uint32_t raw_reads      = 0; // custom-parameter reads replaced
	uint32_t float_fields   = 0; // word-to-float conversions that became field reads
	uint32_t rounded_reads  = 0; // reads rebuilt from rounded fields (the word or dword used as bits)
	uint32_t stale_reads    = 0; // reads only kept as the old value of an overwritten register
	uint32_t flat_reads     = 0; // P0-only custom reads (Raw)
	uint32_t zero_deltas    = 0; // P10/P20 of an ordinary parameter
	uint32_t half_reads     = 0; // packed halves
	uint32_t signed_fields  = 0; // of float_fields
	uint32_t unsigned_fields = 0;
	uint32_t half_fields     = 0;

	[[nodiscard]] bool Any() const {
		return raw_reads != 0 || zero_deltas != 0 || half_reads != 0;
	}
};

RawVertexUnpackStats UnpackRawVertexParameters(Program& program, const ShaderPixelInputInfo& pixel);
// What the pass did in this thread's last call (tools and tests: the pass runs inside
// CompileProgram).
[[nodiscard]] const RawVertexUnpackStats& LastRawVertexUnpackStats();

// The vertex side: the export data (U32x4, the dwords of a parameter) of a field alias, built
// before `at`: component c holds the field slot.codes[c] of the source dwords as float bits (S16,
// U16: the word converted; F16: the half widened; Raw: the dword), zero without a field.
Value EmitParameterFieldExport(Block& block, Block::iterator at, Value data,
                               const ShaderParameterFieldSlot& slot);

} // namespace Libs::Graphics::ShaderRecompiler::IR
