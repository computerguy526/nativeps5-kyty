#ifndef EMULATOR_SRC_GRAPHICS_SHADER_PERVERTEXSHADER_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_PERVERTEXSHADER_H_

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

struct ShaderPixelInputInfo;
struct ShaderVertexInputInfo;

// nps5, KYTY_PS_PER_VERTEX=gs: raw-vertex pixel inputs without PerVertexKHR.
//
// PS5 pixel shaders read the raw vertices of a parameter with V_INTERP_MOV_F32 (P0, P10, P20).
// Astro Bot's material shaders do so for their packed "custom interpolation" parameters: 16-bit
// fields in one dword per vertex, unpacked and interpolated by the shader itself. Vulkan gives
// such reads PerVertexKHR inputs (VK_KHR_fragment_shader_barycentric); Metal, and so KosmicKrisp,
// has no per-vertex fragment inputs. A geometry stage between the vertex and the pixel shader
// supplies them instead: it passes every triangle through unchanged and adds, for each per-vertex
// parameter, the three vertices' values as outputs that are the same at all three corners. The
// pixel shader declares them as three flat inputs (ShaderPixelPerVertexSlots). The emitted
// triangle keeps the input's vertex order, so the barycentrics, the provoking vertex of the other
// flat parameters and the winding stay as they are.

// What the generated stage forwards and adds.
struct PerVertexGeometryInterface {
	struct Parameter {
		uint32_t input_location = 0; // the vertex shader's export
		// The output location; vertex 0's when raw.
		uint32_t output_location = 0;
		// Raw vertices: locations of vertices 1 and 2 (output_location is vertex 0's).
		bool     raw = false;
		uint32_t vertex1_location = 0;
		uint32_t vertex2_location = 0;
		// The vertex shader exports it (otherwise the outputs are zero).
		bool exported = false;
	};
	std::vector<Parameter> parameters;
	uint32_t               clip_distances = 0;
	uint32_t               cull_distances = 0;
	bool                   primitive_id   = false; // the pixel shader reads gl_PrimitiveID
	bool                   invariant      = false; // gl_Position is Invariant in the vertex shader
	// The vertex shader writes a built-in a geometry stage cannot forward (layer, viewport index):
	// no stage can be generated.
	bool unsupported = false;
};

// Whether the pixel shader of a pipeline needs the generated stage: it was compiled with emulated
// per-vertex inputs and has some.
[[nodiscard]] bool PerVertexGeometryNeeded(const ShaderPixelInputInfo* pixel_info);

[[nodiscard]] PerVertexGeometryInterface DescribePerVertexGeometry(
    const ShaderVertexInputInfo& vertex_info, const ShaderPixelInputInfo& pixel_info);

// The geometry stage's SPIR-V (triangles in, one triangle strip of three vertices out).
[[nodiscard]] std::vector<uint32_t> BuildPerVertexGeometryShader(
    const PerVertexGeometryInterface& interface);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_SHADER_PERVERTEXSHADER_H_
