// precise (2026-11-27): MadMode::Position's contract, pinned on the SPIR-V the back end emits.
//
// Why: the v1.4 base (a3cb91f85) left three of ASTRO's five mesh programs with NO NoContraction decoration and every MAD as a
// fused FMA (v1.2.3: 2,364 decorations in the largest). A mesh program stores its vertices to LDS and exports what it loads
// back; the position slice walked operands only, never through LDS, and before the v1.4 base it reached the arithmetic by
// accident (through the stale arm of an EXEC merge that the base now removes). Nothing failed: the keys of the Metal rows
// moved and positions were no longer provably exact. These tests make that a failure:
//   1. contract: synthetic guest programs (tests/PrecisionCorpus.h, no game data) where a MAD / lerp that feeds the position
//      MUST be FMul + FAdd (FSub + FMul + FAdd) with NoContraction and never an FMA or FMix, in a vertex program, a vertex
//      program whose position goes through LDS and a mesh program (position through LDS, the NGG shape); the value that feeds
//      only a parameter stays fused. KYTY_POSITION_SLICE_LDS=0 must reproduce the regression (negative control).
//   2. census: NoContraction / Invariant / FAdd / FSub / FMul / FMA / FMix / uncontract-protected ("relaxed") counts per program
//      and MadMode, compared with tests/data/precision_census.golden. ANY change fails until the golden is updated on purpose:
//          precision_spirv_tests --update      (then review the diff of the golden and say why in the commit)
//   3. IR: IR::CollectPositionSlice with and without the shared-memory hop; the switch is in the codegen fingerprint.
// Same sources as the dylib plus a main: builds and runs on arm64 macOS (no emulator, no GPU).
#include "PrecisionCorpus.h"
#include "graphics/shader/recompiler/CodegenFingerprint.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/PrecisionAnalysis.h"
#include "graphics/shader/shader.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#ifndef PRECISION_GOLDEN
#define PRECISION_GOLDEN "tests/data/precision_census.golden"
#endif

using namespace Libs::Graphics;
namespace SR = Libs::Graphics::ShaderRecompiler;
namespace PC = PrecisionCorpus;

namespace {
int g_failures = 0;
void Expect(bool condition, const std::string& what) {
	if (!condition) {
		std::printf("PrecisionSpirvTests: failed: %s\n", what.c_str());
		g_failures++;
	}
}

struct Census {
	uint32_t no_contraction = 0, invariant = 0, fadd = 0, fsub = 0, fmul = 0, fdiv = 0, fma = 0, fmix = 0;
	// Float arithmetic results the host compiler may contract (no NoContraction): what KosmicKrisp prints in
	// math_mode(relaxed) blocks.
	uint32_t relaxed = 0;
	const char* stage = "?";
	[[nodiscard]] std::string Text() const {
		char line[256];
		std::snprintf(line, sizeof(line), "stage=%s nocontraction=%u invariant=%u fadd=%u fsub=%u fmul=%u fdiv=%u fma=%u fmix=%u relaxed=%u",
		              stage, no_contraction, invariant, fadd, fsub, fmul, fdiv, fma, fmix, relaxed);
		return line;
	}
};

// Counts straight from the module words (SPIR-V 1.x: OpDecorate 71, OpMemberDecorate 72, OpExtInst 12, OpEntryPoint 15;
// NoContraction 42, Invariant 18; GLSL.std.450 FMix 46, Fma 50).
Census Count(const std::vector<uint32_t>& spirv) {
	Census                       c;
	std::unordered_set<uint32_t> exact, arith;
	for (size_t at = 5; at < spirv.size();) {
		const uint32_t count = spirv[at] >> 16u, op = spirv[at] & 0xffffu;
		if (count == 0 || at + count > spirv.size()) break;
		const uint32_t* w = &spirv[at];
		switch (op) {
			case 15:
				c.stage = w[1] == 0 ? "vs" : w[1] == 4 ? "fs" : w[1] == 5 ? "cs" : (w[1] == 5365 || w[1] == 5268) ? "mesh" : "other";
				break;
			case 71:
				if (w[2] == 42) { c.no_contraction++; exact.insert(w[1]); }
				if (w[2] == 18) c.invariant++;
				break;
			case 72:
				if (w[3] == 18) c.invariant++;
				break;
			case 129: c.fadd++; arith.insert(w[2]); break;
			case 131: c.fsub++; arith.insert(w[2]); break;
			case 133: c.fmul++; arith.insert(w[2]); break;
			case 136: c.fdiv++; arith.insert(w[2]); break;
			case 12:
				if (count >= 5 && w[4] == 50) { c.fma++; arith.insert(w[2]); }
				if (count >= 5 && w[4] == 46) { c.fmix++; arith.insert(w[2]); }
				break;
			default: break;
		}
		at += count;
	}
	for (const auto id: arith) c.relaxed += exact.contains(id) ? 0u : 1u;
	return c;
}

enum class Stage { Vertex, VertexLds, Mesh, Pixel, Compute };

struct Program {
	const char* name;
	Stage       stage;
	PC::Code    code;
};

void FillMesh(ShaderVertexInputInfo& vertex) {
	auto& mesh = vertex.mesh;
	mesh.threads_num[0] = 32; mesh.threads_num[1] = 1; mesh.threads_num[2] = 1;
	mesh.wave_size            = 32;
	mesh.host_subgroup_size   = 32;
	mesh.lds_size_dwords      = 64;
	mesh.input_primitive      = 4; // kTriList
	mesh.primitives_per_group = 8;
	mesh.vertices_per_group   = 24;
	mesh.max_vertices         = 24;
	mesh.max_primitives       = 8;
	mesh.passes               = 1;
	vertex.logical_stage      = ShaderType::Mesh;
	vertex.wave_size          = 32;
}

struct Built {
	std::vector<uint32_t> spirv;
	size_t                slice_mads = 0; // FPMad32 instructions in IR::CollectPositionSlice (hop as the options say)
	size_t                slice_mads_no_hop = 0;
};

Built Build(const Program& program, SR::MadMode mode, bool lds_hop) {
	auto options               = SR::GetCodegenOptions();
	options.mad_mode           = mode;
	options.position_slice_lds = lds_hop;
	options.fast_math          = 0;
	options.gpu_accuracy       = 0;
	SR::SetCodegenOptions(options);
	std::array<uint32_t, 64> user_data {};
	ShaderVertexInputInfo    vertex;
	ShaderPixelInputInfo     pixel;
	ShaderComputeInputInfo   compute {};
	SR::CompileOptions       copt;
	copt.shader_hash = 0x5052454349534500ull + static_cast<uint64_t>(program.stage);
	switch (program.stage) {
		case Stage::Vertex:
			copt.stage             = ShaderType::Vertex;
			copt.input_info.vertex = &vertex;
			break;
		case Stage::VertexLds:
			copt.stage             = ShaderType::Vertex;
			vertex.lds_size_dwords = 64;
			copt.input_info.vertex = &vertex;
			break;
		case Stage::Mesh:
			copt.stage     = ShaderType::Mesh;
			copt.wave_size = 32;
			FillMesh(vertex);
			copt.input_info.vertex = &vertex;
			break;
		case Stage::Pixel:
			copt.stage            = ShaderType::Pixel;
			copt.input_info.pixel = &pixel;
			break;
		case Stage::Compute:
			user_data[3]  = 3u << 28u; // as OvguardSpirvTests
			user_data[50] = 1u << 20u;
			user_data[51] = 3u << 28u;
			copt.stage                 = ShaderType::Compute;
			compute.host_subgroup_size = 64;
			compute.lds_size_dwords    = 64;
			compute.threads_num[0]     = 64; compute.threads_num[1] = 1; compute.threads_num[2] = 1;
			copt.input_info.compute    = &compute;
			break;
	}
	copt.user_data  = user_data;
	auto translated = SR::TranslateProgram(program.code, copt);
	Built built;
	if (translated.skip_dispatch) {
		Expect(false, std::string(program.name) + ": the recompiler skipped the program (" + translated.skip_reason + ")");
		return built;
	}
	const auto mads = [&](bool hop) {
		std::unordered_set<const SR::IR::Inst*> slice;
		SR::IR::CollectPositionSlice(translated.program, hop, slice);
		size_t n = 0;
		for (const auto* inst: slice) n += inst->GetOpcode() == SR::IR::ValueOpcode::FPMad32 ? 1u : 0u;
		return n;
	};
	built.slice_mads        = mads(lds_hop);
	built.slice_mads_no_hop = mads(false);
	SR::IR::ResourceSpecialization specialization;
	built.spirv = SR::CompileProgram(std::move(translated), copt, specialization).spirv;
	return built;
}

const char* ModeName(SR::MadMode mode) {
	return mode == SR::MadMode::Exact ? "exact" : mode == SR::MadMode::Position ? "position" : "fused";
}

std::string ReadFile(const char* path) {
	std::ifstream in(path, std::ios::binary);
	std::stringstream text;
	text << in.rdbuf();
	return text.str();
}
} // namespace

int main(int argc, char** argv) {
	const bool  update = argc > 1 && std::strcmp(argv[1], "--update") == 0;
	const char* golden = argc > 2 ? argv[2] : (argc > 1 && !update ? argv[1] : PRECISION_GOLDEN);
	const auto  saved  = SR::GetCodegenOptions();

	const std::vector<Program> corpus = {
	    {"vs_mad", Stage::Vertex, PC::DirectMad()},       {"vs_lerp", Stage::Vertex, PC::DirectLerp()},
	    {"vs_lds_mad", Stage::VertexLds, PC::LdsMad()},   {"mesh_lds_mad", Stage::Mesh, PC::LdsMad()},
	    {"mesh_mad", Stage::Mesh, PC::DirectMad()},       {"mesh_lerp", Stage::Mesh, PC::DirectLerp()},
	    {"fs_mad", Stage::Pixel, PC::PixelMad()},         {"cs_lds_mad", Stage::Compute, PC::ComputeLdsMad()},
	};

	// 2. census (also the data of the contract checks below)
	std::string census = "# precision census of the synthetic corpus (tests/PrecisionCorpus.h), written by precision_spirv_tests --update.\n"
	                     "# A change here is a change of what the host compiler may contract. Update on purpose only, and say why.\n";
	const auto row = [&](const Program& program, SR::MadMode mode, bool hop) {
		const auto built = Build(program, mode, hop);
		Expect(!built.spirv.empty(), std::string(program.name) + ": SPIR-V emitted");
		const auto c = Count(built.spirv);
		census += std::string(program.name) + " " + ModeName(mode) + (hop ? "" : " lds_hop_off") + " " + c.Text() + "\n";
		return std::pair {c, built};
	};
	for (const auto& program: corpus) {
		const auto name  = std::string(program.name);
		const bool gfx   = program.stage == Stage::Vertex || program.stage == Stage::VertexLds || program.stage == Stage::Mesh;
		const bool lds   = program.stage == Stage::VertexLds || (program.stage == Stage::Mesh && name == "mesh_lds_mad");
		const bool lerp  = name.find("lerp") != std::string::npos;
		const auto exact = row(program, SR::MadMode::Exact, true).first;
		const auto [position, built] = row(program, SR::MadMode::Position, true);
		const auto fused             = row(program, SR::MadMode::Fused, true).first;

		// 1. contract
		Expect(exact.fma == 0 && exact.fmix == 0 && exact.relaxed == 0, name + " exact: no FMA, no FMix, every float op NoContraction (" + exact.Text() + ")");
		Expect(fused.no_contraction == 0, name + " fused: no NoContraction (" + fused.Text() + ")");
		Expect(position.fmix == 0, name + " position: never an FMix (" + position.Text() + ")");
		if (gfx && !lerp) {
			// the position MAD is FMul + FAdd, both NoContraction; the parameter MAD is the one fused FMA
			Expect(position.fma == 1 && position.fmul == 1 && position.fadd == 1 && position.no_contraction == 2 && position.relaxed == 1,
			       name + " position: the MAD feeding the position is FMul + FAdd with NoContraction, the parameter MAD an FMA (" + position.Text() + ")");
			Expect(fused.fma == 2, name + " fused: both MADs fused (" + fused.Text() + ")");
			Expect(built.slice_mads == 1, name + ": exactly the position MAD is in IR::CollectPositionSlice");
		}
		if (gfx && lerp) {
			Expect(position.fsub == 2 && position.fmul == 2 && position.fadd == 2 && position.fma == 0 && position.no_contraction == 3 && position.relaxed == 3,
			       name + " position: the lerp feeding the position stays FSub + FMul + FAdd, each NoContraction; the parameter lerp undecorated (" + position.Text() + ")");
		}
		if (program.stage == Stage::Vertex) {
			Expect(position.invariant == 1 && exact.invariant == 1 && fused.invariant == 0, name + ": Invariant position unless fused");
		}
		if (lds) {
			// negative control = the v1.4.0-v1.4.3 emission: without the LDS hop the position MAD is not found
			const auto [off, off_built] = row(program, SR::MadMode::Position, false);
			Expect(off.no_contraction == 0 && off.fma == 2, name + " position, KYTY_POSITION_SLICE_LDS=0: the regression reproduces (no NoContraction, both MADs FMA) (" + off.Text() + ")");
			Expect(built.slice_mads_no_hop == 0 && off_built.slice_mads == 0, name + ": without the hop the slice misses the MAD behind LDS");
		}
		if (!gfx) {
			// no position in a pixel / compute program: MadMode::Position fuses, the LDS hop must not pull compute stores in
			Expect(position.no_contraction == 0 && position.fma == 1 && built.slice_mads == 0, name + " position: no position slice outside the vertex / mesh stages (" + position.Text() + ")");
		}
	}

	// 3. the switch is part of the cache identity
	{
		auto on                = saved;
		on.position_slice_lds  = true;
		auto off               = saved;
		off.position_slice_lds = false;
		SR::SetCodegenOptions(on);
		const auto fp_on = SR::CodegenFingerprint();
		SR::SetCodegenOptions(off);
		const auto fp_off = SR::CodegenFingerprint();
		Expect(fp_on != fp_off, "KYTY_POSITION_SLICE_LDS=0 changes the codegen fingerprint (its programs never share a cache with the default)");
		Expect(saved.position_slice_lds, "the LDS hop is on by default");
		Expect(saved.mad_mode == SR::MadMode::Position, "MadMode::Position is the default");
	}
	SR::SetCodegenOptions(saved);

	if (update) {
		std::ofstream out(golden, std::ios::binary | std::ios::trunc);
		out << census;
		std::printf("PrecisionSpirvTests: wrote %s (review the diff; the commit must say why the census changed)\n", golden);
	} else {
		const auto want = ReadFile(golden);
		if (want != census) {
			g_failures++;
			std::printf("PrecisionSpirvTests: failed: precision census differs from %s\n", golden);
			std::istringstream a(want), b(census);
			std::set<std::string> old_lines, new_lines;
			for (std::string line; std::getline(a, line);) old_lines.insert(line);
			for (std::string line; std::getline(b, line);) new_lines.insert(line);
			for (const auto& line: old_lines) if (!new_lines.contains(line)) std::printf("  - %s\n", line.c_str());
			for (const auto& line: new_lines) if (!old_lines.contains(line)) std::printf("  + %s\n", line.c_str());
			std::printf("  intended? run `precision_spirv_tests --update`, review the golden's diff and explain it in the commit.\n");
		}
	}
	if (g_failures == 0) {
		std::printf("PrecisionSpirvTests: ok (%zu programs x 3 modes, census %s)\n", corpus.size(), update ? "written" : "equal to the golden");
	}
	return g_failures == 0 ? 0 : 1;
}
