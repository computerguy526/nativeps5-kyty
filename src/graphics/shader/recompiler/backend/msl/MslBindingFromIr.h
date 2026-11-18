// T2 msl-mem: IR::ShaderInfo -> BindingModel (resources the program touches) + manifest.
// Used by the direct MSL emitter and, because it only reads ShaderInfo, equally by tooling that
// wants a manifest for the SPIR-V->MSL shortcut or KK variants.
#pragma once

#include "graphics/shader/recompiler/backend/msl/MslBinding.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

namespace Libs::Graphics::ShaderRecompiler::Msl {

// Resource ids equal ShaderInfo indices (buffers[i] -> b<i>, images[i] -> t<i>, samplers[i] -> s<i>).
// lds_bytes/threadgroup come from the stage input info (compute); push_bytes is the 128-byte PushData.
[[nodiscard]] Mem::BindingModel BuildBindingModel(const IR::Program& program, ShaderStageInputInfo input_info);
// Manifest JSON (schema 1, see MslBinding.cpp) for a program; grid defaults to 1 group.
[[nodiscard]] std::string BuildManifestJson(const IR::Program& program, ShaderStageInputInfo input_info,
                                            const std::string& convention = "msl-native-v1");
// LDS bytes of a program's workgroup (0 = none).
[[nodiscard]] uint32_t ProgramLdsBytes(const IR::Program& program, ShaderStageInputInfo input_info);

} // namespace Libs::Graphics::ShaderRecompiler::Msl
