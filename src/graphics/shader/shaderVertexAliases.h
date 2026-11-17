#ifndef GRAPHICS_SHADER_SHADERVERTEXALIASES_H_
#define GRAPHICS_SHADER_SHADERVERTEXALIASES_H_

#include "graphics/shader/shader.h"

#include <bit>
#include <cstdint>
#include <span>

// nps5: shader journals and pipeline lists store a vertex input info without its PS parameter
// aliases (ShaderStoredInputInfoSize), but the source's static key holds them: the 0x7A11A5 tag, the
// mask and one word per aliased destination (BuildStageStaticKey), a field alias's field codes in
// the word's upper bits (KYTY_PS_PER_VERTEX=unpack). A replay that clears the aliases derives another
// key and never builds the alias variant, so the first draw compiles it. This reads them back.

namespace Libs::Graphics {

// Restores param_alias_mask / param_alias_source of a stored vertex input info from the static key
// recorded with it. True when the key has no aliases (mask cleared) or well-formed ones; false when
// the alias words are malformed (the info is left without aliases).
[[nodiscard]] inline bool ShaderRestoreVertexAliases(ShaderVertexInputInfo& info, std::span<const uint32_t> static_key) {
	info.param_alias_mask = 0;
	info.param_alias_source.fill(0);
	// The words before the tag, as BuildStageStaticKey writes them.
	size_t at = 8u + (info.lds_size_dwords != 0 ? 1u : 0u) + (info.user_clip_planes != 0 ? 1u : 0u);
	if (at >= static_key.size() || static_key[at] != 0x7A11A5u) return true;
	if (at + 2 > static_key.size()) return false;
	const uint32_t mask  = static_key[at + 1];
	const auto     count = static_cast<size_t>(std::popcount(mask));
	if (mask == 0 || at + 2 + count > static_key.size()) return false;
	at += 2;
	for (uint32_t destination = 0; destination < 32u; ++destination) {
		if ((mask & (1u << destination)) != 0) info.param_alias_source[destination] = static_key[at++];
	}
	info.param_alias_mask = mask;
	return true;
}

} // namespace Libs::Graphics

#endif
