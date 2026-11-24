#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCKNOWNFILL_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCKNOWNFILL_H_

// KYTY_DCC_KNOWN_FILL_ALIASED decision (textureCache.cpp, DCC clear-metadata discovery), split out so it
// can be unit tested alone (tests/DccKnownFillTests.cpp). Pure functions of plain values.
//
// The guest's DCC fast clear is a uniform fill of the metadata bytes. When the buffer cache still holds a
// recorded fill for the whole range (nothing wrote it since), every slice's code is that byte and the clear
// can be decided without a GPU readback. If images alias the metadata bytes ("aliased"), the decision is
// taken only with KYTY_DCC_KNOWN_FILL_ALIASED=1; =verify keeps the CPU fallback and checks it against the
// record.

#include <cstdint>
#include <cstring>

namespace Libs::Graphics::DccKnownFill {

enum class Mode : int { Off = 0, Take = 1, Verify = 2 };

// "1" -> Take, "verify" -> Verify, anything else -> Off.
[[nodiscard]] inline Mode ParseMode(const char* value) noexcept {
	if (value == nullptr) {
		return Mode::Off;
	}
	return std::strcmp(value, "1") == 0        ? Mode::Take
	       : std::strcmp(value, "verify") == 0 ? Mode::Verify
	                                           : Mode::Off;
}

struct Decision {
	bool    uniform_known = false; // a record exists and all four bytes of the word are equal
	bool    take_known    = false; // decide the clear from the record, no readback
	bool    verify        = false; // aliased + uniform + verify mode: compare the readback to the record
	uint8_t known_byte    = 0;
};

// has_record: the buffer cache holds a current recorded fill for the whole range (a stale or missing record
// is has_record == false). word: the recorded 32-bit fill. unaliased: no image lives over the bytes.
[[nodiscard]] inline Decision Decide(Mode mode, bool has_record, uint32_t word, bool unaliased) noexcept {
	Decision d;
	if (!has_record) {
		return d;
	}
	d.known_byte    = static_cast<uint8_t>(word & 0xffu);
	d.uniform_known = word == static_cast<uint32_t>(d.known_byte) * 0x01010101u;
	d.take_known    = d.uniform_known && (unaliased || mode == Mode::Take);
	d.verify        = d.uniform_known && !unaliased && mode == Mode::Verify;
	return d;
}

} // namespace Libs::Graphics::DccKnownFill

#endif
