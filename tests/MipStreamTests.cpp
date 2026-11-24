// nps5 mipstream: unit test of mipStream.h (table, hysteresis, view shift, policies).
#include "graphics/host_gpu/renderer/mipStream.h"
#include <cassert>
#include <cstdio>
using namespace Libs::Graphics;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)
int main() {
	setenv("KYTY_TEX_MIP_AUTO", "1", 1);
	setenv("KYTY_TEX_MIP_AUTO_HOLD", "10", 1);
	setenv("KYTY_TEX_MIP_AUTO_STATIC_FRAMES", "5", 1);
	std::atomic<uint64_t> frame {100};
	MipStream::g_frame_source = &frame;
	const auto& c = MipStream::Get();
	CHECK(c.auto_on && c.view_shift);
	const uint64_t a = 0x12345600ull, b = 0x22345600ull;
	// unseen -> start
	CHECK(MipStream::PolicySkip(a) == 2);
	MipStream::NoteSeen(a, 0);
	CHECK(MipStream::PolicySkip(a) == 2);
	frame = 106;
	CHECK(MipStream::PolicySkip(a) == 0); // static after 5 frames
	MipStream::NoteSeen(b, 0);
	MipStream::NoteStreamed(b);
	frame = 120;
	CHECK(MipStream::PolicySkip(b) == 2); // streamed, unmeasured
	MipStream::NoteMeasured(b, 3);
	CHECK(MipStream::PolicySkip(b) == 3);
	MipStream::NoteMeasured(b, 1);
	CHECK(MipStream::PolicySkip(b) == 1); // finer at once
	frame = 125;
	MipStream::NoteMeasured(b, 4);
	CHECK(MipStream::PolicySkip(b) == 1); // held
	frame = 140;
	MipStream::NoteMeasured(b, 4);
	CHECK(MipStream::PolicySkip(b) == 3); // released, capped at max 3
	// view shift: T# base 0, last 12, min_lod 2.0, address a; native unknown
	uint32_t t[8] = {};
	t[0] = static_cast<uint32_t>(a >> 8); t[1] = (2u * 256u) << 8u; t[3] = (0u << 12u) | (12u << 16u);
	CHECK(MipStream::TsharpAddress(t) == a);
	CHECK(MipStream::ViewShiftBase(t) == 2);
	MipStream::NoteNative(a, 3);
	CHECK(MipStream::ViewShiftBase(t) == 3);
	t[1] = 0; CHECK(MipStream::ViewShiftBase(t) == 3);
	MipStream::NoteNative(a, 0); CHECK(MipStream::ViewShiftBase(t) == 0);
	t[3] = (5u << 12u) | (5u << 16u); CHECK(MipStream::ViewShiftBase(t) == 5); // single level
	// seen bits survive NoteNative
	CHECK((MipStream::Read(a) & MipStream::SeenBit) != 0);
	// proxy estimate: does not override a real measurement, sets ProxyBit only
	const uint64_t pc = 0x32345600ull;
	MipStream::NoteSeen(pc, 0);
	MipStream::NoteMeasured(pc, 2, true);
	CHECK(MipStream::PolicySkip(pc) == 2);
	CHECK((MipStream::Read(pc) & MipStream::ProxyBit) != 0 && (MipStream::Read(pc) & MipStream::StreamedBit) == 0);
	MipStream::NoteMeasured(b, 0);
	MipStream::NoteMeasured(b, 3, true);
	CHECK(MipStream::PolicySkip(b) == 0);
	// table fill: many addresses
	int stored = 0;
	for (uint64_t i = 1; i < 20000; i++) { MipStream::NoteSeen(i << 12, 1); stored += MipStream::Read(i << 12) != 0; }
	CHECK(stored > 19000);
	std::printf("mipstream_test: %s (%d fails, stored %d)\n", fails ? "FAIL" : "PASS", fails, stored);
	return fails != 0;
}
