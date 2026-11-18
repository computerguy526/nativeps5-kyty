#include "graphics/shader/recompiler/backend/msl/MslWaveOps.h"

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

namespace {

// Reads `src` from guest lane `target` (uint expr); lanes whose source is inactive read 0.
std::string ShuffleActive(MemOut& o, const std::string& src, const std::string& target,
                          const std::string& src_active, bool fetch_inactive = false) {
	auto t = o.Def("uint", target);
	auto v = o.Def("uint", "simd_shuffle(" + src + ", ushort(" + t + "))");
	if (fetch_inactive) return v;
	auto a = o.Def("bool", "simd_shuffle(uint(" + src_active + "), ushort(" + t + ")) != 0u");
	return o.Def("uint", a + " ? " + v + " : 0u");
}

struct Target {
	std::string lane;
	std::string valid;
};

Target DppTarget(const WaveCtx& w, const DppFlags& f, MemOut& o) {
	const std::string L = w.lane;
	const uint32_t    c = f.control;
	if (f.dpp8) {
		// each lane of a group of 8 selects with 3 bits
		return {"((" + L + " & ~7u) | ((" + U(c) + " >> ((" + L + " & 7u) * 3u)) & 7u))", "true"};
	}
	if (c <= 0xffu)
		return {"((" + L + " & ~3u) | ((" + U(c) + " >> ((" + L + " & 3u) * 2u)) & 3u))", "true"};
	if (c >= 0x101u && c <= 0x10fu)
		return {"((" + L + " & ~15u) | ((" + L + " & 15u) + " + U(c & 15u) + "))",
		        "((" + L + " & 15u) < " + U(16u - (c & 15u)) + ")"};
	if (c >= 0x111u && c <= 0x11fu)
		return {"((" + L + " & ~15u) | ((" + L + " & 15u) - " + U(c & 15u) + "))",
		        "((" + L + " & 15u) >= " + U(c & 15u) + ")"};
	if (c >= 0x121u && c <= 0x12fu)
		return {"((" + L + " & ~15u) | (((" + L + " & 15u) - " + U(c & 15u) + ") & 15u))", "true"};
	if (c == 0x140u) return {"(" + L + " ^ 15u)", "true"};
	if (c == 0x141u) return {"(" + L + " ^ 7u)", "true"};
	if (c >= 0x150u && c <= 0x15fu) return {"((" + L + " & ~15u) | " + U(c & 15u) + ")", "true"};
	if (c >= 0x160u && c <= 0x16fu) return {"(" + L + " ^ " + U(c & 15u) + ")", "true"};
	(void)o;
	return {"", ""};
}

} // namespace

std::string WaveBallot(MemOut& o, const WaveCtx& w, const std::string& pred) {
	return o.Def("uint4", "uint4(kyty_ballot(" + pred + ", " + w.lane + "), 0u, 0u, 0u)");
}
std::string WaveAny(MemOut& o, const WaveCtx& w, const std::string& pred) {
	return o.Def("bool", "kyty_any(" + pred + ", " + w.lane + ")");
}
std::string WaveReadFirstLane(MemOut& o, const WaveCtx&, const std::string& v) {
	return o.Def("uint", "simd_broadcast_first(" + v + ")");
}
std::string WaveReadLane(MemOut& o, const WaveCtx&, const std::string& v, const std::string& lane) {
	return o.Def("uint", "simd_shuffle(" + v + ", ushort(" + lane + " & 31u))");
}
std::string WaveWriteLane(MemOut& o, const std::string& guest_lane, const std::string& v,
                          const std::string& old, const std::string& target) {
	return o.Def("uint", "(" + guest_lane + " == (" + target + " & 63u)) ? " + v + " : " + old);
}

std::string WaveDppMove(MemOut& o, const WaveCtx& w, const DppFlags& f, const std::string& src,
                        const std::string& src_active) {
	const Target t = DppTarget(w, f, o);
	if (t.lane.empty()) {
		o.Fail("DPP control 0x" + std::to_string(f.control) +
		       " unsupported (wave shifts / row broadcast need wave64)");
		return "0u";
	}
	auto lane = o.Def("uint", t.lane);
	auto v    = o.Def("uint", "simd_shuffle(" + src + ", ushort(" + lane + "))");
	if (f.fetch_inactive) return v;
	auto a = o.Def("bool", "simd_shuffle(uint(" + src_active + "), ushort(" + lane + ")) != 0u");
	return o.Def("uint", "(" + t.valid + " && " + a + ") ? " + v + " : 0u");
}

std::string WaveDppUpdate(MemOut& o, const WaveCtx& w, const DppFlags& f, const std::string& moved,
                          const std::string& old, const std::string& exec) {
	std::string write = exec;
	if (f.bank_mask != 0xfu)
		write += " && (((" + U(f.bank_mask) + " >> ((" + w.lane + " >> 2u) & 3u)) & 1u) != 0u)";
	if (f.row_mask != 0xfu)
		write += " && (((" + U(f.row_mask) + " >> ((" + w.lane + " >> 4u) & 1u)) & 1u) != 0u)";
	const bool row_shift =
	    !f.dpp8 && f.control > 0x100u && f.control < 0x120u && f.control != 0x110u;
	if (!f.bound_control && row_shift) {
		const Target t = DppTarget(w, f, o);
		write += " && " + t.valid;
	}
	return o.Def("uint", "(" + write + ") ? " + moved + " : " + old);
}

std::string WaveSwizzle(MemOut& o, const WaveCtx& w, uint32_t c, const std::string& src,
                        const std::string& src_active) {
	const std::string L = w.lane;
	std::string       target;
	if ((c & 0xc000u) == 0xc000u) {
		const uint32_t mask = c & 0x1fu, rot = (c >> 5u) & 0x1fu;
		const uint32_t delta = (c & 0x400u) != 0u ? ((32u - rot) & 0x1fu) : rot;
		target = "((" + L + " & 31u & " + U(mask) + ") | (((" + L + " & 31u) + " + U(delta) +
		         ") & 31u & " + U(~mask & 31u) + "))";
	} else if ((c & 0x8000u) != 0u) {
		target = "((" + L + " & ~3u) | ((" + U(c) + " >> ((" + L + " & 3u) << 1u)) & 3u))";
	} else {
		target = "((((" + L + " & 31u & " + U(c & 0x1fu) + ") | " + U((c >> 5u) & 0x1fu) + ") ^ " +
		         U((c >> 10u) & 0x1fu) + "))";
	}
	return ShuffleActive(o, src, target, src_active);
}

std::string WaveBpermute(MemOut& o, const WaveCtx&, const std::string& src, const std::string& addr,
                         const std::string& src_active) {
	return ShuffleActive(o, src, "((" + addr + " >> 2u) & 31u)", src_active);
}

std::string WavePermute(MemOut& o, const WaveCtx& w, const std::string& src,
                        const std::string& addr, const std::string& active) {
	// Forward permute: lane i receives from the highest active lane j whose address selects i.
	auto writers = o.Def("uint", "kyty_ballot(" + active + ", " + w.lane + ")");
	for (uint32_t bit = 0; bit < 5; bit++) {
		auto m  = o.Def("uint", "kyty_ballot((" + addr + " & " + U(1u << (bit + 2)) + ") != 0u, " +
		                            w.lane + ")");
		writers = o.Def("uint", writers + " & ((" + w.lane + " & " + U(1u << bit) + ") != 0u ? " +
		                            m + " : ~" + m + ")");
	}
	auto act    = o.Def("bool", writers + " != 0u");
	auto source = o.Def("uint", act + " ? (31u - clz(" + writers + ")) : " + w.lane);
	auto v      = o.Def("uint", "simd_shuffle(" + src + ", ushort(" + source + "))");
	return o.Def("uint", act + " ? " + v + " : 0u");
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
