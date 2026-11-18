// T2 msl-mem: GCN wave operations -> Metal simdgroup intrinsics.
// Host model: one Metal simdgroup (32 lanes on Apple GPUs) == one guest wave of 32 lanes (RDNA
// wave32 or a wave64 run "lane-local"/split into two guest waves). Wave64 semantics across two
// simdgroups are NOT implemented here (T1 decides wave size; ballot is returned as uint4 {lo32, 0,
// 0, 0} so the IR type is unchanged). `lane` is the name of a uint holding
// thread_index_in_simdgroup.
#pragma once

#include "graphics/shader/recompiler/backend/msl/MslMemCommon.h"

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

struct WaveCtx {
	std::string lane = "lane"; // uint, thread_index_in_simdgroup
};

std::string WaveBallot(MemOut& o, const WaveCtx& w, const std::string& pred); // uint4 {mask,0,0,0}
std::string WaveAny(MemOut& o, const WaveCtx& w, const std::string& pred);    // bool
std::string WaveReadFirstLane(MemOut& o, const WaveCtx& w, const std::string& v); // uint
std::string WaveReadLane(MemOut& o, const WaveCtx& w, const std::string& v,
                         const std::string& lane);
// guest_lane: 0..63 guest lane of this thread/half (not the simd lane).
std::string WaveWriteLane(MemOut& o, const std::string& guest_lane, const std::string& v,
                          const std::string& old, const std::string& target);

struct DppFlags {
	uint32_t control  = 0;
	uint32_t row_mask = 0xf, bank_mask = 0xf;
	bool     bound_control = false, fetch_inactive = false, dpp8 = false;
};
// DppMoveU32(src, exec_src): returns src from the DPP-selected lane (0 if invalid/inactive).
std::string WaveDppMove(MemOut& o, const WaveCtx& w, const DppFlags& f, const std::string& src,
                        const std::string& src_active);
// DppUpdateU32(moved, old, exec): masked select between the moved value and `old`.
std::string WaveDppUpdate(MemOut& o, const WaveCtx& w, const DppFlags& f, const std::string& moved,
                          const std::string& old, const std::string& exec);
// DS_SWIZZLE_B32 with an immediate control word; src_active = exec predicate of the source lane.
std::string WaveSwizzle(MemOut& o, const WaveCtx& w, uint32_t control, const std::string& src,
                        const std::string& src_active);
std::string WaveBpermute(MemOut& o, const WaveCtx& w, const std::string& src,
                         const std::string& byte_addr, const std::string& src_active);
std::string WavePermute(MemOut& o, const WaveCtx& w, const std::string& src,
                        const std::string& byte_addr, const std::string& active);

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
