// T2 msl-mem: LDS (DS_* local data share) -> Metal threadgroup memory, plus DS atomics.
// LDS is the entry parameter `threadgroup uint* lds [[threadgroup(0)]]` (bytes in the manifest's
// threadgroup_memory[0]). Out-of-range reads return 0 and writes are dropped (hardware rule), via
// a compile-time `addr + N <= lds_bytes` guard that Metal folds away for constant offsets.
#pragma once

#include "graphics/shader/recompiler/backend/msl/MslBufferOps.h"

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

struct LdsInfo {
	uint32_t bytes = 0;
	// mslrest: LDS in KK's SharedMemory device buffer (Kyty lds_storage, guest LDS > Metal's 32
	// KB): pointer space "coherent device" (as KK's NIR->MSL emits for the Coherent SPIR-V buffer)
	// and a run-time byte bound (0 for a workgroup the bound range holds no region for). Empty =
	// threadgroup memory with the static bound `bytes`.
	std::string space = "threadgroup";
	std::string bound;
};
// Per-emission switch (kk-bda + SharedMemory binding): LDS ops read the device region `lds` bounded
// by `kk_lds_bytes`.
void SetLdsDevice(bool on);
// meshpath: kk-bda guest-address loads through the BDA page table (KkBdaLocals declares
// kk_bda_pt/_ptn and kk_bda_fb/_fbn from set 0's BdaPagetable / FaultBuffer descriptors).
void SetKkBdaPagetable(bool on);
bool KkBdaPagetable();
bool LdsIsDevice();
// mslrest4: graphics stages (no workgroup): LDS = per-invocation `thread` array of `bytes` (Kyty
// spirv LdsDwordCount Function variable: pixel max(EXTRA_LDS dwords,1), vertex max(lds dwords,1));
// 0 = off.
void     SetLdsThread(uint32_t bytes);
uint32_t LdsThreadBytes();
LdsInfo  MakeLdsInfo(uint32_t bytes);

std::string LdsLoad(MemOut& o, const LdsInfo& l, const std::string& addr, uint32_t bits,
                    uint32_t comps);
void        LdsStore(MemOut& o, const LdsInfo& l, const std::string& addr, uint32_t bits,
                     const std::vector<std::string>& values);
// 32-bit DS atomics; returns pre-op uint. cmp for CmpSwap. FMin/FMax are CAS loops.
std::string LdsAtomic32(MemOut& o, const LdsInfo& l, const std::string& addr, AtomicOp op,
                        const std::string& value, const std::string& cmp = "0u");
// DS_MSKOR_B32: mem = (mem & ~mask) | value. Returns pre-op value.
std::string LdsMaskedOr32(MemOut& o, const LdsInfo& l, const std::string& addr,
                          const std::string& mask, const std::string& value);
// 64-bit LDS atomics (value is ulong): Add/Or/And/Xor/Sub by halves (pair not atomic as a unit);
// others report o.error. Returns pre-op ulong.
std::string LdsAtomic64(MemOut& o, const LdsInfo& l, const std::string& addr, AtomicOp op,
                        const std::string& value);
// Workgroup barrier + LDS visibility (S_BARRIER / SharedMemoryBarrier).
void LdsBarrier(MemOut& o);

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
