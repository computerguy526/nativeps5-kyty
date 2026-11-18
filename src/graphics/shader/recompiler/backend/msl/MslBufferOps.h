// T2 msl-mem: GCN buffer memory (BUFFER_*/TBUFFER_*/S_BUFFER_LOAD/atomics) -> MSL.
// Buffers are `device uchar*` in the argument buffer; accesses cast to packed_* / sized scalars so
// no alignment beyond the element size is assumed. No `coherent` qualifier is emitted (KK's
// `coherent device` loads defeat Apple's cache; the guest's coherent bit only needs it for
// inter-workgroup flag spins, selectable via BufferOpts::coherent_loads).
#pragma once

#include <cstdlib>
#include "graphics/shader/recompiler/backend/msl/MslBinding.h"

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

// KYTY_MSL_BOUNDS=0 drops every per-access descriptor-size guard (host guarantees in-range descriptors); default = guarded.
inline bool BoundsElideFromEnv() {
	static const bool v = [] { const char* e = std::getenv("KYTY_MSL_BOUNDS"); return e != nullptr && e[0] == '0'; }();
	return v;
}

struct BufferOpts {
	bool coherent_loads   = false; // emit `volatile` + relaxed atomic-ish loads (rarely needed)
	bool elide_bounds     = BoundsElideFromEnv(); // global override: trust the harness/host to size buffers (no guard)
};

// Descriptor-derived address inputs (all MSL uint expressions; empty = 0).
struct BufferAddrIn {
	std::string index;    // vindex  (idxen)
	std::string offset;   // voffset (offen)
	std::string soffset;  // SGPR offset
	uint32_t    imm = 0;  // instruction immediate
	std::string lane;     // lane id (0..63) expression for ADD_TID; empty if unused
};

// Returns the name of a `uint` holding the byte address (GCN swizzle/stride/add_tid applied).
std::string BufferByteAddress(MemOut& o, const BufferBinding& b, const BufferAddrIn& in);

// Raw loads: bits 8/16 return a zero-extended uint; 32 with comps 1..4 returns uint/uint2/uint3/uint4.
// Out-of-bounds (descriptor size) reads return 0.
std::string LoadRaw(MemOut& o, const BufferBinding& b, const std::string& addr, uint32_t bits,
                    uint32_t comps, const BufferOpts& opts = {});
// values: one uint expression per component (low bits used for 8/16).
void StoreRaw(MemOut& o, const BufferBinding& b, const std::string& addr, uint32_t bits,
              const std::vector<std::string>& values, const BufferOpts& opts = {});

// Formatted (TBUFFER / MUBUF format from the V#): format is a Prospero::BufferFormat value,
// dst_sel the 12-bit descriptor swizzle (DstSel(x,y,z,w), 3 bits each). Returns a uint4 of raw bits.
std::string LoadFormatted(MemOut& o, const BufferBinding& b, const std::string& addr, uint32_t format,
                          uint32_t dst_sel, const BufferOpts& opts = {});
// values: up to 4 uint-bit expressions (float bits for float formats).
void StoreFormatted(MemOut& o, const BufferBinding& b, const std::string& addr, uint32_t format,
                    const std::vector<std::string>& values, const BufferOpts& opts = {});

enum class AtomicOp { Swap, CmpSwap, Add, Sub, SMin, UMin, SMax, UMax, And, Or, Xor, Inc, Dec, FMin, FMax };
// 32-bit: returns the pre-op value (uint). cmp used only by CmpSwap. Guarded by bounds.
std::string BufferAtomic32(MemOut& o, const BufferBinding& b, const std::string& addr, AtomicOp op,
                           const std::string& value, const std::string& cmp = "0u", const BufferOpts& opts = {});
// Atomic RMW on an already-formed `<space> atomic_uint*` expression (space: "device"/"threadgroup");
// shared with the LDS emitter. Emits at `indent`, returns the name of the pre-op uint.
std::string AtomicRmw32(MemOut& o, const std::string& atomic_ptr, const char* space, AtomicOp op,
                        const std::string& value, const std::string& cmp = "0u", std::string indent = "");
// 64-bit: value/cmp are `ulong`. Metal has no general 64-bit device atomics on this toolchain, so
// Add/Sub/And/Or/Xor/UMax/UMin/SMax/SMin are done on the two halves (each half atomic; the pair is
// NOT atomic as a unit; Add/Sub propagate carry) and Swap/CmpSwap use a per-address 32-bit spin
// lock on the low word's... (unsupported -> error). Returns pre-op value (ulong).
std::string BufferAtomic64(MemOut& o, const BufferBinding& b, const std::string& addr, AtomicOp op,
                           const std::string& value, const std::string& cmp = "0ul", const BufferOpts& opts = {});

// S_BUFFER_LOAD: constant-cache dword read at (resource, uint offset); returns uint.
std::string ReadConstBuffer(MemOut& o, const BufferBinding& b, const std::string& byte_offset, const BufferOpts& opts = {});

// MSL helper functions (ufloat convert etc.). Emit once at file scope before any LoadFormatted use.
std::string BufferHelpersPrelude();

// Guest/physical address memory (LoadAddress*/StoreAddress*, GetAddressResource): `a` is a 64-bit
// guest address expression (ulong). Reads are unaligned-safe (aligned dword merge, as the SPIR-V
// BDA path); 32-bit stores use byte stores when misaligned. Guest->device translation is the macro
// T2_GUEST_PTR(a) (identity by default: the runtime maps guest memory at the same device addresses,
// or overrides the macro with a page-table lookup).
std::string AddressHelpersPrelude();
std::string AddressLoad(MemOut& o, const std::string& a, uint32_t bits);
void        AddressStore(MemOut& o, const std::string& a, uint32_t bits, const std::string& value);

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
