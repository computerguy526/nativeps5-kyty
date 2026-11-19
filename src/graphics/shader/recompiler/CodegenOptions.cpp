#include "graphics/shader/recompiler/CodegenOptions.h"

#include "common/logging/log.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <cinttypes>
#include <string>
#include <string_view>

namespace Libs::Graphics::ShaderRecompiler {
namespace {

// Unset or empty keeps the default; "0" disables; anything else enables.
bool EnvFlag(const char* name, bool default_value) {
	const auto* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return default_value;
	}
	return std::strcmp(value, "0") != 0;
}

// File of hex hashes (first token of each line, '#' starts a comment). Missing file -> *ok=false.
std::vector<uint64_t> ReadHashFile(const char* path, bool* ok) {
	std::vector<uint64_t> out;
	std::ifstream         in(path);
	*ok = in.is_open();
	std::string line;
	while (std::getline(in, line)) {
		if (const auto hash = line.find('#'); hash != std::string::npos) {
			line.resize(hash);
		}
		const auto begin = line.find_first_not_of(" \t\r");
		if (begin == std::string::npos) {
			continue;
		}
		out.push_back(std::strtoull(line.c_str() + begin, nullptr, 16));
	}
	std::sort(out.begin(), out.end());
	out.erase(std::unique(out.begin(), out.end()), out.end());
	return out;
}

} // namespace

// LodStatsCounter::PlainVariant()'s reading of the same variable: unset, empty or anything but
// "0" and "verify" is on.
bool LodStatsPlainVariantOn(const char* value) {
	return value == nullptr || value[0] == '\0' ||
	       (std::strcmp(value, "0") != 0 && std::strcmp(value, "verify") != 0);
}

namespace {

CodegenOptions FromEnvironment() {
	CodegenOptions options;
	options.movrel_range = EnvFlag("KYTY_MOVREL_RANGE", options.movrel_range);
	options.movrel_known_zeros = EnvFlag("KYTY_MOVREL_KNOWN_ZEROS", options.movrel_known_zeros);
	options.movrel_switch      = EnvFlag("KYTY_MOVREL_SWITCH", options.movrel_switch);
	options.uniform_lane_reads = EnvFlag("KYTY_UNIFORM_LANE_READS", options.uniform_lane_reads);
	options.short_f32_helpers  = EnvFlag("KYTY_SHORT_F32_HELPERS", options.short_f32_helpers);
	options.fast_float_min_max = EnvFlag("KYTY_FAST_FMINMAX", options.fast_float_min_max);
	options.native_float_min_max = EnvFlag("KYTY_NATIVE_FMINMAX", options.native_float_min_max);
	options.dcache_invalidate = EnvFlag("KYTY_DCACHE_INV", options.dcache_invalidate);
	options.readfirstlane_exec_zero =
	    EnvFlag("KYTY_READFIRSTLANE_EXEC0", options.readfirstlane_exec_zero);
	options.uniform_buffers      = EnvFlag("KYTY_UNIFORM_BUFFERS", options.uniform_buffers);
	options.wave64_warps         = EnvFlag("KYTY_WAVE64_WARPS", options.wave64_warps);
	options.fast_pkrtz         = EnvFlag("KYTY_FAST_PKRTZ", options.fast_pkrtz);
	options.single_f2i_saturation =
	    EnvFlag("KYTY_SINGLE_F2I_SATURATION", options.single_f2i_saturation);
	options.lod_stats_gate = EnvFlag("KYTY_LOD_STATS_GATE", options.lod_stats_gate);
	options.lod_stats_early_tests =
	    EnvFlag("KYTY_LOD_STATS_EARLY_TESTS", options.lod_stats_early_tests) &&
	    LodStatsPlainVariantOn(std::getenv("KYTY_LOD_STATS_PLAIN_VARIANT"));
	options.robust_buffer_loads = EnvFlag("KYTY_ROBUST_BUFFER_LOADS", options.robust_buffer_loads);
	options.branchless_loads    = EnvFlag("KYTY_BRANCHLESS_LOADS", options.branchless_loads);
	options.branchless_lds      = EnvFlag("KYTY_BRANCHLESS_LDS", options.branchless_lds);
	options.lane_local          = EnvFlag("KYTY_LANE_LOCAL", options.lane_local);
	options.readonly_buffers    = EnvFlag("KYTY_READONLY_BUFFERS", options.readonly_buffers);
	options.interp_modes = EnvFlag("KYTY_INTERP_MODES", options.interp_modes);
	options.sample_offsets = EnvFlag("KYTY_SAMPLE_OFFSETS", options.sample_offsets);
	options.sample_lod_clamp = EnvFlag("KYTY_SAMPLE_LOD_CLAMP", options.sample_lod_clamp);
	options.host_ftz_inputs  = EnvFlag("KYTY_HOST_FTZ_INPUTS", options.host_ftz_inputs);
	options.exec_selects     = EnvFlag("KYTY_EXEC_SELECTS", options.exec_selects);
	options.ps_append_live_election =
	    EnvFlag("KYTY_PS_APPEND_LIVE_ELECTION", options.ps_append_live_election);
	if (const auto* mode = std::getenv("KYTY_PS_LIVE_EXEC"); mode != nullptr && mode[0] != '\0') {
		options.ps_live_exec = std::strcmp(mode, "all") == 0 ? PsLiveExec::All
		                       : std::strcmp(mode, "0") == 0 ? PsLiveExec::Off
		                                                     : PsLiveExec::AppendConsume;
	}
	if (const auto* budget = std::getenv("KYTY_LOOP_GUARD"); budget != nullptr) {
		options.loop_guard_budget = static_cast<uint32_t>(std::strtoul(budget, nullptr, 0));
	}
	options.ps_helper_atomics_skip =
	    EnvFlag("KYTY_PS_HELPER_ATOMICS_SKIP", options.ps_helper_atomics_skip);
	options.volatile_loads         = EnvFlag("KYTY_VOLATILE_LOADS", options.volatile_loads);
	options.cfg_clone_exits = EnvFlag("KYTY_CFG_CLONE_EXITS", options.cfg_clone_exits);
	if (const auto* list = std::getenv("KYTY_LOOP_GUARD_SHADERS"); list != nullptr) {
		options.loop_guard_all = std::strcmp(list, "all") == 0 || std::strcmp(list, "*") == 0;
		std::string_view text(options.loop_guard_all ? "" : list);
		while (!text.empty()) {
			const auto comma = text.find(',');
			const auto token = std::string(text.substr(0, comma));
			if (!token.empty()) {
				options.loop_guard_shaders.push_back(std::strtoull(token.c_str(), nullptr, 16));
			}
			if (comma == std::string_view::npos) {
				break;
			}
			text.remove_prefix(comma + 1);
		}
	}
	options.srt_variant_reads = EnvFlag("KYTY_SRT_VARIANT_READS", options.srt_variant_reads);
	options.realtime_clock    = EnvFlag("KYTY_REALTIME_CLOCK", options.realtime_clock);
	options.dpp_skip_inactive = EnvFlag("KYTY_DPP_SKIP_INACTIVE", options.dpp_skip_inactive);
	options.lane_reductions   = EnvFlag("KYTY_LANE_REDUCTIONS", options.lane_reductions);
	options.ir_linear_uses    = EnvFlag("KYTY_IR_LINEAR_USES", options.ir_linear_uses);
	options.fold_lane_masks   = EnvFlag("KYTY_FOLD_LANE_MASKS", options.fold_lane_masks);
	if (const auto* cap = std::getenv("KYTY_DISPATCHER_CAP"); cap != nullptr && cap[0] != '\0') {
		options.dispatcher_cap = static_cast<uint32_t>(std::strtoul(cap, nullptr, 0));
	}
	// KYTY_NATIVE_INDIRECT_MESH=1|on|verify|exit, or unset/empty (the default, renderer/meshIndirect.h:
	// GPU-converted indirect mesh draws); 0 and "empty" keep the pushed-dword-only mesh draw
	// parameters.
	{
		const auto* mode = std::getenv("KYTY_NATIVE_INDIRECT_MESH");
		options.mesh_indirect_params =
		    mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "1") == 0 ||
		    std::strcmp(mode, "on") == 0 || std::strcmp(mode, "verify") == 0 ||
		    std::strcmp(mode, "exit") == 0;
	}
	options.rt_stub      = EnvFlag("KYTY_RT_STUB", options.rt_stub);
	if (const auto* mode = std::getenv("KYTY_RT_SOFTWARE");
	    mode != nullptr && mode[0] != '\0' && std::strcmp(mode, "auto") != 0) {
		options.rt_software      = std::strcmp(mode, "0") != 0;
		options.rt_software_auto = false;
	}
	if (const auto* mode = std::getenv("KYTY_RT_FUNCTION"); mode != nullptr && mode[0] != '\0') {
		options.rt_function = std::min<uint32_t>(static_cast<uint32_t>(std::strtoul(mode, nullptr, 10)), 2u);
	}
	options.rt_type6     = EnvFlag("KYTY_RT_TYPE6", options.rt_type6);
	options.bda_writes        = EnvFlag("KYTY_BDA_WRITES", options.bda_writes);
	options.bda_store_log     = EnvFlag("KYTY_BDA_STORE_LOG", options.bda_store_log);
	if (const auto* budget = std::getenv("KYTY_RT_NODE_BUDGET");
	    budget != nullptr && budget[0] != '\0') {
		options.rt_node_budget = static_cast<uint32_t>(std::strtoul(budget, nullptr, 0));
	}
	options.rt_node_stats = EnvFlag("KYTY_RT_NODE_STATS", options.rt_node_stats);
	if (const auto* mode = std::getenv("KYTY_LDS_WAITCNT_BARRIER"); mode != nullptr && mode[0] != '\0') {
		options.lds_waitcnt_barrier = std::strcmp(mode, "0") == 0 || std::strcmp(mode, "off") == 0
		                                  ? LdsWaitcntBarrier::Off
		                              : std::strcmp(mode, "workgroup") == 0 ? LdsWaitcntBarrier::Workgroup
		                                                                    : LdsWaitcntBarrier::Subgroup;
	}
	if (const auto* mode = std::getenv("KYTY_WAVE32_CLUSTERS"); mode != nullptr && mode[0] != '\0') {
		options.wave32_clusters = std::strcmp(mode, "0") == 0 || std::strcmp(mode, "off") == 0
		                              ? Wave32Clusters::Off
		                          : std::strcmp(mode, "force") == 0 ? Wave32Clusters::Force
		                                                            : Wave32Clusters::Auto;
	}
	if (const auto* mode = std::getenv("KYTY_LDS_DEVICE_BUFFER"); mode != nullptr && mode[0] != '\0') {
		options.lds_device_buffer = std::strcmp(mode, "0") == 0 || std::strcmp(mode, "off") == 0
		                                ? LdsDeviceBuffer::Off
		                            : std::strcmp(mode, "force") == 0 ? LdsDeviceBuffer::Force
		                                                              : LdsDeviceBuffer::Auto;
	}
	if (const auto* limit = std::getenv("KYTY_LDS_LIMIT_OVERRIDE"); limit != nullptr && limit[0] != '\0') {
		options.lds_limit_override = static_cast<uint32_t>(std::strtoul(limit, nullptr, 0));
	}
	if (const auto* mode = std::getenv("KYTY_PS_PER_VERTEX");
	    mode != nullptr && (std::strcmp(mode, "gs") == 0 || std::strcmp(mode, "2") == 0 ||
	                        std::strcmp(mode, "unpack") == 0 || std::strcmp(mode, "3") == 0)) {
		options.ps_per_vertex = true;
	} else {
		options.ps_per_vertex = EnvFlag("KYTY_PS_PER_VERTEX", options.ps_per_vertex);
	}
	options.clip_guard    = EnvFlag("KYTY_CLIP_GUARD", options.clip_guard);
	if (const auto* mode = std::getenv("KYTY_MAD_MODE"); mode != nullptr) {
		if (std::strcmp(mode, "exact") == 0) {
			options.mad_mode = MadMode::Exact;
		} else if (std::strcmp(mode, "fused") == 0) {
			options.mad_mode = MadMode::Fused;
		} else if (std::strcmp(mode, "position") == 0) {
			options.mad_mode = MadMode::Position;
		}
	}
	if (const auto* level = std::getenv("KYTY_FAST_MATH"); level != nullptr) {
		const auto value     = std::strtoul(level, nullptr, 10);
		options.fast_math    = static_cast<uint8_t>(value >= 3 ? 1ul : value);
		options.fast_math_vs = value >= 3;
	}
	options.fast_math_vs = EnvFlag("KYTY_FAST_MATH_VS", options.fast_math_vs);
	if (const auto* path = std::getenv("KYTY_FAST_MATH_DENY"); path != nullptr && path[0] != '\0') {
		bool ok              = false;
		options.fast_math_deny = ReadHashFile(path, &ok);
	}
	if (const auto* path = std::getenv("KYTY_FAST_MATH_ALLOW"); path != nullptr && path[0] != '\0') {
		options.fast_math_allow     = ReadHashFile(path, &options.fast_math_allow_set);
	}
	if (const auto* stages = std::getenv("KYTY_FAST_MATH_STAGES"); stages != nullptr) {
		const std::string text(stages);
		options.fast_math_stages =
		    static_cast<uint8_t>((text.find("fs") != std::string::npos ? 1u : 0u) |
		                         (text.find("cs") != std::string::npos ? 2u : 0u));
	}
	return options;
}

CodegenOptions& Storage() {
	static CodegenOptions options = FromEnvironment();
	return options;
}

} // namespace

namespace {
std::atomic<bool> g_host_compute_subgroups_32 {false};
} // namespace

namespace {
std::atomic<bool> g_host_buffer_int64_atomics {true};
} // namespace

void SetHostBufferInt64Atomics(bool supported) {
	g_host_buffer_int64_atomics.store(supported, std::memory_order_relaxed);
}

bool GetHostBufferInt64Atomics() {
	return g_host_buffer_int64_atomics.load(std::memory_order_relaxed);
}

namespace {
std::atomic<bool> g_host_vertex_subgroups {true};
} // namespace

namespace {
std::atomic<bool> g_host_ps_per_vertex {true};
} // namespace

void SetHostPsPerVertex(bool supported) {
	g_host_ps_per_vertex.store(supported, std::memory_order_relaxed);
}

bool GetHostPsPerVertex() {
	return g_host_ps_per_vertex.load(std::memory_order_relaxed);
}

namespace {
// The word of KYTY_PS_PER_VERTEX (kept outside CodegenOptions: read once). 0: none or a plain
// switch, 1: "gs", 2: "unpack".
int PsPerVertexWord() {
	static const int word = [] {
		const auto* mode = std::getenv("KYTY_PS_PER_VERTEX");
		if (mode == nullptr) {
			return 0;
		}
		if (std::strcmp(mode, "gs") == 0 || std::strcmp(mode, "2") == 0) {
			return 1;
		}
		if (std::strcmp(mode, "unpack") == 0 || std::strcmp(mode, "3") == 0) {
			return 2;
		}
		return 0;
	}();
	return word;
}
std::atomic<int> g_ps_per_vertex_test_mode {-1};
} // namespace

void SetPsPerVertexModeForTests(int mode) {
	g_ps_per_vertex_test_mode.store(mode, std::memory_order_relaxed);
}

bool PsPerVertexGeometry() {
	if (const auto test = g_ps_per_vertex_test_mode.load(std::memory_order_relaxed); test >= 0) {
		return test == 1;
	}
	return GetCodegenOptions().ps_per_vertex && PsPerVertexWord() == 1;
}

bool PsPerVertexUnpack() {
	if (const auto test = g_ps_per_vertex_test_mode.load(std::memory_order_relaxed); test >= 0) {
		return test == 2;
	}
	return GetCodegenOptions().ps_per_vertex &&
	       (PsPerVertexWord() == 2 || (PsPerVertexWord() == 0 && !GetHostPsPerVertex()));
}

void SetHostVertexSubgroups(bool supported) {
	g_host_vertex_subgroups.store(supported, std::memory_order_relaxed);
}

bool GetHostVertexSubgroups() {
	return g_host_vertex_subgroups.load(std::memory_order_relaxed);
}

void SetHostComputeSubgroups32(bool fixed_32) {
	g_host_compute_subgroups_32.store(fixed_32, std::memory_order_relaxed);
}

bool GetHostComputeSubgroups32() {
	return g_host_compute_subgroups_32.load(std::memory_order_relaxed);
}

namespace {
std::atomic<bool> g_host_uniform_buffer_robust {false};
} // namespace

void SetHostUniformBufferRobust(bool robust) {
	g_host_uniform_buffer_robust.store(robust, std::memory_order_relaxed);
}

bool GetHostUniformBufferRobust() {
	return g_host_uniform_buffer_robust.load(std::memory_order_relaxed);
}

const CodegenOptions& GetCodegenOptions() {
	return Storage();
}

bool LoopGuardApplies(uint64_t shader_hash) {
	const auto& options = Storage();
	return options.loop_guard_budget != 0 &&
	       (options.loop_guard_all || std::ranges::find(options.loop_guard_shaders, shader_hash) !=
	                                      options.loop_guard_shaders.end());
}

bool FastMathShaderAllowed(uint64_t shader_hash, const char* stage) {
	const auto& options = Storage();
	const bool  allowed =
	    (!options.fast_math_allow_set ||
	     std::binary_search(options.fast_math_allow.begin(), options.fast_math_allow.end(), shader_hash)) &&
	    !std::binary_search(options.fast_math_deny.begin(), options.fast_math_deny.end(), shader_hash);
	// KYTY_FAST_MATH_LOG=1: one line per guest hash (with its stage) the first time a decision is made, lists set or not
	// (checks that the lists hit real hashes and harvests the hashes a scene uses).
	static const bool        log_on = EnvFlag("KYTY_FAST_MATH_LOG", false);
	static std::mutex        mutex;
	static std::set<uint64_t> seen;
	if (log_on) {
		std::lock_guard lock(mutex);
		if (seen.insert(shader_hash).second) {
			LOGF("[fastmath] shader 0x%016" PRIx64 " %s %s\n", shader_hash, stage, allowed ? "FAST" : "EXACT (listed)");
		}
	}
	return allowed;
}

void SetCodegenOptions(const CodegenOptions& options) {
	Storage() = options;
}

namespace {
thread_local int g_rt_function_override = -1;
} // namespace

uint32_t RtFunctionMode() {
	return g_rt_function_override >= 0 ? static_cast<uint32_t>(g_rt_function_override)
	                                   : Storage().rt_function;
}

void SetThreadRtFunctionOverride(int mode) {
	g_rt_function_override = mode;
}

ComputeLdsPlan PlanComputeLds(uint32_t requested_dwords, uint32_t device_limit_bytes) {
	const auto& options = Storage();
	ComputeLdsPlan plan;
	plan.limit_bytes = options.lds_limit_override != 0
	                       ? std::min(device_limit_bytes, options.lds_limit_override)
	                       : device_limit_bytes;
	const auto limit_dwords = plan.limit_bytes / 4u;
	plan.dwords             = requested_dwords;
	switch (options.lds_device_buffer) {
		case LdsDeviceBuffer::Force: plan.storage = requested_dwords != 0; break;
		case LdsDeviceBuffer::Auto: plan.storage = requested_dwords > limit_dwords; break;
		case LdsDeviceBuffer::Off: break;
	}
	if (!plan.storage && requested_dwords > limit_dwords) {
		plan.dwords  = limit_dwords;
		plan.clamped = true;
	}
	return plan;
}

} // namespace Libs::Graphics::ShaderRecompiler
