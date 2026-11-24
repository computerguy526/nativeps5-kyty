#include "graphics/host_gpu/renderer/inputProfileDraw.h"

#include "common/guestAddress.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics::InputProfileDraw {

namespace {

// 'B'/'W' info bits.
constexpr uint32_t kRead      = 1u << 0u;
constexpr uint32_t kWritten   = 1u << 1u;
constexpr uint32_t kAtomic    = 1u << 2u;
constexpr uint32_t kFormatted = 1u << 3u;
constexpr uint32_t kScalar    = 1u << 4u;
constexpr uint32_t kGpuDirty  = 1u << 5u; // some samples skipped: GPU-modified range
constexpr uint32_t kUnmapped  = 1u << 6u;

// Readable guest bytes of [address, address + size): null when unmapped. valid: one bit per
// InputProfile::kChunkBytes chunk that is not GPU-modified (a GPU-modified CPU copy may be stale,
// and reading it must not start a read-back; those chunks are skipped).
const void* Readable(BufferCache& cache, uint64_t address, uint64_t& size, uint32_t& info,
                     uint32_t& valid) {
	valid = 0;
	if (address == 0 || size == 0) {
		return nullptr;
	}
	const uint64_t mapped = Libs::LibKernel::Memory::ClampRangeSizeQuiet(address, size);
	if (mapped < size) {
		info |= kUnmapped;
		size = mapped & ~uint64_t {3};
		if (size == 0) {
			return nullptr;
		}
	}
	for (uint64_t o = 0, c = 0; o < size && c < 32; o += InputProfile::kChunkBytes, c++) {
		const uint64_t n = std::min<uint64_t>(InputProfile::kChunkBytes, size - o);
		if (cache.IsRegionGpuModifiedRelaxed(address + o, n)) {
			info |= kGpuDirty;
		} else {
			valid |= 1u << c;
		}
	}
	return valid != 0 ? Guest::Ptr<const void>(address) : nullptr;
}

} // namespace

void NoteImpl(BufferCache& cache, std::span<PreparedBindings* const> stages) noexcept {
	auto&          reg = InputProfile::Global();
	const uint64_t now = InputProfile::NowMs();
	for (auto* stage: stages) {
		if (stage == nullptr || stage->runtime == nullptr || !*stage->runtime) {
			continue;
		}
		const auto&    program  = *stage->runtime->program;
		const auto&    snapshot = *stage->runtime->resources;
		const uint64_t hash     = program.shader_hash;
		const auto     st       = static_cast<uint32_t>(program.stage);
		if (!reg.BeginSample(hash, st, 0, now)) {
			continue;
		}
		// 'U': the shader data dwords as bound (user SGPR dwords first, memory offsets from
		// memory_offset_dword on). meta: memory_offset_dword, push start (or ~0), user dwords.
		{
			const uint32_t meta[3] = {
			    program.bindings.memory_offset_dword, program.bindings.push_data_start_dword,
			    static_cast<uint32_t>(program.bindings.user_data_registers.size())};
			reg.AddData(hash, st, 'U', 0, stage->shader_data.data(), stage->shader_data.size() * 4,
			            stage->shader_data.size() * 4, meta, 3, 0);
		}
		if (!snapshot.flattened_srt.empty()) {
			reg.AddData(hash, st, 'R', 0, snapshot.flattened_srt.data(),
			            snapshot.flattened_srt.size() * 4, snapshot.flattened_srt.size() * 4,
			            nullptr, 0, 0);
		}
		const uint32_t cap = reg.Cfg().read_bytes;
		for (uint32_t i = 0; i < stage->buffer_sources.size() && i < program.info.buffers.size();
		     i++) {
			const auto& src     = stage->buffer_sources[i];
			const auto& res     = program.info.buffers[i];
			uint32_t    info    = (res.read ? kRead : 0) | (res.written ? kWritten : 0) |
			                      (res.atomic ? kAtomic : 0) | (res.formatted ? kFormatted : 0) |
			                      (res.scalar ? kScalar : 0);
			uint32_t    meta[6] = {};
			if (i < snapshot.buffers.size()) {
				for (uint32_t w = 0; w < 4; w++) {
					meta[w] = snapshot.buffers[i].dwords[w];
				}
			}
			meta[4] = res.max_byte_extent;
			meta[5] = res.packed_stride;
			if (res.written || res.atomic) {
				reg.AddDescriptor(hash, st, 'W', i, meta, 6, src.size, info);
				continue;
			}
			uint64_t    n     = std::min<uint64_t>(src.size, cap);
			uint32_t    valid = 0;
			const void* data  = Readable(cache, src.address, n, info, valid);
			reg.AddData(hash, st, 'B', i, data, data != nullptr ? n : 0, src.size, meta, 6, info, 0,
			            valid);
		}
		for (uint32_t b = 0; b < snapshot.images.size(); b++) {
			reg.AddDescriptor(hash, st, 'T', b, snapshot.images[b].dwords.data(),
			                  snapshot.images[b].dword_count);
		}
		for (uint32_t s = 0; s < snapshot.samplers.size(); s++) {
			reg.AddDescriptor(hash, st, 'S', s, snapshot.samplers[s].dwords.data(),
			                  snapshot.samplers[s].dword_count);
		}
	}
}

void NoteVertexImpl(BufferCache& cache, const ShaderVertexInputInfo& info) noexcept {
	if (!info.stage) {
		return;
	}
	auto&          reg  = InputProfile::Global();
	const uint64_t hash = info.stage.program->shader_hash;
	const auto     st   = static_cast<uint32_t>(info.stage.program->stage);
	if (!reg.BeginSample(hash, st, 1, InputProfile::NowMs())) {
		return;
	}
	for (int i = 0; i < info.buffers_num && i < ShaderVertexInputInfo::RES_MAX; i++) {
		const auto& vb      = info.buffers[i];
		uint32_t    flags   = 0;
		uint32_t    meta[8] = {vb.stride, vb.num_records, static_cast<uint32_t>(vb.attr_num)};
		for (int a = 0; a < vb.attr_num && a < 5; a++) {
			meta[3 + a] = vb.attr_offsets[a];
		}
		const uint64_t verts = std::min<uint64_t>(vb.num_records, reg.Cfg().vertex_count);
		uint64_t       n     = std::min<uint64_t>(uint64_t {vb.stride} * verts, 16384);
		uint32_t       valid = 0;
		const void*    data  = Readable(cache, vb.addr, n, flags, valid);
		reg.AddData(hash, st, 'V', static_cast<uint32_t>(i), data, data != nullptr ? n : 0,
		            uint64_t {vb.stride} * vb.num_records, meta, 8, flags, vb.stride, valid);
	}
}

void NoteMeshImpl(BufferCache& cache, const ShaderVertexInputInfo& info, uint32_t index_count,
                  uint32_t instances, uint64_t index_address, uint32_t index_size, uint32_t groups,
                  uint32_t segments) noexcept {
	if (!info.stage) {
		return;
	}
	auto&          reg  = InputProfile::Global();
	const uint64_t hash = info.stage.program->shader_hash;
	const auto     st   = static_cast<uint32_t>(info.stage.program->stage);
	if (!reg.BeginSample(hash, st, 2, InputProfile::NowMs())) {
		return;
	}
	const auto&    mesh    = info.mesh;
	const uint32_t meta[8] = {index_count,
	                          instances,
	                          index_size,
	                          groups,
	                          segments,
	                          mesh.threads_num[0],
	                          mesh.lds_size_dwords,
	                          (mesh.vertices_per_group & 0xffffu) |
	                              (mesh.primitives_per_group << 16u)};
	uint32_t       flags   = 0;
	uint32_t       valid   = 0;
	uint64_t    n = std::min<uint64_t>(uint64_t {index_count} * index_size, 16384) & ~uint64_t {3};
	const void* data = index_size != 0 ? Readable(cache, index_address, n, flags, valid) : nullptr;
	reg.AddData(hash, st, 'M', 0, data, data != nullptr ? n : 0,
	            uint64_t {index_count} * index_size, meta, 8, flags, 0, valid);
}

} // namespace Libs::Graphics::InputProfileDraw
