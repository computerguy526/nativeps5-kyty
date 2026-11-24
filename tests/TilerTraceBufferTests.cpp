// Buffer-mode tiler trace aggregation (graphics/host_gpu/renderer/image/tilerTraceBuffer.h).
#include "graphics/host_gpu/renderer/image/tilerTraceBuffer.h"

#include <cstdio>
#include <cstdlib>

namespace {
namespace T = Libs::Graphics::TilerTrace;
void Check(bool c, const char* m) {
	if (!c) {
		std::fprintf(stderr, "TilerTraceBufferTests: %s\n", m);
		std::abort();
	}
}
} // namespace

int main() {
	T::BufferAggregate agg;
	const T::BufferKey a {"detile.buf", "upload.refresh_w", 8, 2, 64, 64, 1};
	const T::BufferKey b {"tile.buf", "download.buffer_sync", 16, 2, 32, 32, 1};
	agg.Add(a, 32, 1000);
	agg.Add(a, 32, 500);
	agg.Add(b, 64, 2000);
	Check(agg.Size() == 2, "two keys");
	Check(agg.TotalBytes() == 3500, "sum bytes");
	const auto& v = agg.Map().at(a);
	Check(v.passes == 2 && v.infos == 64 && v.bytes == 1500, "per-key sums");
	Check(T::BufferAggregate::Format(a, v) ==
	          "[tilertrace] detile.buf upload.refresh_w elem=8 family=2 64x64x1 passes=2 infos=64 bytes=1500",
	      "format");
	agg.Add({"detile.buf", "other", 8, 2, 64, 64, 1}, 1, 1);
	Check(agg.Size() == 3, "reason splits keys");
	agg.Clear();
	Check(agg.Size() == 0 && agg.TotalBytes() == 0, "window reset");
	std::puts("TilerTraceBufferTests: ok");
	return 0;
}
