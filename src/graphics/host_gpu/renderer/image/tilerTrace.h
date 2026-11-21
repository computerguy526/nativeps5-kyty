#pragma once
// KYTY_TILER_TRACE=<seconds> (diagnostic, default off): every <seconds> stderr gets one line per
// (direction, caller reason, guest image) that went through the tiler's image passes
// (DetileToImage = guest bytes -> image, TileFromImage = image -> guest bytes) in that window:
// count, bytes, format, extent, tile mode, and the image's modified flags at the call.
// It answers why render targets cross the tiler every frame. No effect on rendering.
#include <cstdint>

namespace Libs::Graphics {
class Image;

namespace TilerTrace {

[[nodiscard]] bool Enabled() noexcept;

// The innermost caller reason of this thread ("upload.refresh", "download.cpu_read", ...).
class Reason {
public:
	explicit Reason(const char* why) noexcept;
	~Reason();
	Reason(const Reason&)            = delete;
	Reason& operator=(const Reason&) = delete;

private:
	const char* m_previous;
};

void Note(bool to_image, const Image& image, uint64_t bytes) noexcept;
// A named event about an image ("event <what>" lines; e.g. the partial-rebuild path and its
// fallbacks). bytes is what the event concerns.
void Event(const char* what, const Image& image, uint64_t bytes) noexcept;
// A GPU buffer write marked the image buffer-modified ("event bwrite.<reason>.<cover>", cover =
// exact / inside / covers / partial relative to the image's range; reason = this thread's
// Reason, set by the writer: storage, bda, fill, write_data, copy, preserve).
void BufferWrite(const Image& image, uint64_t address, uint64_t size) noexcept;
// "event <prefix>.<this thread's Reason>".
void EventWithReason(const char* prefix, const Image& image, uint64_t bytes) noexcept;
// A named counter without an image ("count <what> n=<n> bytes=<bytes>" lines per window): GPU
// buffer fills by caller (fill.<who>) and the tiler's image-pass dispatches (dispatch.store /
// dispatch.load, bytes = threads).
void Count(const char* what, uint64_t n, uint64_t bytes) noexcept;

} // namespace TilerTrace
} // namespace Libs::Graphics
