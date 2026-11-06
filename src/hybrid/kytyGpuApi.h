#ifndef KYTY_HYBRID_KYTYGPUAPI_H_
#define KYTY_HYBRID_KYTYGPUAPI_H_

/* AstroHybrid: the C interface between AnyPS5's native AstroBot.exe and kyty_gpu.dll.
 *
 * Both sides include this header with different toolchains (AnyPS5: MinGW GCC, Itanium C++ ABI;
 * Kyty: clang-cl, MSVC ABI). Only C types cross it. Functions in the two tables below use the
 * platform's default C calling convention (Microsoft x64 on Windows), which both toolchains agree
 * on. The NID-named exports of kyty_gpu.dll are different: they are called by game code and use
 * the System V calling convention, like AnyPS5's own prx exports (APS5_VABI).
 *
 * Versioning: the host passes KYTY_GPU_ABI_VERSION and sizeof() of its structs. The DLL refuses a
 * different major version or a struct smaller than the fields it reads; fields are only ever
 * appended. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KYTY_GPU_ABI_VERSION_MAJOR 1u
#define KYTY_GPU_ABI_VERSION_MINOR 1u
#define KYTY_GPU_ABI_VERSION ((KYTY_GPU_ABI_VERSION_MAJOR << 16u) | KYTY_GPU_ABI_VERSION_MINOR)

#if defined(_WIN32)
#if defined(KYTY_GPU_BUILDING_DLL)
#define KYTY_GPU_EXPORT __declspec(dllexport)
#else
#define KYTY_GPU_EXPORT __declspec(dllimport)
#endif
#else
#define KYTY_GPU_EXPORT __attribute__((visibility("default")))
#endif

typedef enum KytyGpuStatus {
	KYTY_GPU_OK                  = 0,
	KYTY_GPU_E_VERSION           = -1, /* major version or struct size not supported */
	KYTY_GPU_E_ARGUMENT          = -2,
	KYTY_GPU_E_STATE             = -3, /* called in the wrong lifecycle state */
	KYTY_GPU_E_NOT_IMPLEMENTED   = -4, /* interface present, implementation is a later milestone */
	KYTY_GPU_E_HOST_CALLBACK     = -5, /* a required host callback is missing */
	KYTY_GPU_E_UNKNOWN_NID       = -6,
} KytyGpuStatus;

/* How an exported NID reaches its implementation. */
typedef enum KytyGpuBinding {
	KYTY_GPU_BINDING_KYTY_VERIFIED = 1, /* Kyty's HLE function; argument layout compared with AnyPS5's */
	KYTY_GPU_BINDING_KYTY          = 2, /* Kyty's HLE function; game ABI, layout not yet compared */
	KYTY_GPU_BINDING_BRIDGE        = 3, /* implemented by the bridge (host handles, AnyPS5 semantics) */
} KytyGpuBinding;

typedef struct KytyGpuNidInfo {
	const char* nid;      /* export name, e.g. "UglJIZjGssM" */
	const char* library;  /* AnyPS5 prx the game imports it from, e.g. "libSceAgcDriver.prx" */
	const char* function; /* implementation name, for logs */
	uint32_t    binding;  /* KytyGpuBinding */
	uint32_t    bound;    /* nonzero once the export reaches its implementation */
} KytyGpuNidInfo;

/* Memory flags for register_memory. */
#define KYTY_GPU_MEMORY_GPU_READ  0x1u
#define KYTY_GPU_MEMORY_GPU_WRITE 0x2u
#define KYTY_GPU_MEMORY_DIRECT    0x4u /* direct memory (has a backing alias) */

/* GPU -> host event kinds (host_callbacks.gpu_event). */
#define KYTY_GPU_EVENT_EOP_INTERRUPT 1u /* ident: AGC event id (queue), data: unused */
#define KYTY_GPU_EVENT_FLIP          2u /* ident: video out handle, data: flip argument */
#define KYTY_GPU_EVENT_VBLANK        3u /* ident: video out handle */

typedef struct KytyGpuHostCallbacks {
	uint32_t struct_size; /* sizeof(KytyGpuHostCallbacks) on the host */
	uint32_t abi_version; /* KYTY_GPU_ABI_VERSION of the host */
	void*    user;        /* passed back to every callback */

	/* Diagnostics. level: 0 info, 1 warning, 2 error. May be null. */
	void (*log)(void* user, int level, const char* message);

	/* The game's event queues live in AnyPS5's libkernel. The bridge forwards the AGC and VideoOut
	 * event registrations to the host, which owns the queue handles, and raises events through
	 * deliver_event. Return 0 or a negative SCE error. */
	int (*add_gpu_event)(void* user, int64_t equeue, uint32_t kind, int64_t ident, void* udata);
	int (*delete_gpu_event)(void* user, int64_t equeue, uint32_t kind, int64_t ident);
	void (*deliver_event)(void* user, uint32_t kind, int64_t ident, uint64_t data);

	/* Guest memory as AnyPS5's libkernel maps it. Return 0 and fill the range holding address, or
	 * a negative value when it is not mapped. alias: an always-readable host view of the same
	 * bytes, or null when the guest view itself is the backing. */
	int (*query_memory)(void* user, uint64_t address, uint64_t* base, uint64_t* size,
	                    uint32_t* flags, const void** alias);

	/* The title's app0 directory (UTF-8), for shader and pipeline caches. May be null. */
	const char* (*app0_path)(void* user);

	/* ABI 1.1. Kyty's kernel memory owns the GPU-visible (direct) memory and reserves the free
	 * guest address space when the renderer starts. The host's own guest arena holds that space,
	 * so it hands one range of `size` bytes to Kyty first and returns its base. A nonzero *base
	 * on input asks for exactly that range (Kyty's fixed-address mappings). The range stays a
	 * reserved placeholder (MEM_RESERVE_PLACEHOLDER) with its own boundaries, never MEM_FREE:
	 * Kyty adopts it as it is. */
	int (*donate_guest_range)(void* user, uint64_t size, uint64_t* base);
} KytyGpuHostCallbacks;

/* The host module kyty_gpu.dll looks for when game code reaches an export before any host called
 * init: LoadLibrary(KYTY_GPU_HOST_MODULE or "libAstroHybridHost.prx"), then this function. */
#define KYTY_GPU_HOST_MODULE_DEFAULT "libAstroHybridHost.prx"
#define KYTY_GPU_HOST_ENTRY "AstroHybridHostCallbacks_nid_no_patch"
typedef const KytyGpuHostCallbacks* (*KytyGpuHostEntryFn)(void);

typedef struct KytyGpuPacket { /* identical to AGC's Packet and AnyPS5's Packet */
	uint32_t* addr;
	uint32_t  dw_num;
	uint8_t   flags;
	uint8_t   reserved[3];
} KytyGpuPacket;

typedef struct KytyGpuApi {
	uint32_t struct_size;
	uint32_t abi_version;
	const char* build; /* Kyty git version of the DLL */

	/* Lifecycle. init binds every NID export (no graphics device yet); start_graphics creates
	 * Kyty's Vulkan renderer and presentation; shutdown stops them. */
	int (*init)(const KytyGpuHostCallbacks* host);
	int (*start_graphics)(void);
	void (*shutdown)(void);

	/* Guest memory notifications from AnyPS5's libkernel. */
	int (*register_memory)(uint64_t base, uint64_t size, uint32_t flags, const void* alias);
	int (*unregister_memory)(uint64_t base, uint64_t size);
	int (*invalidate_memory)(uint64_t base, uint64_t size); /* the CPU wrote the range */

	/* Command submission (same semantics as the AGC driver calls). */
	int (*submit_dcb)(const KytyGpuPacket* packet);
	int (*submit_acb)(uint32_t queue, const KytyGpuPacket* packet);

	/* Export table: count and the entry for an index or NID (null when out of range / unknown). */
	uint32_t (*nid_count)(void);
	const KytyGpuNidInfo* (*nid_info)(uint32_t index);
	const KytyGpuNidInfo* (*find_nid)(const char* nid);
} KytyGpuApi;

/* The DLL's only C entry point; the host asks for the major version it was built with. Returns
 * null for an unsupported version. */
KYTY_GPU_EXPORT const KytyGpuApi* KytyGpuGetApi(uint32_t abi_version);
typedef const KytyGpuApi* (*KytyGpuGetApiFn)(uint32_t abi_version);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus)
static_assert(sizeof(KytyGpuPacket) == 16, "KytyGpuPacket layout");
static_assert(offsetof(KytyGpuPacket, dw_num) == 8, "KytyGpuPacket layout");
static_assert(offsetof(KytyGpuPacket, flags) == 12, "KytyGpuPacket layout");
#else
_Static_assert(sizeof(KytyGpuPacket) == 16, "KytyGpuPacket layout");
_Static_assert(offsetof(KytyGpuPacket, dw_num) == 8, "KytyGpuPacket layout");
_Static_assert(offsetof(KytyGpuPacket, flags) == 12, "KytyGpuPacket layout");
#endif

#endif /* KYTY_HYBRID_KYTYGPUAPI_H_ */
