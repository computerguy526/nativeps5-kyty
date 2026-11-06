/* AstroHybrid milestone 1 acceptance test for kyty_gpu.dll.
 *
 * Built with AnyPS5's toolchain (MinGW GCC), not Kyty's, so it checks the C interface and the
 * System V NID exports across the two compilers, the way AstroBot.exe will use them:
 *   1. load the DLL (optionally after AnyPS5's libwinpthread-1.dll, as inside AstroBot.exe)
 *   2. version negotiation and struct layouts of kytyGpuApi.h
 *   3. every NID in the table is exported, and every NID export is in the table
 *   4. init binds every export; a wrong host ABI is refused
 *   5. System V calls through the thunks: a bridge function reaches the host callback with the
 *      exact arguments, Kyty functions return their AGC results
 *   6. lifecycle errors: start_graphics is not implemented yet, submit before graphics is refused
 *
 * Usage: kytyGpuLoadTest <kyty_gpu.dll> [<libwinpthread-1.dll to load first>]
 * Exit code 0 when every check passes. */

#include "../../src/hybrid/kytyGpuApi.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

#define SYSV __attribute__((sysv_abi))

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(condition, ...)                                                                      \
	do {                                                                                           \
		++g_checks;                                                                                \
		if (!(condition)) {                                                                        \
			++g_failures;                                                                          \
			printf("FAIL %s:%d: ", __FILE__, __LINE__);                                            \
			printf(__VA_ARGS__);                                                                   \
			printf("\n");                                                                          \
		}                                                                                          \
	} while (0)

struct HostRecord {
	int      calls;
	int64_t  equeue;
	uint32_t kind;
	int64_t  ident;
	void*    udata;
};

static struct HostRecord g_record;

static void HostLog(void* user, int level, const char* message) {
	(void)user;
	printf("  [dll %d] %s\n", level, message);
}

static int HostAddGpuEvent(void* user, int64_t equeue, uint32_t kind, int64_t ident, void* udata) {
	struct HostRecord* record = (struct HostRecord*)user;
	++record->calls;
	record->equeue = equeue;
	record->kind   = kind;
	record->ident  = ident;
	record->udata  = udata;
	return 42;
}

typedef int (SYSV *AddEqEventFn)(int64_t equeue, int id, void* udata);
typedef int (SYSV *RegisterOwnerFn)(uint32_t* owner, const char* name);
typedef uint32_t (SYSV *NopGetSizeFn)(uint32_t dwords);
typedef void (SYSV *TrinityFn)(uint8_t* result);

static size_t CountExports(HMODULE module, int (*visit)(const char* name, void* context), void* context) {
	const BYTE* base = (const BYTE*)module;
	const IMAGE_NT_HEADERS* nt = (const IMAGE_NT_HEADERS*)(base + ((const IMAGE_DOS_HEADER*)base)->e_lfanew);
	const IMAGE_DATA_DIRECTORY* dir = &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
	const IMAGE_EXPORT_DIRECTORY* exports = (const IMAGE_EXPORT_DIRECTORY*)(base + dir->VirtualAddress);
	const DWORD* names = (const DWORD*)(base + exports->AddressOfNames);
	for (DWORD i = 0; i < exports->NumberOfNames; ++i) {
		if (visit != NULL) visit((const char*)(base + names[i]), context);
	}
	return exports->NumberOfNames;
}

static int VisitExport(const char* name, void* context) {
	const KytyGpuApi* api = (const KytyGpuApi*)context;
	if (strcmp(name, "KytyGpuGetApi") == 0) return 0;
	CHECK(api->find_nid(name) != NULL, "export %s is not in the NID table", name);
	return 0;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		printf("usage: %s <kyty_gpu.dll> [<libwinpthread-1.dll to load first>]\n", argv[0]);
		return 2;
	}
	if (argc >= 3) {
		HMODULE pthread = LoadLibraryA(argv[2]);
		CHECK(pthread != NULL, "preloading %s failed (%lu)", argv[2], GetLastError());
		printf("preloaded %s\n", argv[2]);
	}
	HMODULE dll = LoadLibraryA(argv[1]);
	if (dll == NULL) {
		printf("FAIL: LoadLibrary(%s) failed with error %lu\n", argv[1], GetLastError());
		return 1;
	}
	if (argc >= 3) {
		char loaded[MAX_PATH] = {0};
		GetModuleFileNameA(GetModuleHandleA("libwinpthread-1.dll"), loaded, MAX_PATH);
		printf("libwinpthread-1.dll in the process: %s\n", loaded);
	}

	KytyGpuGetApiFn get_api = (KytyGpuGetApiFn)(void*)GetProcAddress(dll, "KytyGpuGetApi");
	CHECK(get_api != NULL, "KytyGpuGetApi is not exported");
	if (get_api == NULL) return 1;
	CHECK(get_api((KYTY_GPU_ABI_VERSION_MAJOR + 1u) << 16u) == NULL, "an unsupported major version was accepted");
	const KytyGpuApi* api = get_api(KYTY_GPU_ABI_VERSION);
	CHECK(api != NULL, "KytyGpuGetApi(%#x) returned null", KYTY_GPU_ABI_VERSION);
	if (api == NULL) return 1;
	CHECK(api->struct_size == sizeof(KytyGpuApi), "KytyGpuApi is %u bytes in the DLL, %zu here", api->struct_size, sizeof(KytyGpuApi));
	CHECK(api->abi_version == KYTY_GPU_ABI_VERSION, "DLL ABI %#x, header %#x", api->abi_version, KYTY_GPU_ABI_VERSION);
	printf("kyty_gpu %s, ABI %u.%u\n", api->build, api->abi_version >> 16u, api->abi_version & 0xffffu);

	const uint32_t count = api->nid_count();
	CHECK(count == 82, "%u NIDs in the table, expected the 82 graphics imports of AstroBot.exe", count);
	uint32_t per_library[3] = {0, 0, 0};
	for (uint32_t i = 0; i < count; ++i) {
		const KytyGpuNidInfo* info = api->nid_info(i);
		CHECK(info != NULL && info->nid != NULL, "entry %u is empty", i);
		if (info == NULL) continue;
		CHECK(GetProcAddress(dll, info->nid) != NULL, "%s (%s) is not exported", info->nid, info->function);
		CHECK(info->bound == 0, "%s is bound before init", info->nid);
		per_library[strcmp(info->library, "libSceAgc.prx") == 0 ? 0 : strcmp(info->library, "libSceAgcDriver.prx") == 0 ? 1 : 2]++;
	}
	CHECK(per_library[0] == 65 && per_library[1] == 6 && per_library[2] == 11, "per library %u/%u/%u, expected 65/6/11", per_library[0], per_library[1], per_library[2]);
	CHECK(api->nid_info(count) == NULL, "nid_info past the end is not null");
	CHECK(api->find_nid("UglJIZjGssM") != NULL && api->find_nid("notAnExport") == NULL, "find_nid");
	CHECK(CountExports(dll, VisitExport, (void*)api) == count + 1, "the DLL exports %zu names, expected %u NIDs + KytyGpuGetApi", CountExports(dll, NULL, NULL), count);

	KytyGpuHostCallbacks host;
	memset(&host, 0, sizeof(host));
	host.struct_size = sizeof(host);
	host.abi_version = (KYTY_GPU_ABI_VERSION_MAJOR + 1u) << 16u;
	host.user = &g_record;
	host.log = HostLog;
	host.add_gpu_event = HostAddGpuEvent;
	CHECK(api->init(&host) == KYTY_GPU_E_VERSION, "init accepted a host of another major version");
	host.abi_version = KYTY_GPU_ABI_VERSION;
	host.struct_size = 8;
	CHECK(api->init(&host) == KYTY_GPU_E_VERSION, "init accepted a truncated callback struct");
	host.struct_size = sizeof(host);
	const int init = api->init(&host);
	CHECK(init == KYTY_GPU_OK, "init returned %d", init);
	CHECK(api->init(&host) == KYTY_GPU_E_STATE, "a second init was accepted");
	uint32_t bound = 0;
	for (uint32_t i = 0; i < count; ++i) bound += api->nid_info(i)->bound != 0;
	CHECK(bound == count, "%u of %u exports bound after init", bound, count);

	AddEqEventFn add_eq = (AddEqEventFn)(void*)GetProcAddress(dll, "w2rJhmD+dsE");
	int marker = 0;
	const int added = add_eq(0x123456789abcLL, 7, &marker);
	CHECK(added == 42 && g_record.calls == 1 && g_record.equeue == 0x123456789abcLL && g_record.kind == KYTY_GPU_EVENT_EOP_INTERRUPT && g_record.ident == 7 && g_record.udata == &marker,
	      "sceAgcDriverAddEqEvent through the thunk: result %d, %d calls, eq %llx kind %u ident %lld udata %p",
	      added, g_record.calls, (long long)g_record.equeue, g_record.kind, (long long)g_record.ident, g_record.udata);
	AddEqEventFn add_flip = (AddEqEventFn)(void*)GetProcAddress(dll, "HXzjK9yI30k");
	CHECK(add_flip(-5, 1, NULL) == 42 && g_record.calls == 2 && g_record.kind == KYTY_GPU_EVENT_FLIP && g_record.equeue == -5 && g_record.ident == 1, "sceVideoOutAddFlipEvent through the thunk");

	RegisterOwnerFn register_owner = (RegisterOwnerFn)(void*)GetProcAddress(dll, "X-Nm5KLREeg");
	uint32_t owner = 0xdeadbeefu;
	const int registered = register_owner(&owner, "test");
	CHECK((uint32_t)registered == 0x8A6C9018u && owner == 0xdeadbeefu, "sceAgcDriverRegisterOwner returned %#x and wrote %#x", (uint32_t)registered, owner);
	NopGetSizeFn nop_size = (NopGetSizeFn)(void*)GetProcAddress(dll, "t7PlZ9nt5Lc");
	const uint32_t nop_bytes = nop_size(16);
	CHECK(nop_bytes == 64, "sceAgcCbNopGetSize(16) returned %u, expected 64 bytes as AnyPS5 and Kyty both compute", nop_bytes);
	TrinityFn trinity = (TrinityFn)(void*)GetProcAddress(dll, "BfBDZGbti7A");
	uint8_t trinity_mode = 0xaa;
	trinity(&trinity_mode);
	CHECK(trinity_mode == 0 || trinity_mode == 1, "sceAgcGetIsTrinityMode wrote %u", trinity_mode);
	printf("sceAgcCbNopGetSize(16) = %u, sceAgcGetIsTrinityMode = %u\n", nop_bytes, trinity_mode);

	KytyGpuPacket packet = {NULL, 0, 0, {0, 0, 0}};
	CHECK(api->submit_dcb(&packet) == KYTY_GPU_E_STATE, "submit_dcb before start_graphics was accepted");
	CHECK(api->start_graphics() == KYTY_GPU_E_NOT_IMPLEMENTED, "start_graphics did not report it is not implemented yet");
	CHECK(api->register_memory(0x10000, 0x1000, KYTY_GPU_MEMORY_GPU_READ, NULL) == KYTY_GPU_E_NOT_IMPLEMENTED, "register_memory did not report it is not implemented yet");
	api->shutdown();

	printf("%d checks, %d failed\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
