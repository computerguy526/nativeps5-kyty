/* Test host module for kytyGpuLoadTest --lazy: stands in for AnyPS5's libAstroHybridHost.prx.
 * kyty_gpu.dll loads it by name when game code reaches an export before any init, and binds
 * through these callbacks. Built with MinGW GCC like the real host. Guest range donation is left
 * out: the test process has no AnyPS5 guest arena, so Kyty reserves the free space itself. */

#include "../../src/hybrid/kytyGpuApi.h"

#include <stdio.h>
#include <windows.h>

static volatile LONG g_added;
static volatile LONG g_delivered;

static void HostLog(void* user, int level, const char* message) {
	(void)user;
	printf("  [lazy %d] %s\n", level, message);
}

static int HostAdd(void* user, int64_t equeue, uint32_t kind, int64_t ident, void* udata) {
	(void)user;
	(void)equeue;
	(void)kind;
	(void)ident;
	(void)udata;
	InterlockedIncrement(&g_added);
	return 0;
}

static int HostDelete(void* user, int64_t equeue, uint32_t kind, int64_t ident) {
	(void)user;
	(void)equeue;
	(void)kind;
	(void)ident;
	return 0;
}

static void HostDeliver(void* user, uint32_t kind, int64_t ident, uint64_t data) {
	(void)user;
	(void)kind;
	(void)ident;
	(void)data;
	InterlockedIncrement(&g_delivered);
}

static KytyGpuHostCallbacks g_callbacks = {
    sizeof(KytyGpuHostCallbacks), KYTY_GPU_ABI_VERSION, NULL, HostLog, HostAdd, HostDelete, HostDeliver, NULL, NULL, NULL, NULL, NULL,
};

__declspec(dllexport) const KytyGpuHostCallbacks* AstroHybridHostCallbacks_nid_no_patch(void) {
	return &g_callbacks;
}

__declspec(dllexport) long TestHostAdded(void) {
	return g_added;
}
