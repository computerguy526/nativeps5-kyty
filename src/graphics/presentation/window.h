#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

namespace Libs::Graphics {

class Presenter;

[[nodiscard]] Presenter& WindowInit(uint32_t width, uint32_t height);
void                     WindowRun();
// Same loop as WindowRun() in resumable steps (macOS hosts run one step per main-queue block).
void                     WindowRunBegin();
[[nodiscard]] bool       WindowRunStep(int max_wait_ms);
void                     WindowRunEnd();
// mainhang: WindowRunEnd gave up on a stuck close-time pipeline cache save
// (KYTY_CLOSE_SAVE_TIMEOUT_MS).
[[nodiscard]] bool       WindowCloseSaveAbandoned();
void                     WindowShutdown();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */
