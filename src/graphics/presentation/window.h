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
void                     WindowShutdown();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */
