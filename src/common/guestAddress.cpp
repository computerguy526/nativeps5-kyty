#include "common/guestAddress.h"

namespace Guest {

uint64_t g_bias                                                    = 0;
uint64_t (*g_call_guest)(uint64_t function, const uint64_t* args6) = nullptr;

} // namespace Guest
