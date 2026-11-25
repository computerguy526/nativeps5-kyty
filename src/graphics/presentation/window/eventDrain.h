// The window loop takes the SDL events of one step: the first (waiting up to the step's timeout)
// and then every one already queued, up to a budget. One event per step (what the loop did) caps the
// rate at the step rate: a stepped loop on a busy main queue runs ~60 steps/s while a DualSense
// sends ~2000 sensor events/s (1000 Hz accel + gyro), so the queue backed up and the game saw
// motion minutes late, in bursts (gyrofix2, 2026-11-25). Pure template: tests/ControllerSettingsTests.cpp.
#pragma once

#include <cstdint>

namespace Libs::Graphics {

constexpr int      kEventDrainBudget   = 4096; // events per step at most
constexpr uint64_t kEventDrainBudgetUs = 2000; // ... and at most this long (a sensor event is ~1 us)

// next(timeout_ms) returns true and fills the event when one is available; handle() processes it;
// now_us() is a monotonic clock. Events are handled in queue order, so a button behind thousands of
// sensor events is still handled in this step (the sensor events are applied one by one, which keeps
// the gyro integration exact; each costs a lock and a few multiplies). Returns the number handled.
template <class Next, class Handle, class Clock>
int DrainEvents(Next&& next, Handle&& handle, Clock&& now_us, int first_timeout_ms,
                int budget = kEventDrainBudget, uint64_t budget_us = kEventDrainBudgetUs) {
	int handled = 0;
	if (!next(first_timeout_ms)) {
		return 0;
	}
	const uint64_t start = now_us();
	handle();
	handled++;
	while (handled < budget && now_us() - start < budget_us && next(0)) {
		handle();
		handled++;
	}
	return handled;
}

} // namespace Libs::Graphics
