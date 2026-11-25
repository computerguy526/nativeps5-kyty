// The SDL gamepad sensor event -> Kyty pad (window.cpp ProcessEvent and tests/PadMotionE2ETests.cpp).
#pragma once

#include <SDL3/SDL.h>

#include "libs/controller.h"

namespace Libs::Graphics {

inline void DispatchGamepadSensor(const SDL_GamepadSensorEvent& sensor) {
	if (sensor.sensor == SDL_SENSOR_ACCEL || sensor.sensor == SDL_SENSOR_GYRO) {
		Controller::SetSensor(sensor.which,
		                      sensor.sensor == SDL_SENSOR_ACCEL ? Controller::Sensor::Accel
		                                                        : Controller::Sensor::Gyro,
		                      sensor.data,
		                      sensor.sensor_timestamp != 0 ? sensor.sensor_timestamp / 1000
		                                                   : sensor.timestamp / 1000);
	}
}

} // namespace Libs::Graphics
