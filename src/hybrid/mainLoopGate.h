#ifndef KYTY_HYBRID_MAINLOOPGATE_H_
#define KYTY_HYBRID_MAINLOOPGATE_H_

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

// nps5: the end of a window loop that runs on the host's main queue (kytyGpuBridge.cpp, hybrid
// ABI 1.6 run_on_main_thread). There is no thread to join: shutdown() on another thread waits here
// until the main thread has ended the loop and torn the renderer down, as a join would.
// - The loop side (main thread): Start once; Running before every step; TryBeginEnd / FinishEnd
// around
//   the teardown.
// - The shutdown side (any other thread): WaitEnded. A loop that has not begun to end within
// `grace`
//   (the main thread no longer serves its queue: the process is exiting on it, or it is blocked) is
//   Abandoned: it takes no further step and is never torn down behind the exiting process. A
//   teardown that has begun is waited for without a limit.
// Header only: tests/HybridSeamTests.cpp.
namespace Hybrid {

class MainLoopGate {
public:
	enum class State : uint8_t { Idle, Running, Ending, Ended, Abandoned };

	void Start() {
		std::lock_guard lock(m_mutex);
		if (m_state == State::Idle) {
			m_state = State::Running;
		}
	}

	[[nodiscard]] bool Running() {
		std::lock_guard lock(m_mutex);
		return m_state == State::Running;
	}

	// True once, for the caller that then tears down and calls FinishEnd.
	[[nodiscard]] bool TryBeginEnd() {
		{
			std::lock_guard lock(m_mutex);
			if (m_state != State::Running) {
				return false;
			}
			m_state = State::Ending;
		}
		m_changed.notify_all();
		return true;
	}

	void FinishEnd() {
		{
			std::lock_guard lock(m_mutex);
			m_state = State::Ended;
		}
		m_changed.notify_all();
	}

	// Idle (no loop was started), Ended or Abandoned.
	[[nodiscard]] State WaitEnded(std::chrono::milliseconds grace) {
		std::unique_lock lock(m_mutex);
		if (m_state == State::Idle) {
			return State::Idle;
		}
		m_changed.wait_for(lock, grace, [this] { return m_state != State::Running; });
		if (m_state == State::Running) {
			m_state = State::Abandoned;
			return State::Abandoned;
		}
		m_changed.wait(lock, [this] { return m_state != State::Ending; });
		return m_state;
	}

private:
	std::mutex              m_mutex;
	std::condition_variable m_changed;
	State                   m_state = State::Idle;
};

} // namespace Hybrid

#endif // KYTY_HYBRID_MAINLOOPGATE_H_
