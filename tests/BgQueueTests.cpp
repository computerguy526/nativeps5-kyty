#include "graphics/host_gpu/renderer/pipeline/bgQueue.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <set>
#include <thread>

using namespace Libs::Graphics::BgQueue;

static int g_failed = 0;
static int g_checks = 0;
#define CHECK(cond)                                                       \
	do {                                                                  \
		g_checks++;                                                       \
		if (!(cond)) {                                                    \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			g_failed++;                                                   \
		}                                                                 \
	} while (0)

static std::vector<Job> Jobs(uint32_t n, uint32_t scenes) {
	std::vector<Job> v;
	for (uint32_t i = 0; i < n; i++) v.push_back({1000 + i, i, i % scenes});
	return v;
}

// A pool of fake compile workers: each takes jobs from the queue, "compiles" for `ms`, records the order and the peak
// concurrency, sleeps while the queue says Wait.
struct Pool {
	explicit Pool(Queue& queue): q(queue) {}
	Queue&                 q;
	std::vector<uint64_t>  order;
	std::mutex             m;
	std::atomic<unsigned>  live {0}, peak {0};
	std::atomic<bool>      stop {false};
	std::vector<std::thread> threads;
	int                    ms = 2;
	void Start(unsigned n) {
		for (unsigned i = 0; i < n; i++) {
			threads.emplace_back([this] {
				while (!stop) {
					Job j;
					const auto r = q.Acquire(j);
					if (r == Result::Done) break;
					if (r == Result::Wait) {
						std::this_thread::sleep_for(std::chrono::milliseconds(1));
						continue;
					}
					const auto now = ++live;
					unsigned   p   = peak;
					while (now > p && !peak.compare_exchange_weak(p, now)) {}
					std::this_thread::sleep_for(std::chrono::milliseconds(ms));
					{
						std::scoped_lock l(m);
						order.push_back(j.id);
					}
					--live;
					q.Complete(j);
				}
			});
		}
	}
	void Join() {
		for (auto& t: threads) t.join();
		threads.clear();
	}
};

int main() {
	// ---- ordering with one worker is the job order ----
	{
		Queue q;
		q.Reset(Jobs(20, 4), 1);
		Pool p(q);
		p.Start(1);
		p.Join();
		CHECK(p.order.size() == 20);
		bool sorted = true;
		for (size_t i = 0; i < p.order.size(); i++) sorted = sorted && p.order[i] == 1000 + i;
		CHECK(sorted && q.Finished() && q.Completed() == 20);
	}
	// ---- budget: 8 threads, N workers allowed -> peak concurrency never above N ----
	for (unsigned workers = 1; workers <= 3; workers++) {
		Queue q;
		q.Reset(Jobs(60, 5), workers, 0);
		Pool p(q);
		p.Start(8);
		p.Join();
		CHECK(p.order.size() == 60);
		CHECK(p.peak <= workers);
		CHECK(p.peak == workers); // and the budget is used (60 jobs of 2 ms)
	}
	// ---- every job runs exactly once ----
	{
		Queue q;
		q.Reset(Jobs(200, 7), 3);
		Pool p(q);
		p.ms = 0;
		p.Start(6);
		p.Join();
		std::set<uint64_t> seen(p.order.begin(), p.order.end());
		CHECK(p.order.size() == 200 && seen.size() == 200);
	}
	// ---- acquire semantics ----
	{
		Queue q;
		Job   a, b, c;
		q.Reset(Jobs(3, 1), 2, 0);
		CHECK(q.Acquire(a) == Result::Job && a.id == 1000);
		CHECK(q.Acquire(b) == Result::Job && b.id == 1001);
		CHECK(q.Acquire(c) == Result::Wait && q.Running() == 2); // over budget
		q.Complete(a);
		CHECK(q.Acquire(c) == Result::Job && c.id == 1002);
		CHECK(q.Acquire(a) == Result::Done && q.Pending() == 0); // nothing left to start (b, c still running)
		CHECK(!q.Finished());
		q.Complete(b);
		q.Complete(c);
		CHECK(q.Finished() && q.Completed() == 3);
		// empty queue
		q.Reset({}, 2);
		CHECK(q.Acquire(a) == Result::Done && q.Finished());
		// worker count: 0 means 1, and SetWorkers applies at once
		q.Reset(Jobs(5, 1), 0, 0);
		CHECK(q.Budget() == 1);
		q.SetWorkers(3);
		CHECK(q.Budget() == 3);
	}
	// ---- promotion ----
	{
		Queue q;
		q.Reset(Jobs(12, 3), 1, 1); // scenes: i % 3
		CHECK(q.Promote(2) == 4);   // jobs 2,5,8,11
		Job j;
		std::vector<uint64_t> got;
		CHECK(q.Budget() == 2); // boosted while promoted jobs are pending
		q.Reset(Jobs(12, 3), 1, 1);
		q.Promote(2);
		std::vector<Job> taken;
		while (true) {
			const auto r = q.Acquire(j);
			if (r == Result::Done) break;
			if (r == Result::Wait) {
				q.Complete(taken.front());
				taken.erase(taken.begin());
				continue;
			}
			taken.push_back(j);
			got.push_back(j.id);
		}
		CHECK(got.size() == 12);
		CHECK((got[0] == 1002 && got[1] == 1005 && got[2] == 1008 && got[3] == 1011)); // promoted scene first, in order
		CHECK((got[4] == 1000 && got[5] == 1001 && got[6] == 1003)); // the rest keep their order
		// the boost ends when the promoted scene is done
		Queue r;
		r.Reset(Jobs(6, 3), 1, 2);
		CHECK(r.Budget() == 1);
		CHECK(r.Promote(1) == 2 && r.Budget() == 3);
		CHECK(r.Promote(1) == 2); // again: same jobs still front, no double boost
		Job a, b;
		CHECK(r.Acquire(a) == Result::Job && a.scene == 1);
		CHECK(r.Acquire(b) == Result::Job && b.scene == 1);
		r.Complete(a);
		CHECK(r.Budget() == 3); // one promoted job still running
		r.Complete(b);
		CHECK(r.Budget() == 1);
		CHECK(r.Promote(1) == 0); // nothing pending of it any more
		CHECK(r.Promote(99) == 0);
		// promotion of a scene whose jobs are all running keeps the boost until they finish
		Queue s;
		s.Reset(Jobs(2, 1), 2, 1);
		Job x, y;
		CHECK(s.Acquire(x) == Result::Job && s.Acquire(y) == Result::Job);
		CHECK(s.Promote(0) == 0 && s.Budget() == 2);
		// pending + promoted: paused queue still serves promoted jobs
		Queue t;
		t.Reset(Jobs(6, 2), 1, 1);
		t.SetPaused(true);
		CHECK(t.Acquire(x) == Result::Wait);
		t.Promote(1);
		CHECK(t.Acquire(x) == Result::Job && x.scene == 1);
		t.Complete(x);
	}
	// ---- promotion by id (a scene's keys, whichever scene each job was filed under) ----
	{
		Queue q;
		q.Reset(Jobs(10, 5), 1, 1);
		const std::unordered_set<uint64_t> ids = {1009, 1004, 1007, 5555};
		CHECK(q.PromoteIds(ids, 3) == 3 && q.PromotedPending() && q.Budget() == 2);
		Job j;
		std::vector<uint64_t> got;
		CHECK(q.Acquire(j) == Result::Job);
		got.push_back(j.id);
		CHECK(q.Acquire(j) == Result::Job); // boosted budget of 2
		got.push_back(j.id);
		CHECK(q.Acquire(j) == Result::Wait);
		CHECK(got[0] == 1004 && got[1] == 1007); // the job order inside the promoted set is kept
		q.Complete(Job {1004, 4, 4});
		CHECK(q.Acquire(j) == Result::Job && j.id == 1009);
		q.Complete(Job {1007, 7, 2});
		q.Complete(j);
		CHECK(!q.PromotedPending() && q.Budget() == 1);
		CHECK(q.Acquire(j) == Result::Job && j.id == 1000); // back to the plain order
		CHECK(q.PromoteIds({}, 1) == 0 && q.PromoteIds({424242}, 1) == 0);
	}
	// ---- pause / resume ----
	{
		Queue q;
		q.Reset(Jobs(10, 2), 2, 0);
		Pool p(q);
		q.SetPaused(true);
		CHECK(q.Paused());
		p.Start(3);
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
		CHECK(q.Completed() == 0 && q.Running() == 0 && q.Pending() == 10); // nothing starts while paused
		q.SetPaused(false);
		p.Join();
		CHECK(q.Completed() == 10);
		// a pause mid-run lets the running jobs finish and starts no new one
		Queue q2;
		q2.Reset(Jobs(40, 2), 2, 0);
		Pool p2(q2);
		p2.ms = 5;
		p2.Start(2);
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		q2.SetPaused(true);
		std::this_thread::sleep_for(std::chrono::milliseconds(25));
		const auto at_pause = q2.Completed();
		std::this_thread::sleep_for(std::chrono::milliseconds(40));
		CHECK(q2.Completed() == at_pause && q2.Running() == 0 && q2.Pending() > 0);
		q2.SetPaused(false);
		p2.Join();
		CHECK(q2.Completed() == 40);
	}
	// ---- guard: pure verdicts ----
	{
		Sample s;
		s.now_ns          = 10'000'000'000ull;
		s.last_frame_ms   = 16;
		s.median_frame_ms = 16;
		CHECK(Allows(s));
		s.last_frame_ms = 19.9; // 1.24x
		CHECK(Allows(s));
		s.last_frame_ms = 20.5; // above 1.25x
		CHECK(!Allows(s));
		s.calm_scene = true; // the hub is never throttled
		CHECK(Allows(s));
		s.calm_scene    = false;
		s.last_frame_ms = 16;
		s.last_slow_ns  = s.now_ns - 1'000'000'000ull; // slow frame 1 s ago
		CHECK(!Allows(s));
		s.last_slow_ns = s.now_ns - 3'000'000'000ull;
		CHECK(Allows(s));
		s.last_load_ns = s.now_ns - 500'000'000ull; // loading sign
		CHECK(!Allows(s));
		s.promoted = true; // the load is the one being promoted for
		CHECK(Allows(s));
		s.promoted     = false;
		s.last_load_ns = s.now_ns - 5'000'000'000ull;
		CHECK(Allows(s));
		s.median_frame_ms = 0; // unknown median: no ratio test
		s.last_frame_ms   = 500;
		CHECK(Allows(s));
		s.last_load_ns = s.now_ns + 1; // a clock that went backwards must not pause forever
		CHECK(Allows(s));
	}
	// ---- guard: hysteresis ----
	{
		Guard  g;
		Sample s;
		s.median_frame_ms = 16;
		s.last_frame_ms   = 16;
		s.now_ns          = 1'000'000'000ull;
		CHECK(g.Update(s) && !g.Paused());
		s.last_frame_ms = 40;
		s.now_ns += 16'000'000;
		CHECK(!g.Update(s) && g.Paused());
		s.last_frame_ms = 16;
		s.now_ns += 16'000'000;
		CHECK(!g.Update(s)); // calm, but not for long enough
		s.now_ns += 300'000'000;
		CHECK(!g.Update(s));
		s.now_ns += 300'000'000;
		CHECK(g.Update(s) && !g.Paused()); // 0.6 s calm: resumed
		s.last_frame_ms = 40;
		s.now_ns += 1'000'000;
		CHECK(!g.Update(s));
		s.last_frame_ms = 16;
		s.now_ns += 400'000'000;
		CHECK(!g.Update(s));
		s.last_frame_ms = 40; // a spike inside the calm window restarts it
		s.now_ns += 1'000'000;
		CHECK(!g.Update(s));
		s.last_frame_ms = 16;
		s.now_ns += 400'000'000;
		CHECK(!g.Update(s));
	}
	// ---- median ----
	{
		FrameMedian m(5);
		CHECK(m.Median() == 0);
		for (double v: {10.0, 20.0, 30.0}) m.Add(v);
		CHECK(m.Median() == 20);
		for (double v: {100.0, 100.0, 100.0}) m.Add(v); // window of 5: 100,100,100 + two of the old
		CHECK(m.Median() == 100);
		FrameMedian one(0);
		one.Add(7);
		one.Add(9);
		CHECK(one.Median() == 9);
	}
	// ---- worker count text ----
	{
		CHECK(WorkersFromText(nullptr) == 2 && WorkersFromText("") == 2 && WorkersFromText("0") == 2 && WorkersFromText("x") == 2);
		CHECK(WorkersFromText("1") == 1 && WorkersFromText("3") == 3 && WorkersFromText("2") == 2);
		CHECK(WorkersFromText("99", 2, 8) == 8 && WorkersFromText("99999999999") == 2);
		CHECK(WorkersFromText("-1") == 2 && WorkersFromText("2x") == 2);
	}
	std::printf("BgQueueTests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
