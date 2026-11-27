// Shader precompile journal and replay driver (graphics/host_gpu/renderer/pipeline/shaderPrecompile.h):
// the journal file round-trips, deduplicates, survives a damaged tail, is replaced for another
// identity and respects its size cap; the replay driver visits every entry once, in order on one
// thread, reports progress and stops.
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("ShaderPrecompileTests: failed: %s\n", what);
		g_failures++;
	}
}

std::filesystem::path TestPath(const char* name) {
	return std::filesystem::current_path() / (std::string("shader-precompile-test-") + name + ".bin");
}

std::vector<uint8_t> Identity(const char* text) {
	return std::vector<uint8_t>(text, text + std::char_traits<char>::length(text));
}

ShaderJournal::Settings MakeSettings(const std::filesystem::path& path, const char* identity = "device A") {
	ShaderJournal::Settings settings;
	settings.path              = path;
	settings.identity          = Identity(identity);
	settings.background_writer = false;
	return settings;
}

ShaderJournal::Source MakeSource(uint64_t hash, uint32_t words, ShaderJournal::Kind kind = ShaderJournal::Kind::Pixel) {
	ShaderJournal::Source source;
	source.stage           = kind == ShaderJournal::Kind::Compute ? 4u : kind == ShaderJournal::Kind::Pixel ? 2u : 1u;
	source.kind            = kind;
	source.hash            = hash;
	source.user_data_count = 12;
	source.code_size       = words;
	source.wave_size       = 64;
	source.user_data_base  = 0;
	source.static_state    = {1, 2, static_cast<uint32_t>(hash)};
	source.code.resize(words);
	for (uint32_t i = 0; i < words; i++) source.code[i] = static_cast<uint32_t>(hash) + i;
	source.input_info.assign(40, static_cast<uint8_t>(hash));
	return source;
}

std::vector<uint8_t> Spec(uint8_t value, size_t size = 9) {
	return std::vector<uint8_t>(size, value);
}

void Remove(const std::filesystem::path& path) {
	std::error_code error;
	std::filesystem::remove(path, error);
}

std::vector<uint8_t> ReadAll(const std::filesystem::path& path) {
	std::ifstream in(path, std::ios::binary);
	return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void WriteAll(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void TestRoundTripAndDedup() {
	const auto path = TestPath("roundtrip");
	Remove(path);
	{
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.Sources().empty() && journal.Entries().empty(), "a missing file loads empty");
		Expect(!journal.GetStats().file_found, "a missing file is not found");
		auto a = MakeSource(0x11, 8);
		Expect(!journal.HasSource(a), "an unknown source is not known");
		journal.Record(a, 0, Spec(1));
		Expect(journal.HasSource(a), "a recorded source is known");
		journal.Record(a, 0, Spec(1)); // duplicate permutation
		journal.Record(a, 0, Spec(2)); // other specialization
		journal.Record(a, 3, Spec(1)); // other cursor
		auto identity_only = MakeSource(0x11, 8);
		identity_only.code.clear(); // a known source needs no code
		identity_only.input_info.clear();
		journal.Record(identity_only, 5, Spec(3));
		journal.Record(MakeSource(0x22, 4, ShaderJournal::Kind::Vertex), 0, Spec(4));
		auto no_code = MakeSource(0x33, 4);
		no_code.code.clear();
		journal.Record(no_code, 0, Spec(5)); // a new source without code is dropped
		Expect(!journal.HasSource(no_code), "a source without code is not recorded");
		auto bad_size = MakeSource(0x44, 4);
		bad_size.code_size = 5;
		journal.Record(bad_size, 0, Spec(5));
		Expect(!journal.HasSource(bad_size), "a source whose code size disagrees is not recorded");
		Expect(journal.Flush(), "flush succeeds");
		const auto stats = journal.GetStats();
		Expect(stats.recorded_sources == 2 && stats.recorded_entries == 5, "recorded counts");
	}
	{
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.GetStats().file_found && !journal.GetStats().header_rejected, "file loads");
		Expect(journal.Sources().size() == 2 && journal.Entries().size() == 5, "loaded counts");
		const auto& s = journal.Sources()[0];
		const auto expected = MakeSource(0x11, 8);
		Expect(s.hash == 0x11 && s.stage == expected.stage && s.kind == expected.kind &&
		           s.code == expected.code && s.input_info == expected.input_info &&
		           s.static_state == expected.static_state && s.code_size == 8 &&
		           s.user_data_count == 12 && s.wave_size == 64,
		       "source fields round-trip");
		Expect(journal.Sources()[1].kind == ShaderJournal::Kind::Vertex, "kind round-trips");
		const auto& e = journal.Entries();
		Expect(e[0].source == 0 && e[0].push_data_cursor == 0 && e[0].specialization == Spec(1), "entry 0");
		Expect(e[1].specialization == Spec(2) && e[2].push_data_cursor == 3 && e[3].push_data_cursor == 5 &&
		           e[4].source == 1,
		       "entries keep their order");
		// Loaded content is known: nothing is journaled twice across runs.
		Expect(journal.HasSource(expected), "a loaded source is known");
		journal.Record(expected, 0, Spec(1));
		journal.Record(expected, 0, Spec(7)); // new
		journal.Flush();
		Expect(journal.GetStats().recorded_entries == 1 && journal.GetStats().recorded_sources == 0,
		       "only the new permutation is recorded");
	}
	{
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.Entries().size() == 6 && journal.Entries()[5].specialization == Spec(7),
		       "a later run appends");
	}
	Remove(path);
}

void TestDamagedFile() {
	const auto path = TestPath("damaged");
	Remove(path);
	{
		ShaderJournal journal(MakeSettings(path));
		for (uint32_t i = 0; i < 4; i++) {
			journal.Record(MakeSource(0x100 + i, 6), 0, Spec(static_cast<uint8_t>(i)));
		}
		journal.Flush();
	}
	const auto intact = ReadAll(path);
	// Truncated tail: the intact prefix loads, and a later record continues after it.
	for (const size_t cut: {size_t {1}, size_t {30}, size_t {80}}) {
		auto bytes = intact;
		bytes.resize(bytes.size() - cut);
		WriteAll(path, bytes);
		{
			ShaderJournal journal(MakeSettings(path));
			const auto stats = journal.GetStats();
			Expect(stats.damaged_bytes != 0, "a truncated tail is reported");
			Expect(journal.Entries().size() < 4 && journal.Entries().size() >= 2, "the intact prefix loads");
			for (const auto& entry: journal.Entries()) {
				Expect(entry.source < journal.Sources().size(), "entries refer to loaded sources");
			}
			journal.Record(MakeSource(0x900, 3), 1, Spec(9));
			journal.Flush();
		}
		ShaderJournal again(MakeSettings(path));
		Expect(again.GetStats().damaged_bytes == 0, "the damaged tail was cut off by the next write");
		Expect(!again.Entries().empty() && again.Entries().back().specialization == Spec(9),
		       "the record after the truncation is kept");
	}
	// A flipped byte in a payload: records before it load, it and everything after do not.
	{
		auto bytes = intact;
		bytes[bytes.size() / 2] ^= 0xff;
		WriteAll(path, bytes);
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.GetStats().damaged_bytes != 0 && journal.Entries().size() < 4, "corruption is detected");
	}
	// Garbage and an empty file load as nothing, without a crash.
	WriteAll(path, std::vector<uint8_t>(100, 0xab));
	{
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.Entries().empty() && journal.GetStats().header_rejected, "garbage is rejected");
	}
	WriteAll(path, {});
	{
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.Entries().empty(), "an empty file loads empty");
	}
	Remove(path);
}

void TestIdentity() {
	const auto path = TestPath("identity");
	Remove(path);
	{
		ShaderJournal journal(MakeSettings(path, "device A"));
		journal.Record(MakeSource(0x1, 4), 0, Spec(1));
		journal.Flush();
	}
	{
		ShaderJournal journal(MakeSettings(path, "device B"));
		Expect(journal.Entries().empty() && journal.GetStats().header_rejected, "another identity is rejected");
		Expect(!journal.HasSource(MakeSource(0x1, 4)), "a rejected file's sources are not known");
		journal.Record(MakeSource(0x2, 4), 0, Spec(2));
		journal.Flush();
	}
	{
		ShaderJournal journal(MakeSettings(path, "device B"));
		Expect(journal.Entries().size() == 1 && journal.Sources()[0].hash == 0x2,
		       "the file was replaced by the new identity's records");
	}
	{
		ShaderJournal journal(MakeSettings(path, "device A"));
		Expect(journal.Entries().empty(), "the old identity's records are gone");
	}
	Remove(path);
}

void TestCapacity() {
	const auto path = TestPath("capacity");
	Remove(path);
	auto settings = MakeSettings(path);
	settings.max_file_bytes = 600;
	{
		ShaderJournal journal(settings);
		for (uint32_t i = 0; i < 20; i++) journal.Record(MakeSource(0x500 + i, 16), 0, Spec(1));
		journal.Flush();
		Expect(journal.GetStats().over_capacity, "the cap is reached");
		Expect(std::filesystem::file_size(path) <= 600 + 200, "the file stays near the cap");
	}
	{
		ShaderJournal journal(settings);
		Expect(!journal.Entries().empty() && journal.Entries().size() < 20, "a capped file loads what fit");
	}
	Remove(path);
}

void TestBackgroundWriter() {
	const auto path = TestPath("writer");
	Remove(path);
	auto settings = MakeSettings(path);
	settings.background_writer = true;
	settings.flush_interval_ns = 20'000'000ull;
	{
		ShaderJournal journal(settings);
		journal.Record(MakeSource(0x7, 4), 0, Spec(1));
		for (int i = 0; i < 200 && !std::filesystem::exists(path); i++) {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		Expect(std::filesystem::exists(path), "the writer thread wrote the file");
		journal.Record(MakeSource(0x8, 4), 0, Spec(2)); // written by the destructor at the latest
	}
	ShaderJournal journal(MakeSettings(path));
	Expect(journal.Entries().size() == 2, "records written by the thread and the destructor");
	Remove(path);
}

ShaderJournal MakeJournalWith(const std::filesystem::path& path, uint32_t entries) {
	Remove(path);
	{
		ShaderJournal journal(MakeSettings(path));
		for (uint32_t i = 0; i < entries; i++) {
			journal.Record(MakeSource(0x1000 + i % 7, 4 + i % 7), i, Spec(static_cast<uint8_t>(i)));
		}
		journal.Flush();
	}
	return ShaderJournal(MakeSettings(path));
}

void TestReplayDriver() {
	const auto path = TestPath("replay");
	auto       journal = MakeJournalWith(path, 100);
	Expect(journal.Entries().size() == 100, "journal for replay");

	// One thread: every entry once, in file order.
	{
		std::vector<uint32_t> order;
		std::vector<std::string> lines;
		ShaderPrecompiler::Settings settings;
		settings.threads = 1;
		settings.log     = [&](const std::string& line) { lines.push_back(line); };
		settings.progress_percent = 25;
		std::atomic<int> finished {0};
		settings.on_finished = [&] { finished++; };
		ShaderPrecompiler precompiler(journal, settings, [&](const auto& source, const auto& entry) {
			order.push_back(entry.push_data_cursor);
			Expect(source.code.size() == source.code_size, "replay gets the entry's source");
			if (entry.push_data_cursor == 0) return ShaderPrecompiler::Outcome::Reloaded;
			return entry.push_data_cursor % 4 == 0   ? ShaderPrecompiler::Outcome::Compiled
			       : entry.push_data_cursor % 4 == 1 ? ShaderPrecompiler::Outcome::Present
			       : entry.push_data_cursor % 4 == 2 ? ShaderPrecompiler::Outcome::Skipped
			                                         : ShaderPrecompiler::Outcome::Failed;
		});
		precompiler.Wait();
		const auto progress = precompiler.GetProgress();
		Expect(progress.finished && progress.total == 100 && progress.done == 100, "progress totals");
		Expect(progress.compiled == 24 && progress.reloaded == 1 && progress.present == 25 && progress.skipped == 25 &&
		           progress.failed == 25,
		       "outcome counts");
		bool in_order = order.size() == 100;
		for (uint32_t i = 0; i < order.size(); i++) in_order = in_order && order[i] == i;
		Expect(in_order, "one thread replays in file order");
		Expect(finished == 1, "on_finished runs once");
		Expect(lines.size() >= 3 && lines.back().find("done") != std::string::npos &&
		           lines.back().find("100/100") != std::string::npos,
		       "progress lines and a final line");
		Expect(lines.size() >= 2 && lines[1].find("%)") != std::string::npos, "a progress line has a percentage");
	}
	// Several threads: every entry exactly once.
	{
		std::mutex mutex;
		std::multiset<uint32_t> seen;
		ShaderPrecompiler::Settings settings;
		settings.threads = 4;
		ShaderPrecompiler precompiler(journal, settings, [&](const auto&, const auto& entry) {
			std::scoped_lock lock(mutex);
			seen.insert(entry.push_data_cursor);
			return ShaderPrecompiler::Outcome::Compiled;
		});
		precompiler.Wait();
		bool each_once = seen.size() == 100;
		for (uint32_t i = 0; i < 100; i++) each_once = each_once && seen.count(i) == 1;
		Expect(each_once, "four threads replay every entry exactly once");
	}
	// max_entries, a throwing callback.
	{
		std::atomic<int> calls {0};
		ShaderPrecompiler::Settings settings;
		settings.threads     = 2;
		settings.max_entries = 10;
		ShaderPrecompiler precompiler(journal, settings, [&](const auto&, const auto&) -> ShaderPrecompiler::Outcome {
			calls++;
			throw std::runtime_error("compile failed");
		});
		precompiler.Wait();
		Expect(calls == 10 && precompiler.GetProgress().failed == 10 && precompiler.GetProgress().total == 10,
		       "max_entries limits the replay; an exception counts as a failure");
	}
	// Stop: entries not started are dropped, the destructor returns.
	{
		std::atomic<int> calls {0};
		std::atomic<bool> release {false};
		std::vector<std::string> lines;
		ShaderPrecompiler::Settings settings;
		settings.threads = 1;
		settings.log     = [&](const std::string& line) { lines.push_back(line); };
		ShaderPrecompiler precompiler(journal, settings, [&](const auto&, const auto&) {
			calls++;
			while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
			return ShaderPrecompiler::Outcome::Compiled;
		});
		while (calls == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		precompiler.Stop();
		release = true;
		precompiler.Wait();
		Expect(calls == 1 && precompiler.GetProgress().done == 1, "Stop drops entries not started");
		Expect(!lines.empty() && lines.back().find("stopped") != std::string::npos, "a stopped replay says so");
	}
	// used_entry: the used entries first (file order), then the rest; replay_unused=false drops the rest.
	{
		for (const bool unused: {true, false}) {
			std::vector<uint32_t> order;
			ShaderPrecompiler::Settings settings;
			settings.threads       = 1;
			settings.used_entry    = [](const auto&, const auto& entry) { return entry.push_data_cursor % 10 == 3; };
			settings.replay_unused = unused;
			ShaderPrecompiler precompiler(journal, settings, [&](const auto&, const auto& entry) {
				order.push_back(entry.push_data_cursor);
				return ShaderPrecompiler::Outcome::Reloaded;
			});
			precompiler.Wait();
			bool used_first = order.size() == (unused ? 100u : 10u);
			for (uint32_t i = 0; used_first && i < 10; i++) used_first = order[i] == i * 10 + 3;
			for (uint32_t i = 10, next = 0; used_first && i < order.size(); i++, next++) {
				if (next % 10 == 3) next++;
				used_first = order[i] == next;
			}
			Expect(used_first && precompiler.GetProgress().total == order.size(),
			       unused ? "used entries replay first, then the rest in file order"
			              : "replay_unused=false replays only the used entries");
		}
	}
	// An empty journal finishes at once.
	{
		const auto empty_path = TestPath("replay-empty");
		Remove(empty_path);
		ShaderJournal empty(MakeSettings(empty_path));
		ShaderPrecompiler precompiler(empty, {}, [](const auto&, const auto&) {
			return ShaderPrecompiler::Outcome::Compiled;
		});
		precompiler.Wait();
		Expect(precompiler.GetProgress().finished && precompiler.GetProgress().total == 0, "nothing to replay");
	}
	Remove(path);
}

PipelineJournal::Settings MakePipelineSettings(const std::filesystem::path& path,
                                               const char* identity = "device A", bool sealed = false) {
	PipelineJournal::Settings settings;
	settings.path              = path;
	settings.identity          = Identity(identity);
	settings.background_writer = false;
	settings.sealed_writes     = sealed;
	return settings;
}

// Pipeline journal: keyed records round-trip in first-recorded order, a key is kept once (also
// across runs), a damaged tail keeps the intact prefix, another identity starts over, the cap holds.
void TestPipelineJournal() {
	const auto path = TestPath("pipelines");
	Remove(path);
	{
		PipelineJournal journal(MakePipelineSettings(path));
		Expect(journal.Loaded().empty() && !journal.GetStats().file_found, "pipeline journal: missing file loads empty");
		Expect(journal.Add(0x10, Spec(1, 40)), "pipeline journal: a new key is recorded");
		Expect(!journal.Add(0x10, Spec(2, 40)), "pipeline journal: a known key is not recorded again");
		Expect(journal.Add(0x20), "pipeline journal: an empty payload is fine");
		Expect(journal.Add(0x30, Spec(3, 7)), "pipeline journal: third key");
		Expect(journal.Contains(0x20) && !journal.Contains(0x40), "pipeline journal: contains");
		Expect(journal.Flush(), "pipeline journal: flush");
	}
	{
		PipelineJournal journal(MakePipelineSettings(path));
		const auto& loaded = journal.Loaded();
		Expect(loaded.size() == 3 && loaded[0].key == 0x10 && loaded[0].payload == Spec(1, 40) &&
		           loaded[1].key == 0x20 && loaded[1].payload.empty() && loaded[2].payload == Spec(3, 7),
		       "pipeline journal: records round-trip in order");
		Expect(!journal.Add(0x10, Spec(9, 3)), "pipeline journal: loaded keys are known");
		Expect(journal.Add(0x40, Spec(4, 5)), "pipeline journal: a later run appends");
		journal.ReleaseLoaded();
		Expect(journal.Loaded().empty() && journal.Contains(0x10), "pipeline journal: released content stays known");
	}
	// Damage: a flipped payload byte in the last record drops it and keeps the rest.
	auto bytes = ReadAll(path);
	bytes[bytes.size() - 2] ^= 0xffu;
	WriteAll(path, bytes);
	{
		PipelineJournal journal(MakePipelineSettings(path));
		Expect(journal.Loaded().size() == 3 && journal.GetStats().damaged_bytes != 0,
		       "pipeline journal: damaged tail ignored");
		Expect(journal.Add(0x50, Spec(5, 2)) && journal.Flush(), "pipeline journal: append after damage");
	}
	{
		PipelineJournal journal(MakePipelineSettings(path));
		Expect(journal.Loaded().size() == 4 && journal.Loaded()[3].key == 0x50,
		       "pipeline journal: the damaged tail was cut before appending");
	}
	{
		PipelineJournal journal(MakePipelineSettings(path, "device B"));
		Expect(journal.Loaded().empty() && journal.GetStats().header_rejected,
		       "pipeline journal: another identity is rejected");
		journal.Add(0x60);
		journal.Flush();
	}
	{
		PipelineJournal journal(MakePipelineSettings(path, "device B"));
		Expect(journal.Loaded().size() == 1 && journal.Loaded()[0].key == 0x60,
		       "pipeline journal: the replaced file holds only the new identity's records");
	}
	// Capacity.
	Remove(path);
	{
		auto settings           = MakePipelineSettings(path);
		settings.max_file_bytes = 400;
		PipelineJournal journal(std::move(settings));
		uint64_t accepted = 0;
		for (uint64_t key = 1; key <= 20; key++) accepted += journal.Add(key, Spec(1, 32)) ? 1 : 0;
		Expect(accepted > 0 && accepted < 20 && journal.GetStats().over_capacity, "pipeline journal: size cap");
		Expect(!journal.Contains(20), "pipeline journal: a refused key is not known");
		journal.Flush();
	}
	Expect(std::filesystem::file_size(path) <= 400, "pipeline journal: the file stays under the cap");
	Remove(path);
}

// Known-pipeline list mode: Flush writes only what was recorded before Seal (the keys the saved
// driver cache can hold); Reset empties the file at once.
void TestPipelineJournalSealed() {
	const auto path = TestPath("known");
	Remove(path);
	{
		PipelineJournal journal(MakePipelineSettings(path, "driver 1", true));
		journal.Add(1);
		journal.Add(2);
		journal.Seal();
		journal.Add(3); // created while the driver cache was serialized: not in the saved file
		Expect(journal.Flush(), "known list: flush");
		journal.Seal();
	} // no write at destruction in sealed mode
	{
		PipelineJournal journal(MakePipelineSettings(path, "driver 1", true));
		Expect(journal.Loaded().size() == 2 && journal.Contains(1) && journal.Contains(2) && !journal.Contains(3),
		       "known list: only sealed keys were written");
		journal.Add(3);
		journal.Seal();
		journal.Flush();
	}
	{
		PipelineJournal journal(MakePipelineSettings(path, "driver 1", true));
		Expect(journal.Loaded().size() == 3, "known list: a later save appends");
		journal.Reset();
		Expect(!journal.Contains(1) && journal.KnownCount() == 0, "known list: reset forgets every key");
	}
	{
		PipelineJournal journal(MakePipelineSettings(path, "driver 1", true));
		Expect(journal.Loaded().empty() && !journal.GetStats().header_rejected,
		       "known list: reset leaves an empty file at once");
	}
	{
		PipelineJournal journal(MakePipelineSettings(path, "driver 2", true));
		Expect(journal.GetStats().header_rejected, "known list: another driver is rejected");
	}
	Remove(path);
	// In memory only (an empty path): keys are kept, nothing is read or written.
	{
		PipelineJournal journal(MakePipelineSettings({}, "driver 1", true));
		Expect(journal.Add(7) && journal.Contains(7) && !journal.Add(7), "memory list: keys are kept");
		journal.Seal();
		Expect(journal.Flush() && journal.GetStats().written_bytes == 0, "memory list: nothing is written");
		journal.Reset();
		Expect(!journal.Contains(7), "memory list: reset");
	}
}

// ---- refused programs (refusedPrograms.h): a program a fatal error ended an
// earlier start on is skipped, counted and logged ----

std::string ReadText(const std::filesystem::path &path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void TestRefusedProgramsFile() {
  const auto path = TestPath("refused");
  Remove(path);
  RefusedPrograms none;
  Expect(none.Open(path) == 0 && none.Count() == 0 && !none.Contains(1),
         "a missing file is an empty list");
  {
    std::ofstream out(path);
    out << "# comment\n0x00000000000000AB\tSpirV Fail: writable FLAT "
           "addresses\n0xabcdef0123456789\n0x123\nnot a hash\n"
           "0x00000000000000cdXjunk\n0xZZ0000000000000000\n "
           "0x0000000000000001\n0x0000000000000002 spaced reason\n";
  }
  RefusedPrograms r;
  Expect(r.Open(path) == 3, "three valid lines (either case, with or without a "
                            "reason, space or tab after the hash)");
  Expect(r.Contains(0xab) && r.Contains(0xabcdef0123456789ull) && r.Contains(2),
         "the programs");
  Expect(!r.Contains(0xcd) && !r.Contains(1) && !r.Contains(0x123),
         "junk after the digits, a leading space and short hashes are not "
         "programs");
  Expect(r.Reason(0xab) == "SpirV Fail: writable FLAT addresses" &&
             r.Reason(0xabcdef0123456789ull).empty() && r.Reason(77).empty(),
         "reasons");
  Remove(path);
}

void TestRefusedProgramsRecording() {
  const auto path = TestPath("refused-rec");
  Remove(path);
  RefusedPrograms r;
  r.Open(path);
  Expect(!r.RecordFatal("nothing in flight"),
         "no program in flight: nothing recorded");
  Expect(!std::filesystem::exists(path), "and no file is made");
  RefusedPrograms::Begin(0xfeed);
  Expect(RefusedPrograms::InFlight() == 0xfeed, "the thread names its program");
  std::thread other([&] {
    Expect(RefusedPrograms::InFlight() == 0,
           "another thread's in-flight program is its own");
    Expect(!r.RecordFatal("fatal on a thread with no program"),
           "a fatal error elsewhere records nothing");
  });
  other.join();
  Expect(std::filesystem::exists(path) == false, "still no file");
  Expect(r.RecordFatal("Error: condition (indirect_root >= images.size()) is "
                       "true\nsecond line\twith a tab"),
         "recorded");
  Expect(RefusedPrograms::InFlight() == 0, "recording ends the in-flight mark");
  RefusedPrograms::Begin(0xbeef);
  std::string long_text(500, 'x');
  Expect(r.RecordFatal(long_text), "a second program is appended");
  const auto text = ReadText(path);
  Expect(text.rfind("0x000000000000feed\tError: condition (indirect_root >= "
                    "images.size()) is true\n",
                    0) == 0,
         "first line: hash, tab, first line of the text");
  Expect(text.find("second line") == std::string::npos,
         "only the first line of the text");
  Expect(text.find("0x000000000000beef\t" +
                   std::string(RefusedPrograms::MaxReasonChars, 'x') + "\n") !=
             std::string::npos,
         "reason cut to 120 characters");
  RefusedPrograms again;
  Expect(again.Open(path) == 2 && again.Contains(0xfeed) &&
             again.Contains(0xbeef),
         "the next start reads both");
  // a path that cannot be written is not an error, the process just ends as
  // before
  RefusedPrograms bad;
  bad.Open(std::filesystem::path("/nonexistent-dir-for-test/refused"));
  RefusedPrograms::Begin(5);
  Expect(!bad.RecordFatal("x"), "an unwritable path reports false");
  RefusedPrograms::End();
  Remove(path);
}

void TestRefusedProgramsPathChoice() {
  using R = RefusedPrograms;
  // Off by default: a title that never asked for it (ASTRO) has no file, no skip, no recording.
  Expect(R::ChoosePath(nullptr, nullptr).empty() && R::ChoosePath(nullptr, "").empty() &&
             R::ChoosePath(nullptr, "/c/T.programs.bin").empty(),
         "env unset: off, even with a program cache path");
  Expect(R::ChoosePath("1", "/c/T.programs.bin") ==
             std::filesystem::path("/c/T.programs.bin.refused"),
         "=1: next to the program cache");
  Expect(R::ChoosePath("1", nullptr).empty() && R::ChoosePath("1", "").empty(),
         "=1 without a program cache path: off");
  Expect(R::ChoosePath("/x/r.txt", "/c/T.programs.bin") ==
             std::filesystem::path("/x/r.txt"),
         "an explicit path wins");
  Expect(R::ChoosePath("0", "/c/T.programs.bin").empty() &&
             R::ChoosePath("", "/c/T.programs.bin").empty(),
         "0 or empty turns it off");
}

void TestRefusedProgramsThroughTheFatalHook() {
  const auto path = TestPath("refused-hook");
  Remove(path);
  RefusedPrograms r;
  r.Open(path);
  r.Install();
  RefusedPrograms::Begin(0x77);
  Common::RunFatalHook(
      "Error: SpirV validation failed"); // what Common::DbgReport calls before
                                         // it ends the process
  r.Uninstall();
  Common::SetFatalHook(nullptr);
  RefusedPrograms::Begin(0x78);
  Common::RunFatalHook("after the hook was removed");
  RefusedPrograms::End();
  RefusedPrograms next;
  Expect(next.Open(path) == 1 && next.Contains(0x77) && !next.Contains(0x78),
         "only the program of the fatal error is recorded");
  Remove(path);
}

void TestPrecompileSkipsRefusedPrograms() {
  const auto path = TestPath("refused-replay");
  const auto journal_path = TestPath("refused-replay-journal");
  Remove(path);
  auto journal = MakeJournalWith(
      journal_path, 70); // seven sources 0x1000..0x1006, ten entries each
  // First start: the translation of program 0x1003 ends in a fatal error (stood
  // in for by the hook call).
  {
    RefusedPrograms r;
    r.Open(path);
    r.Install();
    ShaderPrecompiler::Settings settings;
    settings.threads = 1;
    settings.refused = &r;
    int in_flight_ok = 0, compiled = 0;
    ShaderPrecompiler precompiler(
        journal, settings, [&](const auto &source, const auto &) {
          if (RefusedPrograms::InFlight() == source.hash)
            in_flight_ok++;
          compiled++;
          if (source.hash == 0x1003) {
            Common::RunFatalHook("Error: the recompiler refuses this program");
          }
          return ShaderPrecompiler::Outcome::Compiled;
        });
    precompiler.Wait();
    r.Uninstall();
    Common::SetFatalHook(nullptr);
    Expect(in_flight_ok == 70 && compiled == 70,
           "each entry names its program while it is translated");
    Expect(RefusedPrograms::InFlight() == 0, "and none stays named afterwards");
    Expect(precompiler.GetProgress().refused == 0,
           "nothing was skipped in the first start");
  }
  // Second start: the program is skipped, counted, named once per line (at most
  // 8 lines), the others run.
  {
    RefusedPrograms r;
    Expect(r.Open(path) == 1 && r.Contains(0x1003),
           "the record was written by the failing thread");
    std::vector<std::string> lines;
    std::atomic<int> seen_refused{0}, compiled{0};
    ShaderPrecompiler::Settings settings;
    settings.threads = 3;
    settings.refused = &r;
    settings.log = [&](const std::string &line) { lines.push_back(line); };
    ShaderPrecompiler precompiler(journal, settings,
                                  [&](const auto &source, const auto &) {
                                    if (source.hash == 0x1003)
                                      seen_refused++;
                                    compiled++;
                                    return ShaderPrecompiler::Outcome::Compiled;
                                  });
    precompiler.Wait();
    const auto p = precompiler.GetProgress();
    Expect(seen_refused == 0 && compiled == 60,
           "no entry of the refused program is translated");
    Expect(p.done == 70 && p.total == 70 && p.skipped == 10 &&
               p.refused == 10 && p.compiled == 60 && p.failed == 0,
           "counted as skipped and refused");
    int named = 0;
    for (const auto &l : lines)
      named +=
          l.find("program 0x0000000000001003 skipped (refused earlier: Error: "
                 "the recompiler refuses this program)") != std::string::npos
              ? 1
              : 0;
    Expect(named == 8, "the hash is logged for the first 8 entries only");
    Expect(!lines.empty() &&
               lines.back().find("10 of refused programs") != std::string::npos,
           "the final line counts them");
  }
  // Without the record object nothing changes: every entry is translated and
  // nothing is named.
  {
    std::atomic<int> compiled{0}, named{0};
    ShaderPrecompiler::Settings settings;
    settings.threads = 2;
    ShaderPrecompiler precompiler(journal, settings,
                                  [&](const auto &, const auto &) {
                                    compiled++;
                                    if (RefusedPrograms::InFlight() != 0)
                                      named++;
                                    return ShaderPrecompiler::Outcome::Compiled;
                                  });
    precompiler.Wait();
    Expect(compiled == 70 && named == 0 &&
               precompiler.GetProgress().refused == 0,
           "refused == nullptr leaves the replay as it was");
  }
  // A compile that throws still ends the in-flight mark.
  {
    RefusedPrograms r;
    ShaderPrecompiler::Settings settings;
    settings.threads = 1;
    settings.refused = &r;
    settings.max_entries = 3;
    ShaderPrecompiler precompiler(
        journal, settings,
        [&](const auto &, const auto &) -> ShaderPrecompiler::Outcome {
          throw std::runtime_error("x");
        });
    precompiler.Wait();
    Expect(precompiler.GetProgress().failed == 3 &&
               RefusedPrograms::InFlight() == 0,
           "a throwing compile does not leave a program named");
  }
  Remove(path);
  Remove(journal_path);
}

} // namespace

int main() {
	TestRoundTripAndDedup();
	TestDamagedFile();
	TestIdentity();
	TestCapacity();
	TestBackgroundWriter();
	TestReplayDriver();
	TestPipelineJournal();
	TestPipelineJournalSealed();
        TestRefusedProgramsFile();
        TestRefusedProgramsRecording();
        TestRefusedProgramsPathChoice();
        TestRefusedProgramsThroughTheFatalHook();
        TestPrecompileSkipsRefusedPrograms();
        if (g_failures != 0) {
		std::printf("ShaderPrecompileTests: %d checks failed\n", g_failures);
		return 1;
	}
	std::printf("ShaderPrecompileTests: all checks passed\n");
	return 0;
}
