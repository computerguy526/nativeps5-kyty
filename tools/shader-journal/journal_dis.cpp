// nps5 kart: offline disassembler for a Kyty shader journal (_PipelineCache/<title>.shaders.journal).
// The journal holds the guest RDNA2 code of every program a run compiled, so this gives an offline
// corpus of a title's shaders without a title run.
//   journal_dis <journal>                 one line per source (index, kind, hash, words)
//   journal_dis <journal> <hex hash>...   RDNA2 disassembly of those sources
//   journal_dis <journal> --grep <text>   sources whose disassembly contains <text> (with counts)
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct Source {
	uint32_t              stage = 0;
	uint8_t               kind  = 0;
	uint64_t              hash  = 0;
	std::vector<uint32_t> code;
};
template <class T>
bool Get(const std::vector<uint8_t>& d, size_t& p, T& v) {
	if (p + sizeof(T) > d.size()) return false;
	std::memcpy(&v, d.data() + p, sizeof(T));
	p += sizeof(T);
	return true;
}
} // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: journal_dis <journal> [<hex hash>... | --grep <text>]\n");
		return 2;
	}
	std::FILE* f = std::fopen(argv[1], "rb");
	if (f == nullptr) return 1;
	std::vector<uint8_t> d;
	uint8_t              buf[65536];
	for (size_t n; (n = std::fread(buf, 1, sizeof(buf), f)) > 0;) d.insert(d.end(), buf, buf + n);
	std::fclose(f);
	if (d.size() < 16 || std::memcmp(d.data(), "KYSHJRN1", 8) != 0) return 1;
	size_t   p = 12;
	uint32_t n = 0;
	Get(d, p, n);
	p += n + 8;
	std::vector<Source> sources;
	while (p + 24 <= d.size()) {
		uint32_t magic = 0, kind = 0;
		uint64_t size = 0, checksum = 0;
		Get(d, p, magic), Get(d, p, kind), Get(d, p, size), Get(d, p, checksum);
		if (magic != 0x524a484bu || size > d.size() - p) break;
		size_t q = p;
		p += size;
		if (kind != 1 && kind != 3) continue;
		Source   s;
		uint32_t skip32 = 0;
		uint8_t  skip8  = 0;
		Get(d, q, s.stage), Get(d, q, s.kind), Get(d, q, s.hash);
		Get(d, q, skip32), Get(d, q, skip32), Get(d, q, skip32), Get(d, q, skip8);
		Get(d, q, n);
		q += size_t {n} * 4;
		Get(d, q, n);
		s.code.resize(n);
		std::memcpy(s.code.data(), d.data() + q, size_t {n} * 4);
		sources.push_back(std::move(s));
	}
	const char* kinds[] = {"vs", "ps", "cs"};
	const bool  grep    = argc >= 4 && std::strcmp(argv[2], "--grep") == 0;
	for (size_t i = 0; i < sources.size(); i++) {
		const auto& s = sources[i];
		if (argc == 2) {
			std::printf("%4zu %s %016" PRIx64 " words %zu\n", i, kinds[s.kind % 3], s.hash, s.code.size());
			continue;
		}
		bool wanted = grep;
		for (int a = 2; a < argc && !grep; a++) wanted = wanted || std::strtoull(argv[a], nullptr, 16) == s.hash;
		if (!wanted) continue;
		using namespace Libs::Graphics::ShaderRecompiler;
		Decoder::Program program;
		Decoder::DecodeProgram(s.code, program);
		const auto text = Decoder::ProgramToString(program);
		if (grep) {
			size_t count = 0;
			for (size_t at = text.find(argv[3]); at != std::string::npos; at = text.find(argv[3], at + 1)) count++;
			if (count != 0) std::printf("%4zu %s %016" PRIx64 " words %zu matches %zu\n", i, kinds[s.kind % 3], s.hash, s.code.size(), count);
			continue;
		}
		std::printf("==== %zu %s %016" PRIx64 " words %zu\n%s\n", i, kinds[s.kind % 3], s.hash, s.code.size(), text.c_str());
	}
	return 0;
}
