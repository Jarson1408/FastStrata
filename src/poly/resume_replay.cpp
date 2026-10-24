// src/poly/resume_replay.cpp - where the conversation cache takes up the prompts of a session, without the model: the
// prompts go through Conversations as main.cpp's serve loop sends a prompt without pictures (a read of up to 65,536
// tokens with the checkpoints inside it, then the last token), with the daily config's cache (six checkpoints, one
// every 2,048 tokens, 8 slots and 12 GiB to park in) and an engine whose state is a hash of the tokens read
// (conversations_test.cpp's).  test_scripts/resume_bench.py builds and runs it.
//
//   resume_replay LIST [N]      LIST: lines "file tokens turn_at last_message" (resume_bench.py tokenize); N:
//                               --prompt-cache-message N (default 0)
#include "conversations.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

using strata::poly::CacheOptions;
using strata::poly::Conversations;

namespace {

uint64_t mix(uint64_t h, int32_t t) { return (h ^ (uint64_t) (uint32_t) t) * 0x9E3779B97F4A7C15ull + 0x7F4A7C15ull; }

struct Hash {
    static constexpr int64_t ctx = 1 << 18;
    uint64_t h = 1;
    int64_t at = 0;
    std::vector<uint64_t> a = std::vector<uint64_t>(ctx), b = std::vector<uint64_t>(ctx);

    void reset() { h = 1, at = 0; }
    void eval(const int32_t* t, int64_t n) {
        for (int64_t i = 0; i < n; ++i, ++at) {
            a[(size_t) at] = h = mix(h, t[i]);
            b[(size_t) at] = (uint64_t) t[i];
        }
    }
    size_t state_bytes() const { return 8; }
    void save_state(uint8_t* dst) const { std::memcpy(dst, &h, 8); }
    void load_state(const uint8_t* src, int64_t pos) { std::memcpy(&h, src, 8), at = pos; }
    size_t row_bytes() const { return 16; }
    void save_rows(int64_t first, int64_t n, uint8_t* dst) const {
        std::memcpy(dst, &a[(size_t) first], (size_t) n * 8);
        std::memcpy(dst + n * 8, &b[(size_t) first], (size_t) n * 8);
    }
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
        std::memcpy(&a[(size_t) first], src + (first - from) * 8, (size_t) n * 8);
        std::memcpy(&b[(size_t) first], src + count * 8 + (first - from) * 8, (size_t) n * 8);
    }
    bool holds(const std::vector<int32_t>& ids) const {
        uint64_t x = 1;
        for (size_t i = 0; i < ids.size(); ++i)
            if (a[i] != (x = mix(x, ids[i])) || b[i] != (uint64_t) ids[i]) return false;
        return at == (int64_t) ids.size() && h == x;
    }
};

std::vector<int32_t> load(const std::string& path) {
    std::ifstream f(path);
    std::vector<int32_t> v;
    std::string s;
    while (std::getline(f, s, ',')) v.push_back(std::atoi(s.c_str()));
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: resume_replay LIST [N]\n");
        return 2;
    }
    const int64_t message = argc > 2 ? std::atoll(argv[2]) : 0;
    CacheOptions o;
    o.park_mib = 12288;
    o.park_slots = 8;
    Hash e;
    Conversations<Hash> kept(e, o);
    std::mt19937 rng(7);
    std::string dir = argv[1];
    dir = dir.substr(0, dir.find_last_of("/\\") + 1);
    std::ifstream list(argv[1]);
    std::vector<std::pair<int64_t, uint8_t*>> due;
    long long read = 0, total = 0, prompts = 0;
    std::string line;
    std::printf("--prompt-cache-message %lld\n", (long long) message);
    while (std::getline(list, line)) {
        std::istringstream ls(line);
        std::string file;
        int64_t tokens = 0, turn = -1, last = -1;
        ls >> file >> tokens >> turn >> last;
        std::vector<int32_t> ids = load(dir + file);
        const int64_t n = (int64_t) ids.size(), from = kept.resume(ids);
        kept.turn_at(turn);
        if (message > 0) kept.message_at(strata::poly::last_message(ids, from, turn, message));
        for (int64_t i = from; i < n;) {
            const bool end = i == n - 1;
            const int64_t m = end ? 1 : std::min<int64_t>(n - 1 - i, 65536);
            kept.before(i, end);
            if (!end) {
                kept.inside(i, m, due);
                int64_t done = 0;
                for (const auto& d : due) {
                    e.eval(ids.data() + i + done, d.first - done);
                    done = d.first;
                    if (d.second) e.save_state(d.second);
                }
                e.eval(ids.data() + i + done, m - done);
            } else {
                e.eval(ids.data() + i, m);
            }
            kept.read(ids.data() + i, m);
            i += m;
        }
        kept.prompt_was(n);
        for (int k = 0; k < 16; ++k) {   // an answer, which the next prompt does not hold as it was written
            const int32_t t = 1000000 + (int32_t) (rng() % 1000);
            e.eval(&t, 1);
            kept.read(&t, 1);
            ids.push_back(t);
        }
        if (!e.holds(ids)) {
            std::fprintf(stderr, "%s: the engine does not hold the prompt\n", file.c_str());
            return 1;
        }
        std::printf("%-28s %7lld tokens, taken up at %7lld, %6lld read; its last message begins at %lld\n",
                    file.c_str(), (long long) n, (long long) from, (long long) (n - from), (long long) last);
        read += n - from, total += n, ++prompts;
    }
    std::printf("%lld prompts: %lld of their %lld tokens read\n", prompts, read, total);
    return 0;
}
