// src/poly/draft_race_test.cpp - Does a read with the draft block repeat bit for bit?  Each run opens the engine,
// reads the same prompt in several calls on the layer-major path (calls of 64 tokens, stage_ahead 64), then asks the
// draft block for three guesses, one after the other.  The guesses, their probabilities and the state after the read
// must be the same in every run.  Expert swaps and the CPU's share of a small call are off, so neither changes
// between runs; the expert cache is still filled at each load.
// A read that did not give the draft block two or more calls on that path (every expert cached, no MMQ) cannot show
// the race: the test then says SKIP instead of PASS.
//
//   poly_draft_race_test MODEL.gguf [RUNS] [LAYER_SPLIT]     (RUNS >= 2, default 8; LAYER_SPLIT: as --layer-split,
//                                                            empty: one card)
//
// Needs the model file and a card, so it is run by hand.  Exit 0: every run matched the first; 1: a run differed;
// 2: an error or a bad argument; 3: SKIP.
#include "engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using strata::poly::Engine;
using strata::poly::Options;

namespace {

struct Seen {
    int64_t draft_calls = 0;   // the read's calls of the draft block's rows on the layer-major path
    std::vector<int32_t> guesses;
    std::vector<float> probs;
    std::vector<uint8_t> state;
};

Seen run_once(const Options& o, const std::vector<int32_t>& text, int n) {
    auto e = Engine::open(o);
    if (!e->drafts()) throw std::runtime_error("the model has no draft block, or it does not run");
    const int64_t before = e->read_draft_calls_for_test();
    if (!e->read(text.data(), n)) throw std::runtime_error("the read was given up");
    Seen s;
    s.draft_calls = e->read_draft_calls_for_test() - before;
    s.state.resize(e->state_bytes());
    e->save_state(s.state.data());
    int32_t g = text[(size_t) n];
    for (int i = 0; i < 3; ++i) {
        float p = 0.f;
        g = e->draft(g, i, &p);
        s.guesses.push_back(g);
        s.probs.push_back(p);
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [RUNS] [LAYER_SPLIT]\n", argv[0]);
        return 2;
    }
    Options o;
    o.model = argv[1];
    int runs = 8;
    if (argc > 2) {
        char* end = nullptr;
        const long v = std::strtol(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || v < 2 || v > 1000) {
            std::fprintf(stderr, "RUNS is a whole number from 2 to 1000 (one run compares nothing), not '%s'\n", argv[2]);
            return 2;
        }
        runs = (int) v;
    }
    if (argc > 3) o.layer_split = argv[3];
    o.max_context = 4096;
    o.prompt_batch = 64;   // a read of 200 tokens is four calls
    o.stage_ahead = 64;    // and takes the layer-major path, where the draft block's rows come last
    o.window = 4;
    o.expert_swaps = 0;
    o.cpu_share = 0;
    const int n = 200;
    std::vector<int32_t> text(n + 1);
    for (int i = 0; i <= n; ++i) text[(size_t) i] = 1000 + (53 * i + i / 9) % 100000;
    try {
        const Seen first = run_once(o, text, n);
        if (first.draft_calls < 2) {
            std::fprintf(stderr, "SKIP: the read gave the draft block %lld call(s) on the layer-major path; the race "
                                 "needs two or more (every expert cached, or no MMQ?)\n", (long long) first.draft_calls);
            return 3;
        }
        std::fprintf(stderr, "the read gave the draft block %lld calls on the layer-major path\n",
                     (long long) first.draft_calls);
        int differ = 0;
        for (int k = 1; k < runs; ++k) {
            const Seen s = run_once(o, text, n);
            const bool g = s.guesses == first.guesses;
            const bool p = std::memcmp(s.probs.data(), first.probs.data(), s.probs.size() * sizeof(float)) == 0;
            const bool st = s.state == first.state;
            if (s.draft_calls != first.draft_calls) throw std::runtime_error("a run took another read path");
            std::fprintf(stderr, "run %d: guesses %s, probabilities %s, state %s\n", k + 1, g ? "same" : "DIFFER",
                         p ? "same" : "DIFFER", st ? "same" : "DIFFER");
            differ += !(g && p && st);
        }
        std::fprintf(stderr, "%s: %d of %d runs differ from the first (split '%s')\n", differ ? "FAIL" : "PASS", differ,
                     runs - 1, o.layer_split.c_str());
        return differ ? 1 : 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "ERROR: %s\n", ex.what());
        return 2;
    }
}
