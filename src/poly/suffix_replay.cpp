// src/poly/suffix_replay.cpp - the drafts --lookup and --suffix-drafts offer on recorded answers, without the model:
// how much of what the model wrote either source had guessed.
//
//   python test_scripts/agent_bench.py export LABEL OUT.pa     the prompts of a run and the answers it wrote
//   g++ -std=c++20 -O2 -Iinclude -Isrc/poly src/poly/suffix_replay.cpp src/spec/suffix_drafter.cpp -o suffix_replay
//   ./suffix_replay OUT.pa
//
// Lines "P ids" (a request's prompt) and "A ids" (the answer written to it), ids separated by commas, in the order
// the engine got them.  At each place of an answer a source gives a draft as long as the engine's window would take
// (7 tokens when the match is 8 tokens or more, else 3: main.cpp's kCopyMatch and kBlockWindow); the draft's tokens
// that are right and the token after them are taken, as a window takes them, and with no draft one token is.  The
// engine's policy, which may prefer the draft block's guesses, is not modelled: this is what each source could give.
#include "strata/spec/suffix_drafter.hpp"
#include "suffix_index.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Sum {
    long long tokens = 0, drafts = 0, guessed = 0, right = 0;
    void add(const Sum& s) { tokens += s.tokens, drafts += s.drafts, guessed += s.guessed, right += s.right; }
};

std::vector<int32_t> ids(const std::string& s) {
    std::vector<int32_t> v;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, ',');)
        if (!x.empty()) v.push_back((int32_t) std::stol(x));
    return v;
}

// One answer through a source: begin(prompt), then propose / take as a window would.
template<class Begin, class Propose, class Append>
Sum replay(const std::vector<int32_t>& answer, Begin begin, Propose propose, Append append) {
    Sum s;
    begin();
    for (size_t i = 0; i < answer.size();) {
        int32_t d[strata::poly::SuffixIndex::kMaxDraft];
        int match = 0;
        int k = propose(d, match);
        k = std::min(k, match >= 8 ? 7 : 3);
        int ok = 0;
        while (ok < k && i + (size_t) ok < answer.size() && d[ok] == answer[i + (size_t) ok]) ++ok;
        if (k > 0) ++s.drafts, s.guessed += k, s.right += ok;
        const size_t take = std::min(answer.size() - i, (size_t) ok + 1);
        for (size_t j = 0; j < take; ++j) append(answer[i + j]);
        i += take;
        s.tokens += (long long) take;
    }
    return s;
}

void say(const char* name, const Sum& s) {
    std::printf("%s: %lld tokens written; %lld drafts, %lld of their %lld tokens right (%.2f a draft), %.1f%% of the "
                "tokens guessed\n", name, s.tokens, s.drafts, s.right, s.guessed,
                s.drafts ? (double) s.right / (double) s.drafts : 0.0, 100.0 * (double) s.right / (double) s.tokens);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: suffix_replay FILE.pa [index MiB, default 256]\n");
        return 2;
    }
    std::ifstream f(argv[1]);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    strata::poly::SuffixIndex index((size_t) (argc > 2 ? std::stoul(argv[2]) : 256) << 20);
    strata::spec::SuffixDrafter repeats(3, 64, 147456 + 4096);   // --lookup 3, as main.cpp makes it
    std::vector<int32_t> prompt;
    Sum lookup_all, suffix_all;
    int n = 0;
    for (std::string line; std::getline(f, line);) {
        if (line.size() < 2) continue;
        const std::vector<int32_t> v = ids(line.substr(2));
        if (line[0] == 'P') {
            prompt = v;
            continue;
        }
        const Sum a = replay(
            v, [&] { repeats.reset(); repeats.append(prompt.data(), prompt.size()); },
            [&](int32_t* d, int& match) { const int k = repeats.propose(7, d); match = repeats.last_match(); return k; },
            [&](int32_t t) { repeats.append(t); });
        const Sum b = replay(
            v, [&] { index.begin(prompt); },
            [&](int32_t* d, int& match) { const int k = index.propose(7, d); match = index.last_match(); return k; },
            [&](int32_t t) { index.append(t); });
        std::printf("%3d: prompt %6zu, answer %5zu | lookup %4lld drafts, %5lld of %5lld right | suffix %4lld drafts, "
                    "%5lld of %5lld right\n", ++n, prompt.size(), v.size(), a.drafts, a.right, a.guessed, b.drafts,
                    b.right, b.guessed);
        lookup_all.add(a);
        suffix_all.add(b);
    }
    say("lookup", lookup_all);
    say("suffix", suffix_all);
    return 0;
}
