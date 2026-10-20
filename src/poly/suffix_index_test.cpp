// src/poly/suffix_index_test.cpp - SuffixIndex on sequences made for each case: a draft from another sequence, the
// vote of several places, the deepest match over more frequent shorter ones, no draft across a sequence's end, a
// prompt's earlier turns not indexed again, a full ring that forgets its oldest text, a long match found behind a
// trigram seen ten thousand times, and the time a draft and a token take in an index of four million tokens.
//
//   g++ -std=c++20 -O2 -Iinclude src/poly/suffix_index_test.cpp -o suffix_index_test && ./suffix_index_test
#include "suffix_index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using strata::poly::SuffixIndex;

namespace {

int checks = 0;
void check(bool ok, const char* what, double got = 0.0) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s (%.3f)\n", what, got);
        std::exit(1);
    }
}

using Seq = std::vector<int32_t>;

// n tokens no other case uses: from `base` on
Seq fresh(int base, int n) {
    Seq s;
    for (int i = 0; i < n; ++i) s.push_back(base + i);
    return s;
}

Seq cat(std::initializer_list<Seq> parts) {
    Seq s;
    for (const Seq& p : parts) s.insert(s.end(), p.begin(), p.end());
    return s;
}

struct Draft {
    Seq t;
    std::vector<double> sure;
    int match = 0;
};

Draft draft(SuffixIndex& x, int max_k) {
    int32_t out[SuffixIndex::kMaxDraft];
    Draft d;
    const int n = x.propose(max_k, out);
    d.t.assign(out, out + n);
    d.sure.assign(x.sure(), x.sure() + n);
    d.match = x.last_match();
    return d;
}

}  // namespace

int main() {
    const size_t small = 1 << 20;   // a ring of 16k tokens

    {   // another sequence goes on as this one will: its continuation, as long as the match at most
        SuffixIndex x(small);
        const Seq a = fresh(1000, 40);
        x.add(a.data(), a.size());
        x.begin(cat({fresh(2000, 5), Seq(a.begin(), a.begin() + 10)}));
        const Draft d = draft(x, 16);
        check(d.match == 10, "the match is the 10 tokens the two share", d.match);
        check(d.t == Seq(a.begin() + 10, a.begin() + 20), "the draft is what followed, 10 tokens: the match's length");
        check(d.sure.back() == 1.0, "one place: sure", d.sure.back());
        check(draft(x, 4).t.size() == 4, "no more than asked for");
    }
    {   // three places: two go on with X Y, one with Z W; the draft is X Y, two thirds sure
        SuffixIndex x(small);
        const Seq s = fresh(100, 12), X = {7001, 7002, 7003}, Z = {7101, 7102, 7103};
        for (const Seq& seq : {cat({fresh(300, 3), s, Z}), cat({fresh(400, 3), s, X}), cat({fresh(500, 3), s, X})})
            x.add(seq.data(), seq.size());
        x.begin(cat({fresh(600, 3), s}));
        const Draft d = draft(x, 8);
        check(d.t == X, "the token most places have next, then theirs");
        check(std::abs(d.sure[0] - 2.0 / 3.0) < 1e-12 && std::abs(d.sure[2] - 2.0 / 3.0) < 1e-12,
              "two of three places say X, and both go on alike", d.sure[0]);
    }
    {   // a tie goes to the newest place
        SuffixIndex x(small);
        const Seq s = fresh(100, 12);
        for (const Seq& seq : {cat({fresh(300, 3), s, {7001}}), cat({fresh(400, 3), s, {7101}})})
            x.add(seq.data(), seq.size());
        x.begin(cat({fresh(600, 3), s}));
        const Draft d = draft(x, 1);
        check(d.t == Seq{7101} && d.sure[0] == 0.5, "of two places that disagree, the newest", d.sure[0]);
    }
    {   // four places, four tokens: a quarter is under kMinSure, so no draft
        SuffixIndex x(small);
        const Seq s = fresh(100, 12);
        for (int i = 0; i < 4; ++i) {
            const Seq seq = cat({fresh(300 + 10 * i, 3), s, {7000 + i}});
            x.add(seq.data(), seq.size());
        }
        x.begin(cat({fresh(600, 3), s}));
        check(draft(x, 4).t.empty() && x.last_match() == 0, "places that all disagree give no draft");
    }
    {   // the deepest match wins over ten shorter ones that agree among themselves
        SuffixIndex x(small);
        const Seq deep = fresh(100, 20);
        const Seq one = cat({deep, {9001, 9002}});
        x.add(one.data(), one.size());
        for (int i = 0; i < 10; ++i) {
            const Seq seq = cat({fresh(3000 + 10 * i, 4), Seq(deep.end() - 5, deep.end()), {9101, 9102}});
            x.add(seq.data(), seq.size());
        }
        x.begin(cat({fresh(5000, 2), deep}));
        const Draft d = draft(x, 8);
        check(d.match == 20 && d.t == Seq{9001, 9002} && d.sure[1] == 1.0, "the place that matches furthest", d.match);
    }
    {   // a draft stops where its sequence ended, and a match does not run back over the mark before a sequence
        SuffixIndex x(small);
        const Seq a = fresh(100, 10), b = fresh(200, 10);
        x.add(a.data(), a.size());
        x.add(b.data(), b.size());   // in the ring: mark a mark b
        x.begin(cat({fresh(5000, 3), a}));
        check(draft(x, 8).t.empty(), "nothing after a sequence's last token, not even the next sequence's first");
        x.begin(cat({Seq(a.end() - 4, a.end()), Seq(b.begin(), b.begin() + 5)}));
        const Draft d = draft(x, 8);
        check(d.match == 5 && d.t == Seq(b.begin() + 5, b.end()), "b's match is b's own tokens, not a's before it",
              d.match);
    }
    {   // the answer goes in as it is written, and the draft can come from it
        SuffixIndex x(small);
        const Seq p = fresh(100, 30);
        x.begin(p);
        const Seq ans = fresh(400, 12);
        for (int32_t t : ans) x.append(t);
        for (int i = 0; i < 6; ++i) x.append(ans[(size_t) i]);   // the answer starts to repeat itself
        const Draft d = draft(x, 8);
        check(d.match == 6 && d.t == Seq(ans.begin() + 6, ans.end()), "a draft from the answer so far", d.match);
    }
    {   // a prompt that goes on from a conversation seen before: only its new tokens go in; a branch gets a lead
        SuffixIndex x(small);
        const Seq p1 = fresh(100, 500), a1 = fresh(1000, 50), u2 = fresh(2000, 40);
        x.begin(p1);
        for (int32_t t : a1) x.append(t);
        const uint64_t w = x.written();
        x.begin(cat({p1, a1, u2}));
        check(x.written() - w == u2.size(), "the next turn's prompt: its new tokens only", (double) (x.written() - w));
        const uint64_t w2 = x.written();
        x.begin(cat({Seq(p1.begin(), p1.begin() + 300), u2}));   // the conversation taken back to token 300
        check(x.written() - w2 == 1 + SuffixIndex::kDepth + u2.size(), "a branch: a mark, the lead and its new tokens",
              (double) (x.written() - w2));
        // p1's stretch around 250 is in the ring once, so its continuation is sure: one place, or places that agree
        x.begin(cat({fresh(8000, 3), Seq(p1.begin() + 240, p1.begin() + 260)}));
        const Draft d = draft(x, 4);
        check(d.t == Seq(p1.begin() + 260, p1.begin() + 264) && d.sure[3] == 1.0, "and drafts from it as before");
        const uint64_t w3 = x.written();
        x.begin(cat({p1, a1, u2}));   // back to the first conversation: seen whole, nothing new
        check(x.written() - w3 == 1 + SuffixIndex::kDepth, "a conversation seen whole: a mark and the lead only",
              (double) (x.written() - w3));
    }
    {   // a full ring forgets its oldest text, and the memory stays what it was given
        SuffixIndex x(0);
        check(x.capacity() == 4096 && x.bytes() == 4096 * SuffixIndex::kBytesPerToken, "the smallest ring");
        const Seq old = fresh(100, 30);
        x.add(old.data(), old.size());
        std::mt19937 rng(7);
        Seq noise;
        for (int i = 0; i < 100000; ++i) noise.push_back(50000 + (int32_t) (rng() % 30000));
        x.add(noise.data(), noise.size());
        check(x.held() == 4096, "the ring holds its capacity", (double) x.held());
        x.begin(Seq(old.begin(), old.begin() + 10));
        check(draft(x, 8).t.empty(), "text that left the ring gives no draft");
        x.begin(Seq(noise.end() - 200, noise.end() - 190));
        check(draft(x, 8).t == Seq(noise.end() - 190, noise.end() - 182), "recent text does");
    }
    {   // a long match is found behind a trigram seen ten thousand times (the 24- and 8-token chains)
        SuffixIndex x(64u << 20);
        const Seq tri = {11, 12, 13};
        const Seq longer = fresh(100, 27), mid = fresh(300, 7);
        const Seq a = cat({longer, tri, {9001}}), b = cat({mid, tri, {9101}});
        x.add(a.data(), a.size());
        x.add(b.data(), b.size());
        std::mt19937 rng(11);
        for (int i = 0; i < 10000; ++i) {
            const Seq seq = cat({{20000 + (int32_t) (rng() % 5000)}, tri, {30000 + (int32_t) (rng() % 5000)}});
            x.add(seq.data(), seq.size());
        }
        x.begin(cat({{5}, longer, tri}));
        Draft d = draft(x, 8);
        check(d.match == 30 && d.t == Seq{9001}, "a 30-token match behind 10,000 newer places of its trigram", d.match);
        x.begin(cat({{5}, mid, tri}));
        d = draft(x, 8);
        check(d.match == 10 && d.t == Seq{9101}, "a 10-token match too", d.match);
        x.begin(cat({{5}, tri}));
        d = draft(x, 8);
        check(d.match == 0 && d.t.empty(), "the trigram alone: its newest places disagree, no draft", d.match);
    }
    {   // time: an index of four million tokens of code-like text, a draft and a token at a time
        SuffixIndex x(256u << 20);
        std::mt19937 rng(20261019);
        // lines of a few tokens from a small vocabulary, some lines copied from earlier ones, as code has
        std::vector<Seq> lines;
        std::geometric_distribution<int> len(0.15);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        Seq text;
        while (text.size() < 4'000'000) {
            if (!lines.empty() && u(rng) < 0.4) {
                const Seq& l = lines[rng() % lines.size()];
                text.insert(text.end(), l.begin(), l.end());
                continue;
            }
            Seq l;
            for (int i = 0, n = 2 + len(rng); i < n; ++i) l.push_back((int32_t) std::min<double>(-200.0 * std::log(u(rng) + 1e-9), 4000.0));
            l.push_back(198);   // the newline
            text.insert(text.end(), l.begin(), l.end());
            if (lines.size() < 50000) lines.push_back(l);
        }
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        x.add(text.data(), text.size() - 100000);
        const double put_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() /
                              (double) (text.size() - 100000);
        x.begin(Seq(text.end() - 100000, text.end() - 99000));
        int32_t out[SuffixIndex::kMaxDraft];
        long long drafts = 0, right = 0;
        const auto t1 = Clock::now();
        for (size_t i = text.size() - 99000; i < text.size(); ++i) {
            const int n = x.propose(7, out);
            drafts += n > 0;
            for (int j = 0; j < n && i + (size_t) j < text.size() && out[j] == text[i + (size_t) j]; ++j) ++right;
            x.append(text[i]);
        }
        const double us = std::chrono::duration<double, std::micro>(Clock::now() - t1).count() / 99000.0;
        std::printf("suffix_index_test: %zu tokens in the ring (%zu MiB), %.3f us a token to index, %.2f us a draft "
                    "and a token; %lld drafts in 99,000 places, %lld of their tokens right\n",
                    x.held(), x.bytes() >> 20, put_us, us, drafts, right);
        check(x.held() == 4u << 20 || x.held() >= text.size() - 100000, "the ring holds the text");
        check(us < 50.0, "a draft and a token take microseconds", us);
        check(drafts > 0 && right > 0, "and repeated text gives right drafts", (double) right);
    }
    std::printf("suffix_index_test: %d checks passed\n", checks);
    return 0;
}
