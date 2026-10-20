// src/poly/guesses_test.cpp - Guesses on a machine whose times are set: the block goes on guessing after a slow
// stretch, a stalled round and a stalled ask, comes back when windows begin to pay, and costs little where they
// do not.
//
//   g++ -std=c++20 -Iinclude src/poly/guesses_test.cpp src/spec/draft_policy.cpp -o guesses_test && ./guesses_test
#include "guesses.hpp"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>

using strata::poly::Guesses;
using strata::spec::DraftPolicy;

namespace {

int checks = 0;
void check(bool ok, const char* what, double got) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s (%.3f)\n", what, got);
        std::exit(1);
    }
}

// A token alone takes `one` ms, a token more in a window `step` of that, and asking the block `ask`: a block that
// is a model of its own, as Gemma 4's, takes about this much of a token.  A round's time is multiplied by
// slow(round), an ask's by slow_ask(round).
struct Machine {
    double one = 18.0, step = 0.33, ask = 5.0;
    std::function<double(long long)> slow = [](long long) { return 1.0; };
    std::function<double(long long)> slow_ask = [](long long) { return 1.0; };
};

struct Sum {
    long long rounds = 0, windows = 0, tokens = 0;
    double ms = 0.0;
    double share() const { return double(windows) / double(rounds); }   // of the rounds, those with a guess in
    double rate() const { return 1000.0 * double(tokens) / ms; }        // tokens a second
};

// Rounds as Session::generate() makes them, without repeats.  More than half of the block's guesses are nearly
// sure, some are likely and a few are not, and a guess is right as often as the block says.
template<class G>
struct Play {
    G g;
    DraftPolicy policy{4};
    std::mt19937 rng{20261010};
    long long round = 0;

    Sum rounds(const Machine& m, long long n) {
        Sum s;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (long long i = 0; i < n; ++i, ++round) {
            const bool ask = g.begin(true, policy);
            bool right[3] = {};
            double asking = 0.0;
            int n_d = 0;
            if (ask)
                for (; n_d < 3 && g.wants(n_d); ++n_d) {
                    const double kind = u(rng);
                    const double p = kind < 0.55 ? 0.85 + 0.14 * u(rng) : kind < 0.85 ? 0.5 + 0.35 * u(rng) : 0.05 + 0.45 * u(rng);
                    const double took = m.ask * m.slow_ask(round);
                    asking += took;
                    right[n_d] = u(rng) < p;
                    if (!g.takes(n_d, p, took)) break;
                }
            const int t = 1 + n_d;
            int ok = 0;
            while (ok < n_d && right[ok]) ++ok;
            const double ms = m.one * (1.0 + m.step * (t - 1)) * m.slow(round) + asking;
            policy.observe(false, t, ok, 0, g.end(t, 1 + ok, ms, false));
            ++s.rounds, s.windows += t > 1, s.tokens += 1 + ok, s.ms += ms;
        }
        return s;
    }
};

// What each case measures, for any rule with Guesses' four calls.
struct Cases {
    double steady = 0, steady_rate = 0;   // the share of rounds with a guess on the machine as it is, and tokens/s
    double after_stretch = 0;             // that share after 30 rounds that took 20 times as long
    double after_round = 0;               // after one round that took 30 times as long
    double after_ask = 0;                 // after one ask that took 100 times as long
    double after_cold = 0;                // after 400 rounds in which a token more cost nearly a token
    double dear_rate = 0;                 // tokens/s where a token more always costs nearly a token
};

template<class G>
Cases cases() {
    Cases c;
    const Machine plain;
    {
        Play<G> p;
        p.rounds(plain, 300);
        const Sum s = p.rounds(plain, 2000);
        c.steady = s.share(), c.steady_rate = s.rate();
    }
    {
        Play<G> p;
        Machine m;
        m.slow = [](long long r) { return r >= 300 && r < 330 ? 20.0 : 1.0; };
        p.rounds(m, 300 + 30 + 100);
        c.after_stretch = p.rounds(m, 300).share();
    }
    {
        Play<G> p;
        Machine m;
        m.slow = [](long long r) { return r == 300 ? 30.0 : 1.0; };
        p.rounds(m, 301);
        c.after_round = p.rounds(m, 300).share();
    }
    {
        Play<G> p;
        Machine m;
        m.slow_ask = [](long long r) { return r == 300 ? 100.0 : 1.0; };
        p.rounds(m, 301);
        c.after_ask = p.rounds(m, 300).share();
    }
    {
        Play<G> p;
        Machine dear;
        dear.step = 0.97;
        p.rounds(dear, 400);
        p.rounds(plain, 150);
        c.after_cold = p.rounds(plain, 300).share();
    }
    {
        Play<G> p;
        Machine dear;
        dear.step = 0.97;
        p.rounds(dear, 300);
        c.dear_rate = p.rounds(dear, 4000).rate();
    }
    return c;
}

}  // namespace

#ifndef GUESSES_TEST_NO_MAIN
int main() {
    const Cases c = cases<Guesses>();
    const double alone = 1000.0 / Machine().one;   // tokens a second without a guess
    check(c.steady > 0.8, "on a machine as it is, most rounds have a guess", c.steady);
    check(c.steady_rate > 1.05 * alone, "and the answer is written faster for them", c.steady_rate / alone);
    check(c.after_stretch > 0.9 * c.steady, "the guesses come back after a slow stretch", c.after_stretch);
    check(c.after_round > 0.9 * c.steady, "one stalled round does not stop them", c.after_round);
    check(c.after_ask > 0.9 * c.steady, "nor does one stalled ask", c.after_ask);
    check(c.after_cold > 0.9 * c.steady, "they begin when windows begin to pay", c.after_cold);
    check(c.dear_rate > 0.97 * alone, "where a window does not pay, looking again costs under 3%", c.dear_rate / alone);
    std::printf("guesses_test: %d checks passed\n", checks);
    return 0;
}
#endif
