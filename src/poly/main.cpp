// src/poly/main.cpp - strata-poly: GLM 5.3 Flash from its GGUF.
//
//   strata-poly --model M.gguf --tokens 1,2,3 [--max-new N]   one prompt; prints the generated ids
//   strata-poly --model M.gguf --serve                        the line protocol serve/server.py speaks
//
// The protocol is the one `strata --serve` speaks (src/program/generate.cpp): GEN, GENI for a prompt with pictures,
// STOP and QUIT in, and RESUME, PP, T, DONE and ERR out.
#include "conversations.hpp"
#include "cpu_experts.hpp"
#include "engine.hpp"
#include "guesses.hpp"
#include "suffix_index.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#undef far   // a word of this file's
#endif

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"
#endif

namespace {

using strata::poly::CacheOptions;
using strata::poly::ConversationDisk;
using strata::poly::Conversations;
using strata::poly::Engine;
using strata::poly::Guesses;
using strata::poly::Options;
using strata::poly::SuffixIndex;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// serve() saves the experts' use after its first answer and then after an answer when the last save is this long
// ago, and not only when it is told to quit: a server that is ended on Windows takes its engine with it at once.
// A save is a file of some 40 KB and a rename.
constexpr double kProfileSaveMs = 10000.0;

// The most tokens of a window whose guesses are the block's: the token the model has and three guesses, each
// from the one before.  A repeat of the conversation may fill a longer window (Options::window).
constexpr int kBlockWindow = 4;

// --conversation-disk's files by default: a conversation of the whole 144K context with --kv-q8 takes about 2.0 GiB
// (11.7 KiB a token and its six checkpoints' states of 63 MiB), so this holds it and another long one, or several
// short ones (the agent benchmark's eight took 3.0 GiB), the latest used first.
constexpr long long kDiskMib = 4096;

// A window longer than that is for a copy: a repeat of the conversation whose match is this many tokens or more.
// A repeat of a few tokens is in any text, and what followed it there is seldom what follows here for long.
constexpr int kCopyMatch = 8;

struct Sampling {
    float temperature = 0.f;   // 0: the most likely token
    float top_p = 1.f, min_p = 0.f;
    int top_k = 20;            // 0: no limit
    int penalty_last_n = 0;
    float penalty_repeat = 1.f, penalty_freq = 0.f, penalty_present = 0.f;
    unsigned long long seed = 0;
};

// llama.cpp's sampler chain in its default order: penalties, top-k, top-p, min-p, temperature, then the draw.
class Sampler {
public:
    int pick(const float* logits, int n_vocab, const Sampling& s, const std::vector<int32_t>& history,
             std::mt19937_64& rng) {
        if (penalizes(s)) {
            pen_.assign(logits, logits + n_vocab);
            penalize(s, history);
            logits = pen_.data();
        }
        if (s.temperature <= 0.f) return strata::poly::most_likely(logits, n_vocab);

        // The k best, the best first and of equals the lower id.  A token is a candidate only when it beats the
        // worst of the k so far, which is the heap's first; most of a vocabulary never does.
        auto by_logit = [](const Cand& a, const Cand& b) { return a.logit != b.logit ? a.logit > b.logit : a.id < b.id; };
        const size_t k = s.top_k > 0 ? std::min((size_t) s.top_k, (size_t) n_vocab) : (size_t) n_vocab;
        cand_.clear();
        for (int i = 0; i < n_vocab; ++i) {
            if (cand_.size() == k) {
                if (!(logits[i] > cand_.front().logit)) continue;   // an equal with a higher id is no better
                std::pop_heap(cand_.begin(), cand_.end(), by_logit);
                cand_.back() = {logits[i], i};
            } else {
                cand_.push_back({logits[i], i});
            }
            std::push_heap(cand_.begin(), cand_.end(), by_logit);
        }
        std::sort_heap(cand_.begin(), cand_.end(), by_logit);

        if (s.top_p < 1.f) {
            softmax();
            double cum = 0.0;
            size_t keep = cand_.size();
            for (size_t i = 0; i < cand_.size(); ++i) {
                cum += p_[i];
                if (cum >= s.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            cand_.resize(keep);
        }
        if (s.min_p > 0.f) {
            const float floor = cand_[0].logit + std::log(s.min_p);
            size_t keep = 1;
            while (keep < cand_.size() && cand_[keep].logit >= floor) ++keep;
            cand_.resize(keep);
        }
        for (Cand& c : cand_) c.logit /= s.temperature;
        softmax();
        return cand_[std::discrete_distribution<size_t>(p_.begin(), p_.end())(rng)].id;
    }

private:
    struct Cand {
        float logit;
        int id;
    };

    static bool penalizes(const Sampling& s) {
        return s.penalty_last_n != 0 &&
               (s.penalty_repeat != 1.f || s.penalty_freq != 0.f || s.penalty_present != 0.f);
    }

    // on pen_, the logits' copy
    void penalize(const Sampling& s, const std::vector<int32_t>& history) {
        const size_t n = s.penalty_last_n < 0 ? history.size() : std::min(history.size(), (size_t) s.penalty_last_n);
        seen_.clear();
        for (size_t i = history.size() - n; i < history.size(); ++i) ++seen_[history[i]];
        for (const auto& [id, count] : seen_) {
            float& l = pen_[(size_t) id];
            l = l <= 0.f ? l * s.penalty_repeat : l / s.penalty_repeat;
            l -= (float) count * s.penalty_freq + s.penalty_present;
        }
    }

    void softmax() {
        p_.resize(cand_.size());
        float mx = cand_[0].logit;
        for (const Cand& c : cand_) mx = std::max(mx, c.logit);
        double sum = 0.0;
        for (size_t i = 0; i < cand_.size(); ++i) sum += p_[i] = std::exp((double) (cand_[i].logit - mx));
        for (double& p : p_) p /= sum;
    }

    std::vector<Cand> cand_;
    std::vector<float> pen_;
    std::vector<double> p_;
    std::unordered_map<int32_t, int> seen_;
};

bool parse_ids(const char* s, std::vector<int32_t>& out) {
    out.clear();
    while (*s == ' ') ++s;
    while (*s != '\0') {
        char* end = nullptr;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s || v < 0 || v > INT32_MAX) return false;
        out.push_back((int32_t) v);
        s = end;
        while (*s == ' ') ++s;
        if (*s == ',') ++s;
        else if (*s != '\0') return false;
    }
    return !out.empty();
}

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// A request's pictures: strata-vision's records (tools/vision/strata_vision.cpp), one after the other in the
// order of the prompt.
struct Pictures {
    std::vector<float> rows;
    std::vector<strata::poly::Image> at;
};

// Reads a request's pictures and finds their places: each takes the next run of `pad` in ids, a token a row.
// Those ids then become numbers below zero made from the picture, so that the conversation cache tells a picture
// from another one of the same size.  Returns what is wrong, or nothing.
std::string take_pictures(const std::string& path, int32_t pad, int n_embd, std::vector<int32_t>& ids, Pictures& out) {
    struct Record {
        int n, nx;
        size_t off;
    };
    std::vector<Record> records;
    out = Pictures{};
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string err;
    for (;;) {
        int32_t hdr[5];   // 'SVE1', rows, rows of a line, lines, a row's floats
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[1] > 1 << 20 || hdr[2] < 1 || hdr[3] < 1 ||
            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != n_embd) {
            err = "bad embeddings file (expected strata-vision records of width " + std::to_string(n_embd) + ")";
            break;
        }
        const size_t off = out.rows.size(), count = (size_t) hdr[1] * (size_t) n_embd;
        out.rows.resize(off + count);
        if (std::fread(out.rows.data() + off, sizeof(float), count, f) != count) {
            err = "short embeddings file";
            break;
        }
        records.push_back({hdr[1], hdr[2], off});
    }
    std::fclose(f);
    if (!err.empty()) return err;

    size_t i = 0;
    for (size_t k = 0; k < records.size(); ++k) {
        const Record& r = records[k];
        while (i < ids.size() && ids[i] != pad) ++i;
        for (int j = 0; j < r.n; ++j)
            if (i + (size_t) j >= ids.size() || ids[i + (size_t) j] != pad)
                return "picture " + std::to_string(k + 1) + " has " + std::to_string(r.n) +
                       " rows and the prompt fewer places for it";
        const float* rows = out.rows.data() + r.off;
        const int32_t grid[2] = {r.n, r.nx};
        const uint64_t h = fnv1a(rows, (size_t) r.n * (size_t) n_embd * sizeof(float), fnv1a(grid, sizeof grid));
        for (int j = 0; j < r.n; ++j)
            ids[i + (size_t) j] = -1 - (int32_t) ((h + (uint64_t) j * 0x9E3779B97F4A7C15ull) >> 33);
        out.at.push_back({(int64_t) i, r.n, r.nx, rows});
        i += (size_t) r.n;
    }
    if (std::find(ids.begin() + (std::ptrdiff_t) i, ids.end(), pad) != ids.end())
        return "the prompt has more pictures than the embeddings file";
    if (!ids.empty() && ids.back() < 0) return "a prompt cannot end in a picture";
    return {};
}

// stdin on its own thread, so a STOP is seen while a request runs
class Lines {
public:
    Lines() : th_([this] { read(); }) { th_.detach(); }

    std::string pop() {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [this] { return !q_.empty(); });
        std::string s = std::move(q_.front());
        q_.pop_front();
        return s;
    }

    std::atomic<bool> stop{false};

private:
    void read() {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") stop.store(true);
            else push(l);
        }
        push("QUIT");
    }
    void push(std::string s) {
        { std::lock_guard<std::mutex> l(m_); q_.push_back(std::move(s)); }
        cv_.notify_one();
    }

    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    std::thread th_;
};

// What the conversation cache is told on the command line.
struct CacheFlags {
    CacheOptions o;
    long long min_free_mib = 2560;   // parking leaves this much physical RAM free
    int64_t message = 0;             // --prompt-cache-message: the longest last message given a checkpoint; 0: none
    std::string disk;                // --conversation-disk: the folder; empty: none
    long long disk_mib = 0;          // what its files may take; 0: kDiskMib, or park_mib when that is less
    long long disk_min_free_mib = 10240;   // and they leave this much of the volume free
};

struct Session {
    Engine& engine;
    std::vector<int32_t> eos;
    CacheFlags flags;
    Conversations<Engine> kept;   // the conversation in the engine, and the parked ones
    Sampler sampler;
    std::mt19937_64 rng{std::random_device{}()};
    const std::vector<int32_t>* follow = nullptr;   // --follow: the answer's tokens, taken in place of the picks
    const float* picked_from = nullptr;             // the logits of the last token picked
    long long asked = 0, reused = 0;   // prompt tokens of every request, and those of them not read again
    // Guesses at the tokens to come, checked a window at a time (Engine::window).  They come from the block some
    // files carry for it (Engine::draft), and from the conversation: when the last tokens repeat an earlier stretch
    // of it, what followed there is the guess.  The policy learns from the windows so far, their hits and their
    // time, which of the two pays.  It and the repeats are the other engine's (strata/spec).
    int most;                             // the tokens of a window at most; 1: no guesses
    bool look;                            // guess from repeats
    strata::spec::SuffixDrafter repeats;
    strata::spec::DraftPolicy policy;
    Guesses guesses;                      // whether the block is asked in a round, and for how many (guesses.hpp)
    // --suffix-drafts: the guesses from repeats come from every prompt and answer since the start and from the
    // files of --suffix-corpus, with how sure the places they come from are, in place of the conversation's own
    // repeats (suffix_index.hpp)
    SuffixIndex* suffix;

    Session(Engine& e, std::vector<int32_t> eos_ids, const CacheFlags& f, int lookup, SuffixIndex* sx = nullptr)
        : engine(e), eos(std::move(eos_ids)), flags(f), kept(e, with_floor(f)),
          most(lookup > 0 || sx || e.drafts() ? std::min<int>(e.max_window(), strata::spec::DraftPolicy::kMaxT) : 1),
          look((lookup > 0 || sx) && most > 1),
          repeats(std::max(lookup, 3), 64, (size_t) e.max_context() + 4096), policy(most), suffix(sx) {}

    static CacheOptions with_floor(const CacheFlags& f) {
        CacheOptions o = f.o;
        o.log = stderr;
        o.admit = [floor = (uint64_t) f.min_free_mib << 20](size_t bytes) {
            return strata::core::conversation_memory_admit(strata::core::available_host_bytes(), bytes, floor);
        };
        return o;
    }

    bool is_eos(int32_t t) const { return std::find(eos.begin(), eos.end(), t) != eos.end(); }

    struct Result {
        int64_t produced = 0, resume = 0, read = 0;
        int64_t guessed = 0, guessed_right = 0;
        // --suffix-drafts: the lookups, those that gave a draft, the windows that took one, their guesses and those
        // right, and the time the lookups took
        int64_t suffix_lookups = 0, suffix_offers = 0, suffix_windows = 0, suffix_guessed = 0, suffix_right = 0;
        double suffix_ms = 0;
        double prompt_ms = 0, decode_ms = 0;
        const char* finish = "length";
    };

    // Reads the prompt and generates.  on_token gets every token, the end-of-sequence one included.  The rows of
    // the prompt's pictures have ids below zero (take_pictures()); `pad` is the token they stand in for.
    template<class OnToken, class OnProgress>
    Result generate(const std::vector<int32_t>& ids, const Pictures& pictures, int32_t pad, int64_t turn_at,
                    int64_t max_new, const Sampling& s, const std::atomic<bool>& stop, OnToken on_token,
                    OnProgress on_progress) {
        Result r;
        const int64_t n = (int64_t) ids.size();
        Clock::time_point t0 = Clock::now(), shown = t0;   // bringing a parked conversation back is prompt time
        engine.images(pictures.at.data(), (int) pictures.at.size());
        r.resume = kept.resume(ids);
        kept.turn_at(turn_at);
        if (flags.message > 0) kept.message_at(strata::poly::last_message(ids, r.resume, turn_at, flags.message));
        asked += n;
        reused += r.resume;
        on_progress(r.resume, n, 0.0, true);
        if (s.seed != 0) rng.seed(s.seed);

        // The last token goes in alone, for the logits that come with it.  The tokens before it go in as many at
        // a time as the engine takes.  A prompt without pictures goes in several calls' worth at once, which the
        // engine reads faster than the calls (read()), and the checkpoints due inside are the engine's to write.
        // One with pictures goes in a call at a time up to the next checkpoint, a picture whole.
        const bool text = pictures.at.empty();
        std::vector<std::pair<int64_t, uint8_t*>> due;
        std::vector<Engine::Mark> marks;
        for (int64_t i = r.resume; i < n;) {
            if (stop.load()) {
                r.finish = "cancel";
                break;
            }
            const bool last = i == n - 1;
            const int64_t room = std::min<int64_t>(n - 1 - i, text ? engine.max_read() : engine.max_rows());
            const int64_t m = last ? 1 : text ? room : engine.whole(i, kept.until_checkpoint(i, room));
            kept.before(i, last);
            if (text && !last) {
                kept.inside(i, m, due);
                marks.clear();
                for (const auto& d : due) marks.push_back({(int) d.first, d.second});
                // A long prompt is one read.  Between its layers the read says how far it is, and a request that
                // was stopped gives it up there: the conversation is then at its last checkpoint.
                const std::function<bool(int, int)> turn = [&](int done, int of) {
                    if (stop.load()) return false;
                    if (ms_since(shown) >= 500.0) {
                        on_progress(i + m * done / of, n, ms_since(t0), false);
                        shown = Clock::now();
                    }
                    return true;
                };
                if (!engine.read(ids.data() + i, (int) m, marks.data(), (int) marks.size(), &turn)) {
                    kept.rewind();
                    r.finish = "cancel";
                    break;
                }
            } else {
                engine.eval(ids.data() + i, (int) m, last);
            }
            kept.read(ids.data() + i, m);
            i += m;
            r.read += m;
            if (ms_since(shown) >= 500.0 || last) {
                on_progress(i, n, ms_since(t0), false);
                shown = Clock::now();
            }
        }
        kept.prompt_was(n);
        r.prompt_ms = ms_since(t0);
        if (std::strcmp(r.finish, "cancel") == 0) return r;

        t0 = Clock::now();
        std::vector<int32_t> history = kept.seq();
        std::replace_if(history.begin(), history.end(), [](int32_t t) { return t < 0; }, pad);
        if (look && suffix) {
            suffix->begin(history);
        } else if (look) {
            repeats.reset();
            repeats.append(history.data(), history.size());
        }
        const auto pick = [&](const float* logits) {
            picked_from = logits;
            if (follow && r.produced < (int64_t) follow->size()) return (*follow)[(size_t) r.produced];
            return (int32_t) sampler.pick(logits, engine.n_vocab(), s, history, rng);
        };
        // Shows a token.  False when the answer ends with it.
        const auto show = [&](int32_t t) {
            on_token(t);
            ++r.produced;
            history.push_back(t);
            if (look && suffix) suffix->append(t);
            else if (look) repeats.append(t);
            if (is_eos(t)) r.finish = "stop";
            else if (stop.load()) r.finish = "cancel";
            else return r.produced < max_new;
            return false;
        };
        if (max_new > 0) {
            int32_t w[strata::spec::DraftPolicy::kMaxT], d[strata::spec::DraftPolicy::kMaxT];
            int32_t t = pick(engine.logits());   // picked, and not shown yet
            while (show(t)) {
                // t goes into the model now, and with it guesses at what follows when there are any that pay
                const Clock::time_point round = Clock::now();
                const bool room = engine.pos() + most <= engine.max_context();
                const int offered = !(look && room) ? 0 : suffix ? lookup(w + 1, r) : repeats.propose(most - 1, w + 1);
                const int match = suffix ? suffix->last_match() : repeats.last_match();
                const int widest = match >= kCopyMatch ? most : std::min(most, kBlockWindow);   // this round's window
                const int k = std::min(offered, widest - 1);
                // the suffix index's chance that its first i + 1 tokens are right, by the places it saw
                const double* sure = suffix && k > 0 ? suffix->sure() : nullptr;
                const double q = policy.lookup_rate(match);
                // A repeat that fills the window, from a match of a length that has been right nine times in ten,
                // is taken as it is: the block would mostly say the same, and asking it takes time.  Otherwise the
                // block guesses, each guess from the one before, while a guess more pays (Guesses).
                const bool blocks = engine.drafts() && room && !(k == widest - 1 && (sure ? q * sure[k - 1] : q) >= 0.9);
                const bool ask = guesses.begin(blocks, policy);
                // the round's time, as the policy counts it: a window of `in` tokens wrote n of them
                const auto count = [&](int in, int n) {
                    return guesses.end(in, n, ms_since(round), engine.took_record());
                };
                strata::spec::DraftPolicy::Pick use = policy.choose(1, k, match, sure);
                if (ask) {
                    int n_d = 0;
                    for (int32_t from = t; n_d < std::min(most, kBlockWindow) - 1 && guesses.wants(n_d); from = d[n_d++]) {
                        float p = 0.f;
                        const Clock::time_point asked_at = Clock::now();
                        d[n_d] = engine.draft(from, n_d, &p);
                        if (!guesses.takes(n_d, p, ms_since(asked_at))) break;
                    }
                    use = policy.choose(1 + n_d, k, match, sure);
                }
                if (use.t == 1) {
                    engine.eval(&t, 1, true);
                    kept.read(&t, 1);
                    t = pick(engine.logits());
                    policy.observe(false, 1, 0, 0, count(1, 1));
                    continue;
                }
                w[0] = t;
                if (!use.lookup) std::copy(d, d + use.t - 1, w + 1);
                engine.window(w, use.t);
                // logits_of(i) is what follows w[i].  A guess that is what was picked has been through the model
                // already; the token an answer ends with is shown and never fed, as without guesses.
                int right = 0;
                bool more = true;
                t = pick(engine.logits_of(0));
                while (right < use.t - 1 && t == w[right + 1] && (more = show(t))) t = pick(engine.logits_of(++right));
                engine.keep(right + 1);
                kept.read(w, right + 1);
                r.guessed += use.t - 1;
                r.guessed_right += right + (more ? 0 : 1);
                if (suffix && use.lookup) {
                    ++r.suffix_windows;
                    r.suffix_guessed += use.t - 1;
                    r.suffix_right += right + (more ? 0 : 1);
                }
                policy.observe(use.lookup, use.t, right, match, count(use.t, right + 1));
                if (!more) break;
            }
        }
        r.decode_ms = ms_since(t0);
        return r;
    }

    // a draft of the suffix index, timed
    int lookup(int32_t* out, Result& r) {
        const Clock::time_point t0 = Clock::now();
        const int n = suffix->propose(most - 1, out);
        r.suffix_ms += ms_since(t0);
        ++r.suffix_lookups;
        r.suffix_offers += n > 0;
        return n;
    }

    // a line on stderr of what the suffix index did for an answer, and what it holds
    void say_suffix(const Result& r) const {
        if (!suffix) return;
        std::fprintf(stderr, "strata-poly: suffix drafts: %lld of %lld lookups gave one, %lld windows took one, %lld "
                             "of their %lld guesses right; %.2f us a lookup; the index holds %zu of %zu tokens "
                             "(%zu MiB) and %zu MiB of sequences seen\n",
                     (long long) r.suffix_offers, (long long) r.suffix_lookups, (long long) r.suffix_windows,
                     (long long) r.suffix_right, (long long) r.suffix_guessed,
                     r.suffix_lookups > 0 ? 1000.0 * r.suffix_ms / (double) r.suffix_lookups : 0.0, suffix->held(),
                     suffix->capacity(), suffix->bytes() >> 20, suffix->seen_bytes() >> 20);
    }
};

// GEN <max_new> [key=value ...] <id,id,...>, or GENI with the file of the prompt's pictures ahead of the ids and
// the key image_pad, the token whose runs in the ids are the pictures' places.  turn_at is the place in the ids
// where the prompt's last turn begins (the server knows the template's turn tokens): a checkpoint goes there.
bool parse_gen(const std::string& line, long long& max_new, Sampling& s, int32_t& pad, int64_t& turn_at,
               std::string& pictures, std::vector<int32_t>& ids) {
    const bool geni = line.rfind("GENI ", 0) == 0;
    char* p = nullptr;
    max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &p, 10);
    if (max_new < 1) return false;
    pad = -1;
    turn_at = -1;
    for (;;) {   // key=value tokens, then the file and the ids
        while (*p == ' ') ++p;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        const std::string tok(start, (size_t) (p - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            p = const_cast<char*>(start);
            break;
        }
        const std::string key = tok.substr(0, eq);
        const char* v = tok.c_str() + eq + 1;
        if (key == "temperature") s.temperature = std::strtof(v, nullptr);
        else if (key == "top_p") s.top_p = std::strtof(v, nullptr);
        else if (key == "top_k") s.top_k = std::atoi(v);
        else if (key == "min_p") s.min_p = std::strtof(v, nullptr);
        else if (key == "penalty_last_n") s.penalty_last_n = std::atoi(v);
        else if (key == "penalty_repeat") s.penalty_repeat = std::strtof(v, nullptr);
        else if (key == "penalty_freq") s.penalty_freq = std::strtof(v, nullptr);
        else if (key == "penalty_present") s.penalty_present = std::strtof(v, nullptr);
        else if (key == "seed") s.seed = std::strtoull(v, nullptr, 10);
        else if (key == "image_pad") pad = (int32_t) std::atoi(v);
        else if (key == "turn_at") turn_at = std::strtoll(v, nullptr, 10);
        // other keys belong to the other engine and are skipped
    }
    pictures.clear();
    if (geni) {
        const char* end = std::strchr(p, ' ');
        if (!end || end == p || pad < 0) return false;
        pictures.assign(p, (size_t) (end - p));
        p = const_cast<char*>(end);
    }
    return parse_ids(p, ids);
}

// --conversation-disk, after an answer: the conversation in the engine is held in RAM up to its last checkpoint,
// and what the cache holds is handed to the disk's thread, which writes it while the next request is waited for.
void keep_on_disk(Session& ss) {
    ConversationDisk* disk = ss.flags.o.disk;
    if (!disk) return;
    try {
        const Clock::time_point t0 = Clock::now();
        ss.kept.hold();
        const size_t n = ss.kept.store();
        std::fprintf(stderr, "strata-poly: conversation disk: %zu conversations there (%llu MiB), %llu MiB to write; "
                             "%.1f ms\n", n, (unsigned long long) (disk->held_bytes() >> 20),
                     (unsigned long long) (disk->queued_bytes() >> 20), ms_since(t0));
    } catch (const std::exception& e) {   // the card's, from copying the rows: this answer is not kept on disk
        std::fprintf(stderr, "strata-poly: conversation disk: %s\n", e.what());
    }
}

// --conversation-disk, at the end: what went to disk and came from it.
void say_disk(Session& ss) {
    ConversationDisk* disk = ss.flags.o.disk;
    if (!disk) return;
    const Clock::time_point t0 = Clock::now();
    disk->flush();
    const ConversationDisk::Totals t = disk->totals();
    std::fprintf(stderr, "strata-poly: conversation disk: wrote %.1f MiB in %llu files in %.1f s (%.1f s waited for at "
                         "the end), deleted %llu; read %.1f MiB in %.1f s\n", (double) t.written / (1 << 20),
                 (unsigned long long) t.files, t.write_ms / 1000.0, ms_since(t0) / 1000.0,
                 (unsigned long long) t.deleted, (double) t.read / (1 << 20), t.read_ms / 1000.0);
}

int serve(Session& ss) {
    Engine& engine = ss.engine;
    std::printf("INFO context=%d expert_slots=%d expert_cache_mib=%lld pool_workers=%d conversation_cache_mib=%lld "
                "conversation_cache_slots=%d conversation_cache_min_free_mib=%lld vram_free_mib=%lld vram_elastic=1 "
                "engine=" STRATA_VERSION "\n",
                engine.max_context(), engine.cached_experts(), (long long) (engine.cached_gib() * 1024.0),
                engine.threads(), (long long) ss.flags.o.park_mib, ss.flags.o.park_slots, ss.flags.min_free_mib,
                (long long) engine.vram_free_mib());
    std::printf("READY %d stop\n", engine.max_context());
    std::fflush(stdout);

    static Lines lines;   // its thread may still be in a read when this function returns
    Pictures pictures;    // the engine reads the last request's until the next one's are set
    // the experts' use (--expert-profile): as if it was last saved long ago, so that the first answer saves it
    Clock::time_point saved = Clock::now() - std::chrono::milliseconds((long long) kProfileSaveMs);
    for (;;) {
        const std::string line = lines.pop();
        if (line == "QUIT") break;
        if (line.empty()) continue;
        lines.stop.store(false);   // a STOP that came between requests is stale
        long long max_new = 0;
        Sampling s;
        std::vector<int32_t> ids;
        int32_t pad = -1;
        int64_t turn_at = -1;
        std::string file;
        // VRAM [reserve_mib], the server's POST /v1/vram: the expert cache sized again between requests, to leave
        // that much VRAM to other programs; alone, back to what the engine started with
        if (line.rfind("VRAM", 0) == 0 && (line.size() == 4 || line[4] == ' ')) {
            char* end = nullptr;
            const bool bare = line.find_first_not_of(' ', 4) == std::string::npos;
            const long long reserve = bare ? -1 : std::strtoll(line.c_str() + 4, &end, 10);
            if (!bare && (end == line.c_str() + 4 || reserve < 0 || reserve > (1LL << 24))) {
                std::printf("ERR expected: VRAM [reserve_mib]\n");
                std::fflush(stdout);
                continue;
            }
            try {
                const Clock::time_point t0 = Clock::now();
                const int before = engine.cached_experts();
                engine.resize_cache((int) reserve);
                const long long free_mib = (long long) engine.vram_free_mib();
                std::fprintf(stderr, "strata-poly: VRAM %d MiB kept free: the expert cache %d -> %d of %d experts "
                                     "(%.2f of %.2f GiB), %lld MiB free now, in %.0f ms\n",
                             engine.reserve_mib(), before, engine.cached_experts(), engine.full_experts(),
                             engine.cached_gib(), engine.full_gib(), free_mib, ms_since(t0));
                std::printf("VRAM reserve_mib=%d expert_slots=%d expert_slots_full=%d expert_cache_mib=%lld "
                            "expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%d\n",
                            engine.reserve_mib(), engine.cached_experts(), engine.full_experts(),
                            (long long) (engine.cached_gib() * 1024.0), (long long) (engine.full_gib() * 1024.0),
                            free_mib, engine.max_rows());
            } catch (const std::exception& e) {
                std::printf("ERR VRAM: %s\n", e.what());
            }
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("GEN ", 0) != 0 && line.rfind("GENI ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> image_pad=<id> <file> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        if (!parse_gen(line, max_new, s, pad, turn_at, file, ids)) {
            std::printf("ERR bad request\n");
            std::fflush(stdout);
            continue;
        }
        const long long n = (long long) ids.size();
        if (n + max_new > engine.max_context()) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%d)\n", n, max_new,
                        engine.max_context());
            std::fflush(stdout);
            continue;
        }
        if (std::any_of(ids.begin(), ids.end(), [&](int32_t t) { return t >= engine.n_vocab(); })) {
            std::printf("ERR a token id is outside the vocabulary\n");
            std::fflush(stdout);
            continue;
        }
        std::string wrong;
        if (file.empty()) pictures = Pictures{};
        else if (!engine.sees()) wrong = "this model does not read pictures";
        else wrong = take_pictures(file, pad, engine.n_embd(), ids, pictures);
        if (!wrong.empty()) {
            std::printf("ERR %s\n", wrong.c_str());
            std::fflush(stdout);
            continue;
        }

        const strata::poly::Stats before = engine.stats();
        int64_t resume = 0;
        Session::Result r;
        try {
            const int had = engine.cached_experts();
            if (engine.mind_vram())
                std::fprintf(stderr, "strata-poly: the expert cache %d -> %d experts (%.2f GiB) for the VRAM free "
                                     "now, %lld MiB left\n",
                             had, engine.cached_experts(), engine.cached_gib(), (long long) engine.vram_free_mib());
            r = ss.generate(
                ids, pictures, pad, turn_at, max_new, s, lines.stop,
                [](int32_t t) {
                    std::printf("T %d\n", (int) t);
                    std::fflush(stdout);
                },
                [&](int64_t at, int64_t total, double ms, bool first) {
                    if (first) {
                        resume = at;
                        std::printf("RESUME %lld\n", (long long) at);
                    } else {
                        std::printf("PP %lld %lld %.0f %.1f\n", (long long) at, (long long) total, ms,
                                    ms > 0.0 ? 1000.0 * (double) (at - resume) / ms : 0.0);
                    }
                    std::fflush(stdout);
                });
        } catch (const std::exception& e) {
            // the engine's state is unknown now: the conversation it was reading is lost
            ss.kept.drop();
            try { engine.reset(); } catch (const std::exception&) {}
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
            continue;
        }
        // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused>
        //      <experts from VRAM> <experts> <RAM blobs> <file blobs> <file MB> <prompt tokens read> <offloaded>
        const strata::poly::Stats& now = engine.stats();
        // the cache's share of the experts looked up while answering: a prompt batch's go through VRAM either way
        const long long hits = now.gpu_experts - before.gpu_experts;
        const long long misses = now.cpu_experts - before.cpu_experts;
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld 0 0 0.0 %lld 0\n", (long long) r.produced, n,
                    r.prompt_ms, r.decode_ms, r.finish, (long long) r.guessed_right, (long long) r.guessed,
                    (long long) r.resume, hits, hits + misses, (long long) r.read);
        std::fflush(stdout);
        ss.say_suffix(r);
        keep_on_disk(ss);
        if (ms_since(saved) >= kProfileSaveMs) {
            try {
                engine.save_profile();
            } catch (const std::exception& e) {   // the next start's cache is the worse for it, this run is not
                std::fprintf(stderr, "strata-poly: %s\n", e.what());
            }
            saved = Clock::now();
        }
    }
    if (const strata::poly::Stats& st = engine.stats(); st.loans > 0)
        std::fprintf(stderr, "strata-poly: --lend-staging: lent for %lld answers, %.1f ms a loan to lend and %.1f to "
                             "take back\n", (long long) st.loans, st.lend_ms / (double) st.loans,
                     st.end_ms / (double) st.loans);
    std::fprintf(stderr, "strata-poly: %lld of %lld prompt tokens were not read again (%.1f%%); %zu conversations "
                         "parked in %zu MiB, %zu pushed out; %lld MiB of VRAM free at the end\n",
                 ss.reused, ss.asked, ss.asked > 0 ? 100.0 * (double) ss.reused / (double) ss.asked : 0.0,
                 ss.kept.parked(), ss.kept.parked_bytes() >> 20, ss.kept.evictions(), (long long) engine.vram_free_mib());
    say_disk(ss);
    return 0;
}

// The n most likely next tokens with their probabilities, to hold against another engine's on the same prompt.
void print_top(const float* logits, int n_vocab, int n) {
    std::vector<int> order((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) order[(size_t) i] = i;
    n = std::min(n, n_vocab);
    std::partial_sort(order.begin(), order.begin() + n, order.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });
    const float top = logits[order[0]];
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum += std::exp((double) (logits[i] - top));
    std::fprintf(stderr, "top");
    for (int i = 0; i < n; ++i)
        std::fprintf(stderr, " %d:%.6g", order[(size_t) i], std::exp((double) (logits[order[(size_t) i]] - top)) / sum);
    std::fprintf(stderr, "\n");
}

int once(Session& ss, const std::vector<int32_t>& ids, const Pictures& pictures, int32_t pad, long long max_new,
         const Sampling& s, int top_probs) {
    std::atomic<bool> stop{false};
    std::string out;
    const Session::Result r = ss.generate(
        ids, pictures, pad, -1, max_new, s, stop,
        [&](int32_t t) {
            if (top_probs > 0) print_top(ss.picked_from, ss.engine.n_vocab(), top_probs);
            out += (out.empty() ? "" : ",") + std::to_string(t);
        },
        [](int64_t, int64_t, double, bool) {});
    std::printf("%s\n", out.c_str());
    keep_on_disk(ss);
    say_disk(ss);
    const strata::poly::Stats& st = ss.engine.stats();
    const double experts = (double) std::max<int64_t>(st.gpu_experts + st.cpu_experts, 1);
    std::fprintf(stderr,
                 "prompt %lld tokens %.1f tok/s, generated %lld tokens %.2f tok/s (%s); experts from VRAM %.1f%%, "
                 "CPU experts %.1f ms/token, router wait %.1f ms/token, %lld experts swapped into VRAM, "
                 "%lld left out, %lld of %lld guessed tokens right\n",
                 (long long) r.read, r.prompt_ms > 0 ? 1000.0 * (double) r.read / r.prompt_ms : 0.0,
                 (long long) r.produced, r.decode_ms > 0 ? 1000.0 * (double) (r.produced - 1) / r.decode_ms : 0.0,
                 r.finish, 100.0 * (double) st.gpu_experts / experts, st.cpu_ms / (double) std::max<int64_t>(st.tokens, 1),
                 st.route_wait_ms / (double) std::max<int64_t>(st.tokens, 1), (long long) st.swapped_experts,
                 (long long) st.tail_experts, (long long) r.guessed_right, (long long) r.guessed);
    if (st.deferred_experts > 0)
        std::fprintf(stderr, "%lld experts deferred to the next layer\n", (long long) st.deferred_experts);
    ss.say_suffix(r);
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "usage: strata-poly --model FILE.gguf (--serve | --tokens ID,ID,... [--max-new N])\n"
                 "  --max-context N        tokens the attention caches hold (default 8192)\n"
                 "  --threads N            CPU threads for the experts that are not in VRAM (default: all usable)\n"
                 "  --cpu-multi            the CPU's Q4_K, Q5_K and Q6_K experts with this engine's AVX2 rows: a row\n"
                 "                         is read and unpacked once for every token of a window that uses its\n"
                 "                         expert, the down products in wider tasks; the same bits as ggml-cpu's\n"
                 "                         (default: off)\n"
                 "  --vram-reserve-mib N   VRAM left free beside the expert cache, 256 at least (default: a\n"
                 "                         thirty-second of the card, 256 to 768, or a sixteenth, 512 to 1024,\n"
                 "                         when the PC's free RAM is within 4 GiB of the model's files, and then\n"
                 "                         kept whole; else half of it after the first request)\n"
                 "  --layer-split S        the model's layers across the CUDA devices this process sees, in their\n"
                 "                         order: S is auto (by each card's free VRAM) or the first layer of each\n"
                 "                         later card, as 20 or 20,30 (default: the first card alone)\n"
                 "  --expert-profile FILE  expert use counts: read to choose the cached experts, written back on\n"
                 "                         exit and by --serve after an answer, every 10 s at most\n"
                 "  --no-gpu-experts       keep every routed expert on the CPU\n"
                 "  --any-node             on a PC with several NUMA nodes, let the CPU threads run on any of them\n"
                 "                         (default: on the node whose memory holds most of the experts)\n"
                 "  --window N             the most tokens that go through the model at once while an answer is\n"
                 "                         written: the one that is known and guesses at the ones after it, which\n"
                 "                         count when they turn out right (1 = no guesses; default 8, of which\n"
                 "                         the block guesses 3, and 4 for GLM 5.3 Flash)\n"
                 "  --lookup N             guess what followed an earlier repeat of the last N or more tokens\n"
                 "                         (default 3, 0 = never)\n"
                 "  --suffix-drafts MIB    guess from repeats of every prompt and answer since the start (and of\n"
                 "                         --suffix-corpus): what most places that went on like the last tokens\n"
                 "                         had next, in an index of about MIB MiB that drops its oldest text\n"
                 "                         first; in place of --lookup (SuffixDecoding; default 0 = off)\n"
                 "  --suffix-corpus FILE   with --suffix-drafts: token ids to guess from as well, a sequence a line,\n"
                 "                         ids separated by commas; may be given more than once\n"
                 "  --no-draft             do not run the block that guesses ahead\n"
                 "  --draft-model FILE     a file of the same model to take that block from, when --model has none;\n"
                 "                         for Gemma 4, its draft model (a gemma4-assistant file)\n"
                 "  --draft-vocab N        the block guesses among the vocabulary's first N tokens (default:\n"
                 "                         131072 for Qwen3.6, Ornith and Tiel, all of it for the others; 0 = all)\n"
                 "  --no-graphs            launch a token's kernels one by one (default: a layer's are recorded once\n"
                 "                         as a CUDA graph and replayed)\n"
                 "  --kv-whole             give the attention caches VRAM for the whole context at load (default: as\n"
                 "                         the conversation grows, the expert cache having the rest until then)\n"
                 "  --kv-q8                Qwen3.5 / 3.6: the attention caches in Q8_0 (an fp16 scale a 32 values)\n"
                 "                         instead of fp16, 53% of the VRAM a token (default: fp16)\n"
                 "  --kv-q4                Qwen3.5 / 3.6: the attention caches' values in Q4_0 and their keys in Q8_0,\n"
                 "                         41% of fp16's VRAM a token, for a context the card cannot hold with\n"
                 "                         --kv-q8; lossy (default: fp16)\n"
                 "  --kv-q4-keys q8|q4     --kv-q4 with its keys in Q8_0 (as --kv-q4 alone) or in Q4_0 as well,\n"
                 "                         28% of fp16's VRAM a token\n"
                 "  --attn-split           Qwen3.5 / 3.6: the attention of a few tokens (an answer's) as one pass over\n"
                 "                         the caches on the tensor cores, NVIDIA from Turing on; faster at long\n"
                 "                         contexts, and the answers can move a little (default: off)\n"
                 "  --graph-experts        Qwen3.5 / 3.6 on one NVIDIA card from Volta on: an answer's layers each as\n"
                 "                         one recorded launch with their cached experts, the card and the CPU taking\n"
                 "                         turns through pinned memory: the same sums with a graph launch a layer\n"
                 "                         where the engine's thread made some ten CUDA calls (default: off)\n"
                 "  --expert-swaps N       while an answer is written, up to N experts it keeps using are on their way\n"
                 "                         into VRAM, in place of cached ones it uses less (default 64, 0 = the cache\n"
                 "                         stays as it was loaded)\n"
                 "  --prompt-batch N       prompt tokens read at a time (default 1024); about 1.4 MiB of VRAM each\n"
                 "  --elastic-prefill N    Qwen3.5 / 3.6 on one card: a prompt of N tokens or more is read in a wider\n"
                 "                         batch, in VRAM the expert cache lends it until the answer starts, and gets\n"
                 "                         back with the same experts (default 0 = never)\n"
                 "  --elastic-batch N      the widest such batch, as wide as the VRAM free and the cache allow\n"
                 "                         (default 4096)\n"
                 "  --lend-staging         Qwen3.5 / 3.6 on one card: while an answer is written, the VRAM the experts\n"
                 "                         outside the cache pass through in a prompt batch holds more cached\n"
                 "                         experts, and a prompt batch gets it back first (default: off)\n"
                 "  --stage-mib N          VRAM the experts outside the cache pass through in a prompt batch (default\n"
                 "                         2048, or what a layer's experts take when that is less); with less, more\n"
                 "                         of them are copied after the layer has routed\n"
                 "  --stage-ahead N        from N tokens, a batch copies those experts before it routes (default 1024,\n"
                 "                         0 = never)\n"
                 "  --stage-next           Qwen3.5 / 3.6 on one card: a prompt of that many tokens or more gets a\n"
                 "                         second such buffer, which the expert cache lends it until the answer\n"
                 "                         starts, and the next layer's experts are copied into it while a layer\n"
                 "                         runs (default: off)\n"
                 "  --cpu-share F          under that, the share of those experts the CPU computes meanwhile, the\n"
                 "                         ones the fewest tokens chose (default: as the times say; 0 = none)\n"
                 "  --prompt-cache N       checkpoints kept of a conversation, about 150 MB of RAM each (default 6;\n"
                 "                         0 = a prompt is read from its start unless it goes on from the last token)\n"
                 "  --prompt-cache-every N a checkpoint every N prompt tokens, besides the one at a prompt's end\n"
                 "                         (default 2048, 0 = none)\n"
                 "  --prompt-cache-message N   and one where the prompt's last message begins, before the reply's\n"
                 "                         turn (the server's turn_at), when that message is N tokens or fewer, kept\n"
                 "                         until the next prompt: a gateway that ends every prompt with a message the\n"
                 "                         next one leaves out is taken up there (default 0 = none)\n"
                 "  --conversation-cache-mib N   RAM for parked conversations, so that several can take turns\n"
                 "                         without being read again (default 0 = off)\n"
                 "  --conversation-cache-slots N        parked conversations at most (default 4)\n"
                 "  --conversation-cache-min-tokens N   a conversation shorter than this is not parked (default 1024)\n"
                 "  --conversation-cache-min-free-mib N park only while this much physical RAM stays free (default\n"
                 "                         2560)\n"
                 "  --conversation-disk DIR   keep what the conversation cache holds in DIR too, the conversation in\n"
                 "                         the engine up to its last checkpoint after every answer, written in the\n"
                 "                         background; the next engine of the same model, executable and options takes\n"
                 "                         it up, so a restart does not read the conversations again (default: off)\n"
                 "  --conversation-disk-mib N   what its files may take (default 4096, or\n"
                 "                         --conversation-cache-mib when that is less)\n"
                 "  --conversation-disk-min-free-mib N   and the disk keeps this much free beside them (default 10240)\n"
                 "  --eos-ids ID,ID,...    tokens that end a reply (default: the GGUF's eos, eot and eom)\n"
                 "  --temperature F --top-k N --top-p F --min-p F --seed N   sampling for --tokens (default greedy)\n"
                 "  --images FILE          with --tokens: the prompt's pictures, as strata-vision writes them\n"
                 "  --image-pad ID         the token whose runs in --tokens are the pictures' places, a token a row\n"
                 "  --dump-layers          print a checksum of every layer's stages to stderr\n"
                 "  --top-probs N          with --tokens: print the N most likely ids of every step to stderr\n"
                 "  --follow ID,ID,...     with --tokens: write these tokens in place of the ones picked, for\n"
                 "                         --top-probs of the same answer under other settings\n"
                 "                         (--tokens @FILE, --follow @FILE: the ids from a file)\n"
                 "  --tail-skip N          an expert outside VRAM that every token of a window ranks at place N+1\n"
                 "                         or later among its experts is left out: faster, and the answers move a\n"
                 "                         little (default 0 = none is; setup gives Qwen3.6 6)\n"
                 "  --weight-cut P         in place of --tail-skip: a token's experts outside VRAM after its first two\n"
                 "                         are left out once the ones it keeps hold P of its experts' weight, and\n"
                 "                         its weights are scaled to their sum again (default 0 = none is)\n"
                 "  --defer-experts K      with --graph-experts: a layer's experts outside VRAM that every token of a\n"
                 "                         window ranks at place K or later (the first is 1) are computed while the\n"
                 "                         card runs the next layer, and added at its end; not the last layer's\n"
                 "                         (Expert Deferral; default 0 = none is)\n"
                 "environment: STRATA_ROUTE_TAIL_SKIP=N  in place of --tail-skip (0 turns it off)\n");
}

// --conversation-disk: what the files there are valid for (conversation_disk.hpp), a line each, its first word
// saying what: the model's file and the draft model's (full path, size, last write), this engine's executable (its
// size and a hash of its bytes, read once at the start), what a state and a row are, and the options that change
// how they are computed.  Options that change only how fast (threads, --graph-experts, --cpu-multi, staging), what
// is guessed, or the cache's own sizes are not in it.
std::string disk_key(const Options& o, const Engine& e) {
    namespace fs = std::filesystem;
    const auto text = [](const fs::path& p) {
        const std::u8string u = p.u8string();
        return std::string(u.begin(), u.end());
    };
    const auto file = [&](const fs::path& given) {
        std::error_code ec;
        fs::path p = fs::absolute(given, ec);
        if (ec) p = given;
        if (const fs::path c = fs::weakly_canonical(p, ec); !ec) p = c;
        const uintmax_t size = fs::file_size(p, ec);
        const long long when = (long long) fs::last_write_time(p, ec).time_since_epoch().count();
        return text(p) + " " + std::to_string(ec ? 0 : size) + " " + std::to_string(when);
    };
    fs::path exe;
#ifdef _WIN32
    std::wstring w(32768, L'\0');
    w.resize(GetModuleFileNameW(nullptr, w.data(), (DWORD) w.size()));
    exe = w;
#else
    std::error_code ec;
    exe = fs::read_symlink("/proc/self/exe", ec);
#endif
    strata::core::SessionHasher h(0);
    uint64_t size = 0;
    if (std::ifstream f(exe, std::ios::binary); f) {
        std::vector<char> b((size_t) 16 << 20);
        while (f.read(b.data(), (std::streamsize) b.size()) || f.gcount() > 0) {
            h.update(b.data(), (size_t) f.gcount());
            size += (uint64_t) f.gcount();
        }
    }
    char line[512];
    std::string k = "strata-poly conversation disk 1\n";
    k += "model " + file(o.model) + "\n";
    if (!o.draft_model.empty()) k += "draft " + file(o.draft_model) + "\n";
    std::snprintf(line, sizeof line, "engine %llu %016llx\n", (unsigned long long) size, (unsigned long long) h.digest());
    k += line;
    std::snprintf(line, sizeof line, "cache state=%zu row=%zu kv_q8=%d kv_q4=%d context=%d\n", e.state_bytes(),
                  e.row_bytes(), (int) o.kv_q8, o.kv_q4, o.max_context);
    k += line;
    std::snprintf(line, sizeof line, "read prompt_batch=%d elastic_prefill=%d elastic_batch=%d\n", o.prompt_batch,
                  o.elastic_from, o.elastic_batch);
    k += line;
    std::snprintf(line, sizeof line, "compute tail_skip=%d weight_cut=%g defer_experts=%d attn_split=%d gpu_experts=%d "
                                     "layer_split=%s\n", o.tail_skip, (double) o.weight_cut, o.defer_from,
                  (int) o.attn_split, (int) o.gpu_experts, o.layer_split.c_str());
    k += line;
    return k;
}

}  // namespace

// Windows holds the threads of a process it takes for background work to the efficiency cores (power throttling):
// on a laptop with both kinds an answer after a long prompt came at 39 tokens/s where 58.  serve/winjob.py takes the
// engines the server starts out of it; this does the same for an engine anything else starts.
void no_power_throttling() {
#ifdef _WIN32
    for (const ULONG speed_and_timer : {0x1ul | 0x4ul, 0x1ul}) {   // Windows 10 knows the first bit only
        PROCESS_POWER_THROTTLING_STATE off{PROCESS_POWER_THROTTLING_CURRENT_VERSION, speed_and_timer, 0};
        if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &off, sizeof(off))) return;
    }
#endif
}

int main(int argc, char** argv) {
    no_power_throttling();
    Options o;
    CacheFlags cache;
    Sampling s;
    bool want_serve = false;
    long long max_new = 32;
    int top_probs = 0, lookup = 3, image_pad = -1;
    long long suffix_mib = 0;
    bool cpu_multi = false;
    const char* kv_q4 = nullptr;   // --kv-q4's keys: "q8" or "q4"
    std::vector<std::string> corpus;
    std::string images;
    std::vector<int32_t> ids, eos, follow;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-poly: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") o.model = val();
        else if (a == "--serve") want_serve = true;
        else if (a == "--max-context") o.max_context = std::atoi(val());
        else if (a == "--threads") o.threads = std::atoi(val());
        else if (a == "--cpu-multi") cpu_multi = true;
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoi(val());
        else if (a == "--expert-profile") o.profile = val();
        else if (a == "--no-gpu-experts") o.gpu_experts = false;
        else if (a == "--any-node") o.stay_near = false;
        else if (a == "--no-graphs") o.graphs = false;
        else if (a == "--kv-whole") o.kv_grow = false;
        else if (a == "--kv-q8") o.kv_q8 = true;
        else if (a == "--kv-q4") kv_q4 = kv_q4 ? kv_q4 : "q8";
        else if (a == "--kv-q4-keys") kv_q4 = val();
        else if (a == "--attn-split") o.attn_split = true;
        else if (a == "--graph-experts") o.graph_experts = true;
        else if (a == "--window") o.window = std::atoi(val());
        else if (a == "--lookup") lookup = std::atoi(val());
        else if (a == "--suffix-drafts") suffix_mib = std::atoll(val());
        else if (a == "--suffix-corpus") corpus.push_back(val());
        else if (a == "--no-draft") o.draft = false;
        else if (a == "--draft-model") o.draft_model = val();
        else if (a == "--expert-swaps") o.expert_swaps = std::atoi(val());
        else if (a == "--tail-skip") o.tail_skip = std::atoi(val());
        else if (a == "--weight-cut") o.weight_cut = (float) std::atof(val());
        else if (a == "--defer-experts") o.defer_from = std::atoi(val());
        else if (a == "--layer-split") o.layer_split = val();
        else if (a == "--draft-vocab") o.draft_vocab = std::atoi(val());
        else if (a == "--prompt-batch") o.prompt_batch = std::atoi(val());
        else if (a == "--elastic-prefill") o.elastic_from = std::atoi(val());
        else if (a == "--elastic-batch") o.elastic_batch = std::atoi(val());
        else if (a == "--lend-staging") o.lend_stage = true;
        else if (a == "--stage-mib") o.stage_mib = std::atoi(val());
        else if (a == "--stage-ahead") o.stage_ahead = std::atoi(val());
        else if (a == "--stage-next") o.stage_next = true;
        else if (a == "--cpu-share") o.cpu_share = (float) std::atof(val());
        else if (a == "--prompt-cache") cache.o.checkpoints = std::atoi(val());
        else if (a == "--prompt-cache-every") cache.o.every = std::atoll(val());
        else if (a == "--prompt-cache-message") cache.message = std::atoll(val());
        else if (a == "--conversation-cache-mib") cache.o.park_mib = std::atoll(val());
        else if (a == "--conversation-cache-slots") cache.o.park_slots = std::atoi(val());
        else if (a == "--conversation-cache-min-tokens") cache.o.park_min = std::atoll(val());
        else if (a == "--conversation-cache-min-free-mib") cache.min_free_mib = std::atoll(val());
        else if (a == "--conversation-disk") cache.disk = val();
        else if (a == "--conversation-disk-mib") cache.disk_mib = std::atoll(val());
        else if (a == "--conversation-disk-min-free-mib") cache.disk_min_free_mib = std::atoll(val());
        else if (a == "--dump-layers") o.dump = true;
        else if (a == "--images") images = val();
        else if (a == "--image-pad") image_pad = std::atoi(val());
        else if (a == "--top-probs") top_probs = std::atoi(val());
        else if (a == "--max-new") max_new = std::atoll(val());
        else if (a == "--temperature") s.temperature = std::strtof(val(), nullptr);
        else if (a == "--top-k") s.top_k = std::atoi(val());
        else if (a == "--top-p") s.top_p = std::strtof(val(), nullptr);
        else if (a == "--min-p") s.min_p = std::strtof(val(), nullptr);
        else if (a == "--seed") s.seed = std::strtoull(val(), nullptr, 10);
        else if (a == "--tokens" || a == "--eos-ids" || a == "--follow") {
            // @FILE: the ids from a file, for a prompt longer than a command line holds (32k characters on Windows)
            const char* v = val();
            std::string text;
            if (v[0] == '@' && a != "--eos-ids") {
                std::ifstream f(v + 1, std::ios::binary);
                if (!f) {
                    std::fprintf(stderr, "strata-poly: %s: cannot read %s\n", a.c_str(), v + 1);
                    return 2;
                }
                text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
                while (!text.empty() && std::isspace((unsigned char) text.back())) text.pop_back();
                v = text.c_str();
            }
            if (!parse_ids(v, a == "--tokens" ? ids : a == "--eos-ids" ? eos : follow)) {
                std::fprintf(stderr, "strata-poly: %s takes token ids separated by commas\n", a.c_str());
                return 2;
            }
        } else {
            if (a != "--help" && a != "-h") std::fprintf(stderr, "strata-poly: unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (o.model.empty() || (!want_serve && ids.empty()) || max_new < 1) {
        usage();
        return 2;
    }
    if (kv_q4) {
        const std::string keys = kv_q4;
        if (keys != "q8" && keys != "q4") {
            std::fprintf(stderr, "strata-poly: --kv-q4-keys takes q8 or q4\n");
            return 2;
        }
        o.kv_q4 = keys == "q8" ? 8 : 4;
    }
    if (cpu_multi) {
        strata::poly::CpuKernelOptions k;
        k.multi = true;
        strata::poly::set_cpu_kernel_options(k);
        if (!strata::poly::kq::usable())
            std::fprintf(stderr, "strata-poly: --cpu-multi: this CPU has no AVX2, FMA or F16C; the CPU's experts go "
                                 "through ggml-cpu as without it\n");
    }
    // over --tail-skip, as Strata's engine reads it: =0 turns a model's default off
    if (const char* v = std::getenv("STRATA_ROUTE_TAIL_SKIP")) o.tail_skip = std::atoi(v);
    const long long most = 1LL << 24;   // MiB: 16 TiB
    if (cache.o.checkpoints < 0 || cache.o.every < 0 || cache.o.park_mib < 0 || cache.o.park_mib > most ||
        cache.o.park_slots < 0 || cache.o.park_slots > 1024 || cache.o.park_min < 0 || cache.min_free_mib < 0 ||
        cache.min_free_mib > most || cache.disk_mib < 0 || cache.disk_mib > most || cache.disk_min_free_mib < 0 ||
        cache.disk_min_free_mib > most) {
        std::fprintf(stderr, "strata-poly: a cache option is negative or too large\n");
        return 2;
    }
    if (suffix_mib < 0 || suffix_mib > most || (!corpus.empty() && suffix_mib == 0)) {
        std::fprintf(stderr, "strata-poly: --suffix-drafts takes the MiB of its index; --suffix-corpus needs it\n");
        return 2;
    }

    std::unique_ptr<Engine> loaded;
    std::unique_ptr<ConversationDisk> disk;   // after the session that names it is gone, its writes are finished
    try {
        const Clock::time_point t0 = Clock::now();
        loaded = Engine::open(o);
        Engine& engine = *loaded;
        std::fprintf(stderr, "strata-poly: loaded in %.1f s; %d of %d routed experts in VRAM (%.1f GiB), %d CPU threads\n",
                     ms_since(t0) / 1000.0, engine.cached_experts(), engine.total_experts(), engine.cached_gib(),
                     engine.threads());
        if (o.tail_skip > 0)
            std::fprintf(stderr, "strata-poly: tail skip %d: an expert outside VRAM that every token of a window ranks "
                                 "at place %d or later among its experts is left out; STRATA_ROUTE_TAIL_SKIP=0 turns "
                                 "it off\n",
                         o.tail_skip, o.tail_skip + 1);
        // Pages on another node than the threads are read at about two thirds of the speed.  The file cache puts a
        // page on the node of whatever read it first, so a file read back by something else lands on any.
        const bool far = engine.numa_share() > 0.0 && engine.numa_share() < 0.95;
        if (engine.numa_node() >= 0)
            std::fprintf(stderr, "strata-poly: the CPU threads stay on NUMA node %d, which has %.0f%% of the experts' "
                                 "pages%s\n", engine.numa_node(), 100.0 * engine.numa_share(),
                         far ? "; the rest is read from another node, more slowly" : "");
        else if (far)
            std::fprintf(stderr, "strata-poly: the experts' pages are on several NUMA nodes, %.0f%% on the fullest; "
                                 "the CPU reads the ones far from it more slowly\n", 100.0 * engine.numa_share());
        if (eos.empty())
            for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                    "tokenizer.ggml.eom_token_id"}) {
                const int64_t id = engine.meta_u(key, -1);
                if (id >= 0 && std::find(eos.begin(), eos.end(), (int32_t) id) == eos.end()) eos.push_back((int32_t) id);
            }
        std::unique_ptr<SuffixIndex> suffix;
        if (suffix_mib > 0) {
            suffix = std::make_unique<SuffixIndex>((size_t) suffix_mib << 20);
            long long lines = 0, tokens = 0;
            for (const std::string& path : corpus) {
                std::ifstream f(path);
                if (!f) throw std::runtime_error("cannot open --suffix-corpus " + path);
                std::vector<int32_t> seq;
                for (std::string line; std::getline(f, line);) {
                    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
                    if (line.empty()) continue;
                    if (!parse_ids(line.c_str(), seq)) throw std::runtime_error(path + ": a line is not token ids");
                    suffix->add(seq.data(), seq.size());
                    ++lines, tokens += (long long) seq.size();
                }
            }
            std::fprintf(stderr, "strata-poly: suffix drafts: an index of %zu tokens (%zu MiB)", suffix->capacity(),
                         suffix->bytes() >> 20);
            if (!corpus.empty()) std::fprintf(stderr, ", %lld sequences of %lld tokens from --suffix-corpus", lines, tokens);
            std::fprintf(stderr, "\n");
        }
        std::vector<strata::poly::StoredConversation> stored;
        const Clock::time_point disk_t0 = Clock::now();
        const long long disk_mib = cache.disk_mib > 0 ? cache.disk_mib : std::min<long long>(kDiskMib, cache.o.park_mib);
        if (!cache.disk.empty() && (cache.o.park_mib < 1 || cache.o.park_slots < 1)) {
            std::fprintf(stderr, "strata-poly: --conversation-disk keeps what the conversation cache holds, and it is "
                                 "off (--conversation-cache-mib or --conversation-cache-slots 0): nothing goes to disk\n");
        } else if (!cache.disk.empty()) {
            ConversationDisk::Settings d;
            d.dir = ConversationDisk::subfolder(std::filesystem::path(cache.disk), std::filesystem::path(o.model));
            d.key = disk_key(o, engine);
            d.limit = (uint64_t) disk_mib << 20;
            d.min_free = (uint64_t) cache.disk_min_free_mib << 20;
            d.log = stderr;
            disk = std::make_unique<ConversationDisk>();
            if (disk->open(d, stored)) cache.o.disk = disk.get();
        }
        Session ss(engine, eos, cache, lookup, suffix.get());
        if (cache.o.disk) {
            long long tokens = 0;
            for (const strata::poly::StoredConversation& c : stored) tokens += (long long) c.seq.size();
            const size_t found = stored.size(), taken = ss.kept.adopt(std::move(stored));
            std::fprintf(stderr, "strata-poly: conversation disk: %s: %zu of %zu conversations (%lld tokens) taken up, "
                                 "read when a prompt goes back to them; %llu MiB there, up to %lld; %.0f ms\n",
                         cache.disk.c_str(), taken, found, tokens, (unsigned long long) (disk->held_bytes() >> 20),
                         disk_mib, ms_since(disk_t0));
        }
        int rc = 0;
        if (want_serve) {
            rc = serve(ss);
        } else if ((long long) ids.size() + max_new > engine.max_context()) {
            throw std::runtime_error("the prompt and --max-new together exceed --max-context");
        } else {
            Pictures pictures;
            if (!images.empty()) {
                if (!engine.sees()) throw std::runtime_error("this model does not read pictures");
                if (image_pad < 0) throw std::runtime_error("--images needs --image-pad");
                const std::string wrong = take_pictures(images, image_pad, engine.n_embd(), ids, pictures);
                if (!wrong.empty()) throw std::runtime_error(wrong);
            }
            if (!follow.empty()) ss.follow = &follow;
            rc = once(ss, ids, pictures, image_pad, max_new, s, top_probs);
        }
        engine.save_profile();
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-poly: %s\n", e.what());
        if (want_serve) {
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
        }
        return 1;
    }
}
