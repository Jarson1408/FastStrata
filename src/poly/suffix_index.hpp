// src/poly/suffix_index.hpp - guesses from every sequence the engine has read or written since it started: the
// drafts of SuffixDecoding (Oliaro et al., "SuffixDecoding: Extreme Speculative Decoding for Emerging AI
// Applications", NeurIPS 2025, arXiv 2411.04975).
//
// An agent repeats itself.  It reads a file, quotes it in an edit, reads it again, makes the same kind of edit in
// another file, and its tool calls look alike from one turn to the next and from one conversation to the next.
// --lookup guesses from a repeat inside the conversation the engine holds, among the four latest places a trigram
// was seen.  This guesses from every prompt and every answer of every conversation, and from files given to it, and
// where the last tokens were seen in several places it guesses what most of them went on with.
//
// The text is a ring of tokens, sequences one after another with a mark between them; when the ring is full, the
// oldest text goes.  For three lengths K every place is chained to the place before it that ended with the same K
// tokens, as an LZ77 matcher chains them: a table has the newest place each hash of K tokens ended at, and each place
// the distance back to the one before.  A draft starts at the longest K whose last K tokens were seen before.  The
// places on its chain, kWalk at most and the newest first, are compared backwards token by token, up to kDepth
// tokens, and those that match furthest are kept: they are the occurrences of the deepest node of a suffix tree that
// the sequence ends in.  What followed them is then voted on a token at a time.  The token most of them have next is
// the draft's, its share of them is the chance it is right, and the places that had another token drop out.  The
// draft ends where the product of those shares falls under kMinSure, where the places run out, or after as many
// tokens as the match is long (the paper's limit with its factor 1).
//
// A conversation's prompt holds its earlier turns, which the index has seen.  So a prompt's tokens go in only from
// where it leaves the longest sequence it has seen that starts like it (kSeen are remembered), after a lead of the
// kDepth tokens before that point, which a match may run back into and which is not indexed again.
//
// Nothing here knows the model or the engine.  Memory: four bytes a token in the ring and a chain entry and a table
// slot for each length, 40 bytes a token in all, and the remembered sequences, four bytes a token.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::poly {

class SuffixIndex {
public:
    static constexpr int kLevels = 3;
    static constexpr int kLen[kLevels] = {24, 8, 3};   ///< the K of each chain, longest first; 3 is the shortest match
    static constexpr int kDepth = 64;                  ///< a match is compared this far back at most
    static constexpr int kWalk = 64;                   ///< places looked at on a chain
    static constexpr int kMaxDraft = 16;
    static constexpr double kMinSure = 0.3;            ///< a draft ends where the chance it is right so far falls under this
    static constexpr int kSeen = 16;                   ///< sequences remembered, to index only what is new in a prompt
    static constexpr size_t kBytesPerToken = 4 + kLevels * (4 + 8);

    /// An index in about `bytes` (rounded down to a power of two of tokens, 4096 at least).
    explicit SuffixIndex(size_t bytes) {
        size_t cap = 4096;
        while (cap * 2 * kBytesPerToken <= bytes) cap *= 2;
        tok_.assign(cap, kMark);
        mask_ = cap - 1;
        for (Level& l : lv_) {
            l.back.assign(cap, 0);
            l.head.assign(cap, 0);
        }
    }

    size_t capacity() const { return tok_.size(); }   ///< tokens the ring holds
    size_t bytes() const { return capacity() * kBytesPerToken; }
    size_t seen_bytes() const {
        size_t n = 0;
        for (const Seen& s : seen_) n += s.seq.capacity() * sizeof(int32_t);
        return n;
    }
    uint64_t written() const { return end_; }          ///< tokens and marks written since the start
    size_t held() const { return (size_t) std::min<uint64_t>(end_, capacity()); }

    /// A sequence of a corpus (--suffix-corpus): indexed whole, on its own.
    void add(const int32_t* t, size_t n) {
        mark();
        tail_ = -1;
        from_ = end_;
        for (size_t i = 0; i < n; ++i) put(t[i]);
    }

    /// A request's sequence, before its answer: what is new in it is indexed, and the answer goes on from its end.
    void begin(const std::vector<int32_t>& seq) {
        int best = -1;
        size_t same = 0;
        for (int i = 0; i < (int) seen_.size(); ++i) {
            const size_t c = common(seen_[(size_t) i].seq, seq);
            if (c > same || (c == same && c > 0 && i == tail_)) best = i, same = c;
        }
        ++clock_;
        const bool goes_on = best >= 0 && same == seen_[(size_t) best].seq.size();
        if (!goes_on || best != tail_) {
            // a sequence of its own in the ring, after the tokens before `same` that a match may run back into
            mark();
            const size_t lead = same > (size_t) kDepth ? same - kDepth : 0;
            for (size_t i = lead; i < same; ++i) put_quiet(seq[i]);
        }
        if (goes_on) {
            cur_ = best;
            seen_[(size_t) cur_].seq.insert(seen_[(size_t) cur_].seq.end(), seq.begin() + (std::ptrdiff_t) same, seq.end());
        } else {
            cur_ = slot();
            seen_[(size_t) cur_].seq = seq;
        }
        seen_[(size_t) cur_].stamp = clock_;
        tail_ = cur_;
        from_ = end_;
        for (size_t i = same; i < seq.size(); ++i) put(seq[i]);
    }

    /// A token of the answer.
    void append(int32_t t) {
        put(t);
        if (cur_ >= 0) seen_[(size_t) cur_].seq.push_back(t);
    }

    /// Writes up to max_k tokens that may follow the sequence so far to `out`; returns how many.  sure()[i] is then
    /// the chance the first i+1 of them are right, as the places they come from say.
    int propose(int max_k, int32_t* out) {
        match_ = 0;
        max_k = std::min(max_k, kMaxDraft);
        if (max_k < 1 || run_ < kLen[kLevels - 1]) return 0;
        const uint64_t at = end_ - 1, low = first();
        int n = 0;
        for (int li = 0; li < kLevels && n == 0; ++li) {
            const int k = kLen[li];
            if (run_ < (uint64_t) k) continue;
            const Level& l = lv_[(size_t) li];
            uint64_t p = l.head[hash(at, k) & mask_];
            for (int walked = 0; p != 0 && walked < kWalk; ++walked) {
                const uint64_t q = p - 1;
                if (q < low) break;   // older places are gone too
                if (q != at) {
                    const int m = same_back(q, low);
                    if (m >= k && m > 0) {
                        if (m > match_) match_ = m, n = 0;
                        if (m == match_) place_[n++] = q;
                    }
                }
                const uint32_t d = l.back[q & mask_];
                p = d == 0 || d > q ? 0 : q - d + 1;
            }
        }
        if (n == 0) {
            match_ = 0;
            return 0;
        }
        max_k = std::min(max_k, match_);
        double sure = 1.0;
        int k = 0;
        for (; k < max_k; ++k) {
            // what each place that goes on has next; ended places say nothing
            int m = 0;
            for (int i = 0; i < n; ++i) {
                const uint64_t nx = place_[i] + 1 + (uint64_t) k;
                if (nx >= end_) continue;
                const int32_t t = tok_[nx & mask_];
                if (t == kMark) continue;
                place_[m] = place_[i];
                next_[m++] = t;
            }
            if (m == 0) break;
            // the most frequent; of equals, the one the newest place has (places are newest first)
            int best = 0, best_n = 0;
            for (int i = 0; i < m; ++i) {
                int c = 0;
                for (int j = 0; j < m; ++j) c += next_[j] == next_[i];
                if (c > best_n) best = i, best_n = c;
            }
            sure *= (double) best_n / (double) m;
            if (sure < kMinSure) break;
            out[k] = next_[best];
            sure_[k] = sure;
            n = 0;
            for (int i = 0; i < m; ++i)
                if (next_[i] == out[k]) place_[n++] = place_[i];
        }
        if (k == 0) match_ = 0;
        return k;
    }

    int last_match() const { return match_; }   ///< the tokens the last draft's places matched; 0: no draft
    const double* sure() const { return sure_; }

private:
    static constexpr int32_t kMark = -1;   // between sequences; a token is never below zero (pictures are padded)

    struct Level {
        std::vector<uint64_t> head;   // by hash: the newest place + 1 that ended with it; 0: none
        std::vector<uint32_t> back;   // by place in the ring: the distance to the place before it of the same hash
    };
    struct Seen {
        std::vector<int32_t> seq;
        uint64_t stamp = 0;
    };

    uint64_t first() const { return end_ > capacity() ? end_ - capacity() : 0; }

    static size_t common(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        const size_t n = std::min(a.size(), b.size());
        size_t i = 0;
        while (i < n && a[i] == b[i]) ++i;
        return i;
    }

    // the hash of the k tokens that end at `at`
    uint64_t hash(uint64_t at, int k) const {
        uint64_t h = 0x9E3779B97F4A7C15ull * (uint64_t) k;
        for (int i = k - 1; i >= 0; --i) h = (h ^ (uint32_t) tok_[(at - (uint64_t) i) & mask_]) * 0x100000001B3ull;
        h ^= h >> 32;
        h *= 0xD6E8FEB86659FD93ull;
        return h ^ (h >> 32);
    }

    // how many tokens back from `q` are those back from the sequence's end, kDepth at most
    int same_back(uint64_t q, uint64_t low) const {
        const uint64_t at = end_ - 1;
        const int most = (int) std::min<uint64_t>(std::min<uint64_t>(run_, kDepth), q - low + 1);
        int m = 0;
        while (m < most && tok_[(q - (uint64_t) m) & mask_] == tok_[(at - (uint64_t) m) & mask_]) ++m;
        return m;
    }

    void mark() {
        if (end_ > 0 && tok_[(end_ - 1) & mask_] == kMark) return;
        tok_[end_ & mask_] = kMark;
        ++end_;
        run_ = 0;
    }

    // a token that a match may run into but that is not indexed again
    void put_quiet(int32_t t) {
        tok_[end_ & mask_] = t;
        ++end_;
        ++run_;
    }

    void put(int32_t t) {
        const uint64_t at = end_;
        put_quiet(t);
        for (int li = 0; li < kLevels; ++li) {
            const int k = kLen[li];
            Level& l = lv_[(size_t) li];
            const size_t i = (size_t) (at & mask_);
            if (run_ < (uint64_t) k || at < from_) {
                l.back[i] = 0;
                continue;
            }
            uint64_t& h = l.head[hash(at, k) & mask_];
            l.back[i] = h != 0 && at - (h - 1) <= UINT32_MAX ? (uint32_t) (at - (h - 1)) : 0;
            h = at + 1;
        }
    }

    // a slot for a new remembered sequence: an empty one, or the one used longest ago
    int slot() {
        if (seen_.size() < (size_t) kSeen) {
            seen_.emplace_back();
            return (int) seen_.size() - 1;
        }
        int out = 0;
        for (int i = 1; i < (int) seen_.size(); ++i)
            if (seen_[(size_t) i].stamp < seen_[(size_t) out].stamp) out = i;
        seen_[(size_t) out] = Seen{};
        if (tail_ == out) tail_ = -1;
        return out;
    }

    std::vector<int32_t> tok_;
    size_t mask_ = 0;
    Level lv_[kLevels];
    uint64_t end_ = 0;    // tokens and marks written
    uint64_t run_ = 0;    // tokens since the last mark
    uint64_t from_ = 0;   // the first place of the sequence being written that is indexed
    std::vector<Seen> seen_;
    int cur_ = -1;        // the remembered sequence the answer goes on
    int tail_ = -1;       // the remembered sequence the ring ends with; -1: a corpus file's, or none
    uint64_t clock_ = 0;
    int match_ = 0;
    double sure_[kMaxDraft] = {};
    uint64_t place_[kWalk] = {};
    int32_t next_[kWalk] = {};
};

}  // namespace strata::poly
