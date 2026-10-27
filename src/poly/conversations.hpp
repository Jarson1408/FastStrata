// src/poly/conversations.hpp - the conversations a server has read, kept so that a prompt which starts like one of
// them is not read again from its first token.
//
// The engine holds one sequence at a time.  Coming back to an earlier point of a sequence takes two things: the
// recurrent state at that point, which is a checkpoint, and the rows its tokens left in the attention caches.  The
// rows of the sequence in the engine are there already.  A conversation that makes room for another is parked:
// its rows are copied to host memory, up to its last checkpoint.  What is saved is written once and never changed,
// so conversations that start alike share it, and a parked conversation comes back bit for bit.  With a disk
// (--conversation-disk, conversation_disk.hpp) all of it is copied there too, and the next engine takes it up.
//
// Nothing here knows the model.  E is the engine:
//     void reset();
//     size_t state_bytes() const;  void save_state(uint8_t*);  void load_state(const uint8_t*, int64_t pos);
//     size_t row_bytes() const;    void save_rows(int64_t first, int64_t n, uint8_t*);
//     void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n);
#pragma once

#include "conversation_disk.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::poly {

/// Where a prompt's last message begins (Conversations::message_at()): the token that opens the turn at turn_at, the
/// time before that, when it is at most `most` tokens back and past `from`, where the prompt is taken up; -1: none.
/// The other engine's message_checkpoint_boundary, with the token that opens the reply's turn as the turn token.
inline int64_t last_message(const std::vector<int32_t>& ids, int64_t from, int64_t turn_at, int64_t most) {
    if (most < 1 || turn_at <= from || turn_at >= (int64_t) ids.size()) return -1;
    for (int64_t i = turn_at - 1; i > from && turn_at - i <= most; --i)
        if (ids[(size_t) i] == ids[(size_t) turn_at]) return i;
    return -1;
}

struct CacheOptions {
    int checkpoints = 6;       ///< kept per conversation; 0: a prompt is reread unless it goes on from the last token
    int64_t every = 2048;      ///< a checkpoint every so many prompt tokens, besides the one at a prompt's end, the
                               ///< one where its last turn begins (turn_at()) and the one where its last message
                               ///< begins (message_at()); 0: none
    int64_t park_mib = 0;      ///< host memory for parked conversations; 0: none is parked
    int park_slots = 4;        ///< parked conversations at most
    int64_t park_min = 1024;   ///< a conversation with fewer tokens before its last checkpoint is not parked
    std::function<bool(size_t)> admit;   ///< whether the host can spare this many bytes; unset: it can
    FILE* log = nullptr;       ///< where parking is reported; the server reads these lines
    ConversationDisk* disk = nullptr;    ///< where what is parked is kept for the next engine too; null: nowhere
};

template<class E>
class Conversations {
public:
    Conversations(E& engine, CacheOptions o)
        : e_(engine), o_(std::move(o)), slots_((size_t) std::max(o_.park_slots, 0) + 1) {}

    /// The tokens the engine has read.
    const std::vector<int32_t>& seq() const { return slots_[cur_].seq; }

    /// Leaves the engine at the furthest point of `ids` that can be come back to, and returns that position.
    int64_t resume(const std::vector<int32_t>& ids) {
        const int64_t n = (int64_t) ids.size();
        turn_ = message_ = -1;
        size_t best = cur_;
        int64_t at = 0, same = 0;
        for (bool found = false; !found;) {
            best = cur_, at = 0, same = 0;
            for (size_t i = 0; i < slots_.size(); ++i) {
                const int64_t c = common(slots_[i].seq, ids), r = reach(i, c, n);
                if (r > at || (r == at && i == cur_)) best = i, at = r, same = c;
            }
            // what coming back there reads is in RAM; a conversation from the disk whose files cannot be read is
            // dropped, and the next best is taken
            found = at == 0 || ready(best, at);
        }
        ++clock_;
        if (at == 0) {
            // nothing kept starts like this
            if (!cur().seq.empty() && park(slots_.size())) cur_ = take(cur_);
            slots_[cur_] = Slot{};
            e_.reset();
        } else if (best != cur_ || at < len(cur())) {
            if (best != cur_) enter(best, at);
            // The conversation is left at `at`.  If the prompt the client sent for it went on from there, the
            // client may send that again, so the conversation is parked and this one goes on in a copy.
            if (same >= cur().prompt || !branch(at)) cut(cur(), at);
            e_.load_state(state_at(cur(), at), at);
        }
        cur().stamp = clock_;
        return at;
    }

    /// One more checkpoint of the prompt resume() took up: where its chat's last turn begins, which the server
    /// knows from the template.  The chat's next request renders the conversation up to there as this one does,
    /// but this turn's header and reply not always in the same tokens (a template drops the reply's thinking), so
    /// the checkpoint before the prompt's last token is seldom where it goes on from.  A place the read does not
    /// pass adds none.
    void turn_at(int64_t at) { turn_ = at; }

    /// And one where the prompt's last message begins, the turn before that one (--prompt-cache-message).  A client
    /// or a gateway that ends every prompt with a message of its own and leaves it out of the next one (a reminder,
    /// a project-memory frame) sends the next prompt alike only up to there: the checkpoints at the turn and before
    /// the last token are past it.  It is kept, as the prompt's last two are, until the next prompt.
    void message_at(int64_t at) { message_ = at; }

    /// Of `want` tokens from position `at`, how many to read before the next checkpoint is due.
    int64_t until_checkpoint(int64_t at, int64_t want) const {
        if (o_.checkpoints < 1) return want;
        if (o_.every > 0) want = std::min(want, (at / o_.every + 1) * o_.every - at);
        if (message_ > at) want = std::min(want, message_ - at);
        return turn_ > at ? std::min(want, turn_ - at) : want;
    }

    /// Before the engine reads the token at `at`; `last` says it is the prompt's last.  Takes a checkpoint if one is
    /// due.  One sits before the last token of every prompt, so coming back to it leaves a token to read and the
    /// logits that come with it.
    void before(int64_t at, bool last) {
        if (o_.checkpoints < 1 || at < 1 ||
            (!last && at != turn_ && at != message_ && (o_.every < 1 || at % o_.every != 0)))
            return;
        if (const BlobPtr b = mark(at)) e_.save_state(b->p.get());
    }

    /// For an engine that reads past checkpoints in one go, of n tokens from `at`: the checkpoints due after the
    /// read's first token.  `out` gets, for each, how many of the read's tokens come before it and where its
    /// state_bytes() go; null where no state is wanted, and the engine still breaks its work there as if one
    /// were.  The engine writes a state when it has it.  The checkpoints are the conversation's once read() has
    /// the read's tokens, and not at all when the read is given up (rewind()).
    ///
    /// A state is wanted for the checkpoints the conversation would still have, had each been taken in turn as
    /// before() takes one.  A long prompt has more due than are kept, and a state can be hundreds of MiB.
    void inside(int64_t at, int64_t n, std::vector<std::pair<int64_t, uint8_t*>>& out) {
        out.clear();
        due_.clear();
        after_.clear();
        if (o_.checkpoints < 1) return;
        std::vector<int64_t> due;
        if (o_.every > 0)
            for (int64_t p = (at / o_.every + 1) * o_.every; p < at + n; p += o_.every) due.push_back(p);
        if (message_ > at && message_ < at + n && (o_.every < 1 || message_ % o_.every != 0))
            due.insert(std::upper_bound(due.begin(), due.end(), message_), message_);
        if (turn_ > at && turn_ < at + n && (o_.every < 1 || turn_ % o_.every != 0))
            due.insert(std::upper_bound(due.begin(), due.end(), turn_), turn_);
        if (due.empty()) return;
        for (const Checkpoint& m : cur().marks) after_.push_back(m.pos);
        for (const int64_t p : due) {
            after_.push_back(p);
            while (after_.size() > (size_t) o_.checkpoints) after_.erase(after_.begin() + (std::ptrdiff_t) drop_of(after_));
        }
        for (const int64_t p : due) {
            const bool stays = std::find(after_.begin(), after_.end(), p) != after_.end();
            due_.push_back({p, stays ? blob(e_.state_bytes()) : nullptr});
            out.emplace_back(p - at, due_.back().state ? due_.back().state->p.get() : nullptr);
        }
    }

    /// The engine has read these tokens.  A read that inside() was asked about is done with them, and its
    /// checkpoints take their places.
    void read(const int32_t* t, int64_t n) {
        Slot& s = cur();
        s.seq.insert(s.seq.end(), t, t + n);
        if (due_.empty()) return;
        const auto stays = [&](const Checkpoint& m) {
            return m.state && std::find(after_.begin(), after_.end(), m.pos) != after_.end();
        };
        std::vector<Checkpoint> marks;
        for (const Checkpoint& m : s.marks)
            if (stays(m)) marks.push_back(m);
        for (const Checkpoint& m : due_)
            if (stays(m)) marks.push_back(m);
        s.marks = std::move(marks);
        due_.clear();
    }

    /// A read was given up part way: what it left in the engine is not the conversation's, and the checkpoints
    /// inside it are not taken.  The conversation goes back to its last checkpoint, which the engine is left
    /// at, or is forgotten when it has none.  Returns where the conversation is then.
    int64_t rewind() {
        due_.clear();
        Slot& s = cur();
        if (s.marks.empty()) {
            slots_[cur_] = Slot{};
            e_.reset();
            return 0;
        }
        const int64_t at = s.marks.back().pos;
        if (!load(*s.marks.back().state)) {
            slots_[cur_] = Slot{};
            e_.reset();
            return 0;
        }
        cut(s, at);
        e_.load_state(s.marks.back().state->p.get(), at);
        return at;
    }
    /// The first n tokens of the conversation are the prompt the client sent.
    void prompt_was(int64_t n) { cur().prompt = std::min(n, len(cur())); }
    /// After an engine error: the conversation being read is lost.  The caller resets the engine.
    void drop() {
        slots_[cur_] = Slot{};
        due_.clear();
    }

    /// With a disk, after an answer: the rows the conversation in the engine has only there, up to its last
    /// checkpoint, are copied to host memory as parking would copy them, so that store() can put it on disk before
    /// it is parked; parking it then copies only what it read after that.  They count in the budget as soon as
    /// they are held, and the conversation is not parked for it.
    void hold() {
        Slot& s = cur();
        if (!o_.disk || o_.park_mib < 1 || o_.park_slots < 1 || s.marks.empty() || s.marks.back().pos < o_.park_min)
            return;
        const int64_t end = s.marks.back().pos, from = saved(s);
        if (from >= end) return;
        const Clock::time_point t0 = Clock::now();
        const size_t bytes = (size_t) (end - from) * e_.row_bytes();
        const BlobPtr b = fit(bytes, slots_.size(), false) ? blob(bytes) : nullptr;
        if (b) {
            e_.save_rows(from, end - from, b->p.get());
            s.rows.push_back({b, from, end - from, end - from});
        }
        if (o_.log)
            std::fprintf(o_.log, "strata-poly: conversation disk: %s %lld rows (%.1f MiB) of the conversation in the "
                                 "engine in %.1f ms\n", b ? "held" : "no room to hold", (long long) (end - from),
                         (double) bytes / (1 << 20), std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }

    /// With a disk: what the cache holds goes there (ConversationDisk::commit()), the latest used first, each
    /// conversation up to the last checkpoint its saved rows reach.  Returns how many are there then.
    size_t store() {
        if (!o_.disk) return 0;
        std::vector<size_t> order;
        for (size_t i = 0; i < slots_.size(); ++i)
            if (!slots_[i].seq.empty()) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return slots_[a].stamp > slots_[b].stamp; });
        std::vector<ConversationDisk::Want> want;
        for (const size_t i : order) {
            Slot& s = slots_[i];
            const int64_t end = saved(s);
            ConversationDisk::Want w;
            for (const Checkpoint& m : s.marks)
                if (m.pos <= end) {
                    w.c.marks.push_back({m.pos, file_of(*m.state), m.state->bytes});
                    w.blobs.push_back(source(m.state));
                }
            if (w.c.marks.empty()) continue;
            const int64_t last = w.c.marks.back().pos;
            if (!s.disk) s.disk = o_.disk->new_id();
            w.c.id = s.disk;
            w.c.stamp = s.stamp;
            w.c.prompt = std::min(s.prompt, last);
            w.c.seq.assign(s.seq.begin(), s.seq.begin() + (std::ptrdiff_t) last);
            for (const Rows& r : s.rows) {
                if (r.first >= last) break;
                w.c.rows.push_back({file_of(*r.blob), r.blob->bytes, r.first, r.count, std::min(r.keep, last - r.first)});
                w.blobs.push_back(source(r.blob));
            }
            want.push_back(std::move(w));
        }
        return o_.disk->commit(std::move(want));
    }

    /// With a disk: the conversations an engine before this one left there come back as parked ones, the latest
    /// used first while there are slots and budget for them; store() then deletes the files of the others.  What
    /// they hold is read when a prompt first goes back to them (ready()).  Returns how many came back.
    size_t adopt(std::vector<StoredConversation> list) {
        if (!o_.disk) return 0;
        std::sort(list.begin(), list.end(), [](const StoredConversation& a, const StoredConversation& b) {
            return a.stamp > b.stamp;
        });
        std::unordered_map<uint64_t, BlobPtr> files;
        const auto on_disk = [&](uint64_t file, uint64_t bytes) {
            BlobPtr& b = files[file];
            if (!b) {
                b = std::make_shared<Blob>();
                b->bytes = (size_t) bytes;
                b->file = file;
            }
            return b;
        };
        const size_t budget = (size_t) o_.park_mib << 20;
        std::vector<size_t> taken;
        for (StoredConversation& c : list) {
            const size_t at = (cur_ + 1 + taken.size()) % slots_.size();
            if (taken.size() >= (size_t) o_.park_slots || at == cur_) break;
            bool fits = true;
            for (const StoredConversation::Mark& m : c.marks) fits = fits && m.bytes == e_.state_bytes();
            for (const StoredConversation::Rows& r : c.rows) fits = fits && r.bytes == (uint64_t) r.count * e_.row_bytes();
            if (!fits) continue;
            Slot s;
            s.seq = std::move(c.seq);
            s.prompt = c.prompt;
            s.disk = c.id;
            for (const StoredConversation::Mark& m : c.marks) s.marks.push_back({m.pos, on_disk(m.file, m.bytes)});
            for (const StoredConversation::Rows& r : c.rows)
                s.rows.push_back({on_disk(r.file, r.bytes), r.first, r.count, r.keep});
            slots_[at] = std::move(s);
            if (held(slots_.size()).bytes > budget) {
                slots_[at] = Slot{};
                continue;
            }
            taken.push_back(at);
        }
        // the order they were used in, the latest last, before anything this engine does
        for (size_t j = 0; j < taken.size(); ++j) slots_[taken[j]].stamp = taken.size() - j;
        clock_ = taken.size();
        store();
        return taken.size();
    }

    /// The parked conversations, the bytes they hold, and how many were pushed out so far.
    size_t parked() const { return held(slots_.size()).count; }
    size_t parked_bytes() const { return held(slots_.size()).bytes; }
    size_t evictions() const { return evictions_; }

private:
    using Clock = std::chrono::steady_clock;

    struct Blob {
        std::unique_ptr<uint8_t[]> p;   // null: the blob is on disk only, until load() reads it
        size_t bytes = 0;
        uint64_t file = 0;              // its file on disk; 0: none
    };
    using BlobPtr = std::shared_ptr<Blob>;
    struct Checkpoint {
        int64_t pos;
        BlobPtr state;
    };
    // The blob holds `count` rows from row `first`; the first `keep` of them are this conversation's.
    struct Rows {
        BlobPtr blob;
        int64_t first, count, keep;
    };
    struct Slot {
        std::vector<int32_t> seq;
        int64_t prompt = 0;             // seq's first tokens that the client sent
        std::vector<Checkpoint> marks;  // by position
        std::vector<Rows> rows;         // back to back from row 0; a parked conversation has all of its rows
        uint64_t stamp = 0;
        uint64_t disk = 0;              // its file on disk; 0: none
    };
    struct Held {
        size_t count = 0, bytes = 0;
    };

    Slot& cur() { return slots_[cur_]; }
    const Slot& cur() const { return slots_[cur_]; }
    static int64_t len(const Slot& s) { return (int64_t) s.seq.size(); }
    static int64_t saved(const Slot& s) { return s.rows.empty() ? 0 : s.rows.back().first + s.rows.back().keep; }

    static int64_t common(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        return (int64_t) n;
    }

    // How far into a prompt of n tokens slot i gets it, given that they start with `same` tokens alike.
    int64_t reach(size_t i, int64_t same, int64_t n) const {
        const Slot& s = slots_[i];
        if (i == cur_ && same > 0 && same == len(s) && n > same) return same;   // the engine is right there
        int64_t at = 0;
        for (const Checkpoint& m : s.marks)
            if (m.pos <= std::min(same, n - 1)) at = m.pos;
        return at;
    }

    static const uint8_t* state_at(const Slot& s, int64_t at) {
        for (const Checkpoint& m : s.marks)
            if (m.pos == at) return m.state->p.get();
        return nullptr;   // resume() only goes to positions reach() found
    }

    static BlobPtr blob(size_t bytes) {
        auto b = std::make_shared<Blob>();
        b->p.reset(new (std::nothrow) uint8_t[bytes]);
        b->bytes = bytes;
        return b->p ? b : nullptr;
    }

    // A checkpoint of the current conversation at `at`, its state still to be written.  Null: there is one there,
    // or no memory for it.
    BlobPtr mark(int64_t at) {
        Slot& s = cur();
        if (!s.marks.empty() && s.marks.back().pos == at) return nullptr;
        const BlobPtr b = blob(e_.state_bytes());
        if (!b) return nullptr;
        s.marks.push_back({at, b});
        while (s.marks.size() > (size_t) o_.checkpoints) {
            std::vector<int64_t> pos;
            for (const Checkpoint& m : s.marks) pos.push_back(m.pos);
            s.marks.erase(s.marks.begin() + (std::ptrdiff_t) drop_of(pos));
        }
        return b;
    }

    // Of more checkpoints than are kept, the one that goes: the one whose neighbours are closest together.  The
    // last two stay, the end of this prompt and of the one before being where a client most often comes back to,
    // and so does the one where this prompt's last message begins (message_at()) while there are others to go.
    size_t drop_of(const std::vector<int64_t>& pos) const {
        size_t drop = 0;
        int64_t gap = INT64_MAX;
        for (size_t k = 0; k + 2 < pos.size(); ++k) {
            const int64_t g = pos[k + 1] - (k ? pos[k - 1] : 0);
            if (g < gap && pos[k] != message_) gap = g, drop = k;
        }
        return drop;
    }

    static void cut(Slot& s, int64_t at) {
        if (at >= len(s)) return;
        s.seq.resize((size_t) at);
        s.prompt = std::min(s.prompt, at);
        while (!s.marks.empty() && s.marks.back().pos > at) s.marks.pop_back();
        while (!s.rows.empty() && s.rows.back().first >= at) s.rows.pop_back();
        if (!s.rows.empty()) s.rows.back().keep = std::min(s.rows.back().keep, at - s.rows.back().first);
    }

    // The parked conversations and what they hold; a blob several of them share counts once.  The current one
    // counts when `with` is its slot.
    Held held(size_t with) const {
        Held h;
        std::vector<const Blob*> seen;
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot& s = slots_[i];
            if (s.seq.empty() || (i == cur_ && i != with)) continue;
            ++h.count;
            for (const Checkpoint& m : s.marks) seen.push_back(m.state.get());
            for (const Rows& r : s.rows) seen.push_back(r.blob.get());
        }
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (const Blob* b : seen) h.bytes += b->bytes;
        return h;
    }

    // Makes the parked conversations, the current one among them and `extra` bytes more, fit the budget and the
    // slots.  The least recently used goes first, whatever its length: a long conversation can lose its place to
    // several short ones.  Slot `keep` stays; it is the one about to take the engine, if any, so it frees a slot.
    // `parks` false: the current one stays in the engine (hold()) and takes no slot.
    // False: it does not fit, or the host cannot spare the bytes.
    bool fit(size_t extra, size_t keep, bool parks = true) {
        const size_t budget = (size_t) o_.park_mib << 20;
        for (;;) {
            const Held h = held(cur_);
            const size_t leaving = keep < slots_.size() && keep != cur_ && !slots_[keep].seq.empty() ? 1 : 0;
            const size_t staying = !parks && !cur().seq.empty() ? 1 : 0;
            if (h.bytes + extra <= budget && h.count - leaving - staying <= (size_t) o_.park_slots) break;
            size_t out = slots_.size();
            for (size_t i = 0; i < slots_.size(); ++i)
                if (i != cur_ && i != keep && !slots_[i].seq.empty() &&
                    (out == slots_.size() || slots_[i].stamp < slots_[out].stamp))
                    out = i;
            if (out == slots_.size()) return false;
            slots_[out] = Slot{};
            ++evictions_;
        }
        return extra == 0 || !o_.admit || o_.admit(extra);
    }

    // Copies the rows the current conversation still has only in the engine, up to its last checkpoint; what it
    // read after that cannot be come back to and is dropped.  False: it cannot be kept.
    bool park(size_t keep) {
        Slot& s = cur();
        if (o_.park_mib < 1 || o_.park_slots < 1 || s.marks.empty() || s.marks.back().pos < o_.park_min) return false;
        const Clock::time_point t0 = Clock::now();
        const int64_t end = s.marks.back().pos, from = saved(s);
        cut(s, end);
        const size_t bytes = (size_t) (end - from) * e_.row_bytes();
        const BlobPtr b = from < end && fit(bytes, keep) ? blob(bytes) : nullptr;
        if (from < end && !b) {
            say("skipped", end, t0, slots_.size());
            return false;
        }
        if (b) {
            e_.save_rows(from, end - from, b->p.get());
            s.rows.push_back({b, from, end - from, end - from});
        }
        if (!fit(0, keep)) return false;
        say("parked", end, t0, cur_);
        return true;
    }

    // An empty slot other than `keep`.
    size_t take(size_t keep) {
        size_t out = slots_.size();
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (i == keep) continue;
            if (slots_[i].seq.empty()) return i;
            if (out == slots_.size() || slots_[i].stamp < slots_[out].stamp) out = i;
        }
        slots_[out] = Slot{};   // park() left one empty, so this is not reached
        ++evictions_;
        return out;
    }

    // Slot i takes the engine, with its rows before `need` back in the caches.  The conversation that leaves is
    // parked if it can be.  Rows that both have from the same blob are in the caches already.
    void enter(size_t i, int64_t need) {
        const Clock::time_point t0 = Clock::now();
        const Slot& in = slots_[i];
        int64_t have = 0;
        for (size_t k = 0; k < in.rows.size() && k < cur().rows.size() && in.rows[k].blob == cur().rows[k].blob; ++k) {
            have = in.rows[k].first + std::min(in.rows[k].keep, cur().rows[k].keep);
            if (in.rows[k].keep != cur().rows[k].keep) break;
        }
        if (cur().seq.empty() || !park(i)) slots_[cur_] = Slot{};
        for (const Rows& r : in.rows) {
            const int64_t a = std::max(r.first, have), b = std::min(r.first + r.keep, need);
            if (a < b) e_.load_rows(r.blob->p.get(), r.first, r.count, a, b - a);
        }
        cur_ = i;
        say("restored", need, t0, slots_.size());
    }

    // Parks the current conversation whole and goes on at `at` in a copy of its start, which shares what is
    // saved.  False: it cannot be parked.
    bool branch(int64_t at) {
        const size_t from = cur_;
        if (!park(slots_.size())) return false;
        const size_t to = take(from);
        const Slot& a = slots_[from];
        Slot b;
        b.seq.assign(a.seq.begin(), a.seq.begin() + (std::ptrdiff_t) at);
        b.prompt = std::min(a.prompt, at);
        for (const Checkpoint& m : a.marks)
            if (m.pos <= at) b.marks.push_back(m);
        for (const Rows& r : a.rows)
            if (r.first < at) b.rows.push_back({r.blob, r.first, r.count, std::min(r.keep, at - r.first)});
        slots_[to] = std::move(b);
        cur_ = to;
        return true;
    }

    // Slot i's blobs that coming back to `at` reads: its state there, and its rows before it unless it is in the
    // engine.  Those that are on disk only are read now.  False: one could not be read, and the slot is dropped.
    bool ready(size_t i, int64_t at) {
        std::vector<Blob*> need;
        for (const Checkpoint& m : slots_[i].marks)
            if (m.pos == at && !m.state->p) need.push_back(m.state.get());
        if (i != cur_)
            for (const Rows& r : slots_[i].rows)
                if (r.first < at && !r.blob->p) need.push_back(r.blob.get());
        if (need.empty()) return true;
        const Clock::time_point t0 = Clock::now();
        size_t bytes = 0;
        bool ok = true;
        for (Blob* b : need) {
            ok = ok && load(*b);
            bytes += b->bytes;
        }
        if (o_.log)
            std::fprintf(o_.log, "strata-poly: conversation disk: %s %.1f MiB of a conversation of %lld tokens in "
                                 "%.1f ms\n", ok ? "read" : "could not read (dropped it)", (double) bytes / (1 << 20),
                         (long long) at, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        if (!ok) slots_[i] = Slot{};
        return ok;
    }

    // A blob in RAM, read from its file when it is on disk only.  False: it could not be.
    bool load(Blob& b) {
        if (b.p) return true;
        if (!o_.disk || !b.file) return false;
        b.p.reset(new (std::nothrow) uint8_t[b.bytes]);
        if (b.p && o_.disk->read_blob(b.file, b.bytes, b.p.get())) return true;
        b.p.reset();
        return false;
    }

    // A blob's file on disk, named now if it has none.
    uint64_t file_of(Blob& b) {
        if (!b.file) b.file = o_.disk->new_id();
        return b.file;
    }
    static ConversationDisk::Source source(const BlobPtr& b) {
        return {b->file, b->bytes, std::shared_ptr<const void>(b, b->p.get()), b->p.get()};
    }

    // `with` is the current slot when its conversation is the one just parked.
    void say(const char* what, int64_t tokens, Clock::time_point t0, size_t with) const {
        if (!o_.log) return;
        const Held h = held(with);
        std::fprintf(o_.log,
                     "strata-poly: conversation cache: %s %lld tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu\n",
                     what, (long long) tokens, std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                     h.count, h.bytes, evictions_);
    }

    E& e_;
    CacheOptions o_;
    std::vector<Slot> slots_;   // the current conversation and the parked ones
    size_t cur_ = 0;
    uint64_t clock_ = 0;
    size_t evictions_ = 0;
    std::vector<Checkpoint> due_;   // inside()'s checkpoints, until read() makes them the conversation's
    std::vector<int64_t> after_;    // where the conversation's checkpoints are once that read is done
    int64_t turn_ = -1;             // turn_at()'s place in the prompt being read; -1: none
    int64_t message_ = -1;          // message_at()'s; -1: none
};

}  // namespace strata::poly
