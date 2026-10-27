// src/poly/conversation_disk.cpp - see conversation_disk.hpp.
//
// A file is a header (a magic word, the format, a hash of the key, the length of what follows), its bytes, and a
// checksum of them seeded with the key's hash and the file's id, so a file is good only under its own name and key.
#include "conversation_disk.hpp"

#include "strata/core/conversation_file.hpp"   // SessionHasher

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <system_error>
#include <unordered_set>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cwctype>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace strata::poly {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kBlobMagic = 0x4C425053;   // "SPBL"
constexpr uint32_t kConvMagic = 0x56435053;   // "SPCV"
constexpr uint32_t kVersion = 1;
constexpr size_t kHead = 24, kTail = 8;
constexpr size_t kChunk = (size_t) 16 << 20;   // a read or a write, and the checksum's step

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::FILE* open_file(const fs::path& p, const char* mode) {
#ifdef _WIN32
    const std::wstring m(mode, mode + std::strlen(mode));
    std::FILE* f = _wfopen(p.c_str(), m.c_str());
#else
    std::FILE* f = std::fopen(p.c_str(), mode);
#endif
    if (f) std::setvbuf(f, nullptr, _IONBF, 0);   // the reads and writes are large: no copy through a buffer
    return f;
}

bool rename_over(const fs::path& from, const fs::path& to) {
#ifdef _WIN32
    return MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

std::string hex(uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}

// The id in a name made by blob_path() or conv_path(): prefix, 16 hex digits, ext.
bool id_in(const std::string& name, const char* prefix, const char* ext, uint64_t& id) {
    const size_t p = std::strlen(prefix), e = std::strlen(ext);
    if (name.size() != p + 16 + e || name.compare(0, p, prefix) != 0 || name.compare(p + 16, e, ext) != 0) return false;
    id = 0;
    for (size_t i = p; i < p + 16; ++i) {
        const char c = name[i];
        const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (d < 0) return false;
        id = id << 4 | (uint64_t) d;
    }
    return id != 0;
}

void head_of(uint8_t* h, uint32_t magic, uint64_t key, uint64_t bytes) {
    std::memcpy(h, &magic, 4);
    std::memcpy(h + 4, &kVersion, 4);
    std::memcpy(h + 8, &key, 8);
    std::memcpy(h + 16, &bytes, 8);
}

bool head_is(const uint8_t* h, uint32_t magic, uint64_t key, uint64_t bytes) {
    uint8_t want[kHead];
    head_of(want, magic, key, bytes);
    return std::memcmp(h, want, kHead) == 0;
}

// Writes the file under a name of its own and renames it when it is whole.
bool write_file(const fs::path& path, uint32_t magic, uint64_t key, uint64_t seed, const uint8_t* data,
                size_t bytes) {
    fs::path tmp = path;
    tmp += ".tmp";
    std::FILE* f = open_file(tmp, "wb");
    if (!f) return false;
    uint8_t head[kHead];
    head_of(head, magic, key, bytes);
    strata::core::SessionHasher sum(seed);
    bool ok = std::fwrite(head, 1, kHead, f) == kHead;
    for (size_t at = 0; ok && at < bytes; at += kChunk) {
        const size_t n = std::min(kChunk, bytes - at);
        sum.update(data + at, n);
        ok = std::fwrite(data + at, 1, n, f) == n;
    }
    const uint64_t s = sum.digest();
    ok = ok && std::fwrite(&s, 1, kTail, f) == kTail;
    ok = std::fclose(f) == 0 && ok;
    ok = ok && rename_over(tmp, path);
    if (!ok) {
        std::error_code ec;
        fs::remove(tmp, ec);
    }
    return ok;
}

// A payload's numbers, in the order the file has them.
struct Out {
    std::vector<uint8_t> b;
    template<class T> void put(T v) {
        const size_t at = b.size();
        b.resize(at + sizeof v);
        std::memcpy(b.data() + at, &v, sizeof v);
    }
};
struct In {
    const uint8_t* p;
    size_t left;
    template<class T> bool get(T& v) {
        if (left < sizeof v) return false;
        std::memcpy(&v, p, sizeof v);
        p += sizeof v, left -= sizeof v;
        return true;
    }
};

std::vector<uint8_t> payload(const StoredConversation& c) {
    Out o;
    o.b.reserve(48 + c.seq.size() * 4 + c.marks.size() * 24 + c.rows.size() * 40);
    o.put(c.id), o.put(c.stamp), o.put(c.prompt);
    o.put((uint64_t) c.seq.size()), o.put((uint64_t) c.marks.size()), o.put((uint64_t) c.rows.size());
    const size_t at = o.b.size();
    o.b.resize(at + c.seq.size() * 4);
    if (!c.seq.empty()) std::memcpy(o.b.data() + at, c.seq.data(), c.seq.size() * 4);
    for (const StoredConversation::Mark& m : c.marks) o.put(m.pos), o.put(m.file), o.put(m.bytes);
    for (const StoredConversation::Rows& r : c.rows) o.put(r.file), o.put(r.bytes), o.put(r.first), o.put(r.count), o.put(r.keep);
    return o.b;
}

// The first line of two keys that differs, up to its first space: what changed.
std::string what_differs(const std::string& a, const std::string& b) {
    size_t i = 0;
    for (;;) {
        const size_t ea = a.find('\n', i), eb = b.find('\n', i);
        const std::string la = a.substr(i, ea == std::string::npos ? std::string::npos : ea - i);
        const std::string lb = b.substr(i, eb == std::string::npos ? std::string::npos : eb - i);
        if (la != lb) return la.substr(0, la.find(' '));
        if (ea == std::string::npos || eb == std::string::npos || ea != eb) return "the end";
        i = ea + 1;
    }
}

}  // namespace

fs::path ConversationDisk::subfolder(const fs::path& dir, const fs::path& model) {
    std::error_code ec;
    fs::path full = fs::absolute(model, ec);
    if (ec) full = model;
    const fs::path canon = fs::weakly_canonical(full, ec);
    if (!ec) full = canon;
    auto name = full.native();
#ifdef _WIN32
    for (auto& ch : name) ch = (wchar_t) std::towlower(ch);   // one file, whatever case its path was given in
#endif
    const uint64_t h = strata::core::session_hash64(name.data(), name.size() * sizeof(name[0]), 0x53504D4F44454C31ull);
    std::string stem;
    for (const auto ch : model.stem().native()) {
        if (stem.size() >= 48) break;
        const bool plain = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
                           ch == '.' || ch == '-' || ch == '_';
        stem += plain ? (char) ch : '_';
    }
    return dir / (stem + "-" + hex(h).substr(0, 8));
}

fs::path ConversationDisk::blob_path(uint64_t id) const { return s_.dir / ("b-" + hex(id) + ".blob"); }
fs::path ConversationDisk::conv_path(uint64_t id) const { return s_.dir / ("c-" + hex(id) + ".conv"); }

void ConversationDisk::say(const char* fmt, ...) const {
    if (!s_.log) return;
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    std::fprintf(s_.log, "strata-poly: conversation disk: %s\n", line);
}

bool ConversationDisk::open(const Settings& s, std::vector<StoredConversation>& out) {
    s_ = s;
    out.clear();
    key_hash_ = strata::core::session_hash64(s.key.data(), s.key.size(), 0x5350434F4E564B31ull);
    std::error_code ec;
    fs::create_directories(s_.dir, ec);
    if (ec) {
        say("cannot make %s (%s); the conversations stay in RAM only", s_.dir.string().c_str(), ec.message().c_str());
        return false;
    }
    const fs::path lock = s_.dir / "lock";
#ifdef _WIN32
    const HANDLE h = CreateFileW(lock.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        say("%s is in use by another engine (or cannot be opened, error %lu); the conversations stay in RAM only",
            s_.dir.string().c_str(), (unsigned long) GetLastError());
        return false;
    }
    lock_ = h;
#else
    const int fd = ::open(lock.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (fd >= 0) ::close(fd);
        say("%s is in use by another engine (or cannot be opened); the conversations stay in RAM only",
            s_.dir.string().c_str());
        return false;
    }
    lock_ = (void*) (intptr_t) (fd + 1);
#endif

    // The key: files made for another one are deleted.
    std::string old;
    if (std::FILE* f = open_file(s_.dir / "key.txt", "rb")) {
        char b[4096];
        for (size_t n; (n = std::fread(b, 1, sizeof b, f)) > 0;) old.append(b, n);
        std::fclose(f);
    }
    if (old != s_.key) {
        std::vector<fs::path> files;
        for (const fs::directory_entry& e : fs::directory_iterator(s_.dir, ec)) {
            const std::string name = e.path().filename().string();
            if (name.rfind("c-", 0) == 0 || name.rfind("b-", 0) == 0 || e.path().extension() == ".tmp")
                files.push_back(e.path());
        }
        size_t gone = 0;
        for (const fs::path& f : files) gone += fs::remove(f, ec) ? 1 : 0;
        const fs::path tmp = s_.dir / "key.txt.tmp";
        std::FILE* f = open_file(tmp, "wb");
        const bool ok = f && std::fwrite(s_.key.data(), 1, s_.key.size(), f) == s_.key.size();
        if (f) std::fclose(f);
        if (!ok || !rename_over(tmp, s_.dir / "key.txt")) {
            say("cannot write %s; the conversations stay in RAM only", (s_.dir / "key.txt").string().c_str());
            return false;
        }
        if (!old.empty())
            say("the files there were made for another %s; %zu deleted", what_differs(s_.key, old).c_str(), gone);
    }

    // What is there: the blobs whole in size, then the conversations whose blobs are all there.
    std::vector<fs::path> convs, gone;
    for (const fs::directory_entry& e : fs::directory_iterator(s_.dir, ec)) {
        const std::string name = e.path().filename().string();
        uint64_t id = 0, size = e.file_size(ec);
        if (e.path().extension() == ".tmp") {
            gone.push_back(e.path());   // a file an engine was writing when it ended
        } else if (id_in(name, "b-", ".blob", id)) {
            if (size >= kHead + kTail) {
                blobs_[id] = size - kHead - kTail;
                held_ += size - kHead - kTail;
            } else {
                gone.push_back(e.path());
            }
            next_id_ = std::max(next_id_, id + 1);
        } else if (id_in(name, "c-", ".conv", id)) {
            convs.push_back(e.path());
            next_id_ = std::max(next_id_, id + 1);
        }
    }
    for (const fs::path& f : gone) fs::remove(f, ec);
    size_t bad = 0;
    for (const fs::path& p : convs) {
        std::vector<uint8_t> b;
        if (std::FILE* f = open_file(p, "rb")) {
            uint8_t chunk[1 << 16];
            for (size_t n; (n = std::fread(chunk, 1, sizeof chunk, f)) > 0;) b.insert(b.end(), chunk, chunk + n);
            std::fclose(f);
        }
        StoredConversation c;
        uint64_t id = 0;
        id_in(p.filename().string(), "c-", ".conv", id);
        bool ok = parse(b, c) && c.id == id;
        for (const StoredConversation::Mark& m : c.marks) ok = ok && blobs_.count(m.file) && blobs_[m.file] == m.bytes;
        for (const StoredConversation::Rows& r : c.rows) ok = ok && blobs_.count(r.file) && blobs_[r.file] == r.bytes;
        if (!ok) {
            ++bad;
            fs::remove(p, ec);
            continue;
        }
        convs_[c.id] = strata::core::session_hash64(b.data() + kHead, b.size() - kHead - kTail, 0);
        out.push_back(std::move(c));
    }
    if (bad) say("%zu conversation files were cut short or not whole, and were deleted", bad);
    open_ = true;
    th_ = std::thread([this] { run(); });
    return true;
}

bool ConversationDisk::parse(const std::vector<uint8_t>& b, StoredConversation& c) const {
    if (b.size() < kHead + kTail) return false;
    const uint64_t n = b.size() - kHead - kTail;
    if (!head_is(b.data(), kConvMagic, key_hash_, n)) return false;
    In in{b.data() + kHead, (size_t) n};
    uint64_t n_seq = 0, n_marks = 0, n_rows = 0;
    if (!in.get(c.id) || !in.get(c.stamp) || !in.get(c.prompt) || !in.get(n_seq) || !in.get(n_marks) ||
        !in.get(n_rows) || n_seq > (1ull << 28) || n_marks > 4096 || n_rows > (1ull << 20) ||
        in.left != n_seq * 4 + n_marks * 24 + n_rows * 40)
        return false;
    strata::core::SessionHasher sum(key_hash_ ^ c.id);
    sum.update(b.data() + kHead, (size_t) n);
    uint64_t want = 0;
    std::memcpy(&want, b.data() + kHead + n, kTail);
    if (sum.digest() != want) return false;
    c.seq.resize((size_t) n_seq);
    if (n_seq) std::memcpy(c.seq.data(), in.p, (size_t) n_seq * 4);
    in.p += n_seq * 4, in.left -= (size_t) n_seq * 4;
    c.marks.resize((size_t) n_marks);
    for (StoredConversation::Mark& m : c.marks) in.get(m.pos), in.get(m.file), in.get(m.bytes);
    c.rows.resize((size_t) n_rows);
    for (StoredConversation::Rows& r : c.rows) in.get(r.file), in.get(r.bytes), in.get(r.first), in.get(r.count), in.get(r.keep);
    // a checkpoint at least, in order, inside the tokens; rows back to back from row 0 up to the last one
    if (c.marks.empty() || c.prompt < 0 || c.prompt > (int64_t) n_seq) return false;
    for (size_t k = 0; k < c.marks.size(); ++k)
        if (c.marks[k].pos < 1 || c.marks[k].pos > (int64_t) n_seq || (k && c.marks[k].pos <= c.marks[k - 1].pos))
            return false;
    int64_t end = 0;
    for (const StoredConversation::Rows& r : c.rows) {
        if (r.first != end || r.count < 1 || r.keep < 1 || r.keep > r.count) return false;
        end += r.keep;
    }
    return end >= c.marks.back().pos;
}

ConversationDisk::~ConversationDisk() {
    if (th_.joinable()) {
        {
            std::lock_guard<std::mutex> l(m_);
            stop_ = true;
        }
        cv_.notify_all();
        th_.join();
    }
#ifdef _WIN32
    if (lock_) CloseHandle((HANDLE) lock_);
#else
    if (lock_) ::close((int) (intptr_t) lock_ - 1);
#endif
}

void ConversationDisk::put(Job j) {
    {
        std::lock_guard<std::mutex> l(m_);
        if (j.kind == Job::Blob) queued_ += j.bytes;
        q_.push_back(std::move(j));
    }
    cv_.notify_one();
}

size_t ConversationDisk::commit(std::vector<Want> want) {
    if (!open_) return 0;
    uint64_t queued = 0;
    {
        std::lock_guard<std::mutex> l(m_);
        if (broken_) return 0;
        queued = queued_;
    }
    // What the files may take: the limit, and what the volume has free beside the files already there.
    uint64_t room = s_.limit;
    std::error_code ec;
    const fs::space_info sp = fs::space(s_.dir, ec);
    if (!ec) {
        const uint64_t can = sp.available + (held_ - std::min(held_, queued));
        room = std::min(room, can > s_.min_free ? can - s_.min_free : 0);
    }
    std::unordered_map<uint64_t, uint64_t> kept;   // the blobs of the conversations that go to disk
    std::vector<size_t> in;
    uint64_t total = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        uint64_t extra = 0;
        bool whole = true;
        std::unordered_set<uint64_t> mine;
        for (const Source& b : want[i].blobs) {
            if (kept.count(b.file) || !mine.insert(b.file).second) continue;
            extra += b.bytes;
            whole = whole && (b.data || (blobs_.count(b.file) && blobs_[b.file] == b.bytes));
        }
        if (!whole || total + extra > room) continue;   // the older ones may still fit
        for (const Source& b : want[i].blobs) kept.emplace(b.file, b.bytes);
        total += extra;
        in.push_back(i);
    }
    std::unordered_set<uint64_t> named;
    for (const size_t i : in) {
        for (Source& b : want[i].blobs) {
            if (blobs_.count(b.file)) continue;
            blobs_[b.file] = b.bytes;
            held_ += b.bytes;
            put(Job{Job::Blob, b.file, std::move(b.keep), b.data, b.bytes, {}});
        }
    }
    for (const size_t i : in) {
        const StoredConversation& c = want[i].c;
        named.insert(c.id);
        std::vector<uint8_t> p = payload(c);
        const uint64_t h = strata::core::session_hash64(p.data(), p.size(), 0);
        if (convs_.count(c.id) && convs_[c.id] == h) continue;
        convs_[c.id] = h;
        put(Job{Job::Conv, c.id, nullptr, nullptr, 0, std::move(p)});
    }
    for (auto it = convs_.begin(); it != convs_.end();) {
        if (named.count(it->first)) {
            ++it;
            continue;
        }
        put(Job{Job::DropConv, it->first, nullptr, nullptr, 0, {}});
        it = convs_.erase(it);
    }
    for (auto it = blobs_.begin(); it != blobs_.end();) {
        if (kept.count(it->first)) {
            ++it;
            continue;
        }
        put(Job{Job::DropBlob, it->first, nullptr, nullptr, 0, {}});
        held_ -= it->second;
        it = blobs_.erase(it);
    }
    return in.size();
}

bool ConversationDisk::write_blob(const Job& j) {
    return write_file(blob_path(j.id), kBlobMagic, key_hash_, key_hash_ ^ j.id, j.data, j.bytes);
}

bool ConversationDisk::write_conv(const Job& j) {
    // the checksum's seed is the conversation's id, which its payload begins with
    uint64_t id = 0;
    std::memcpy(&id, j.conv.data(), 8);
    return write_file(conv_path(j.id), kConvMagic, key_hash_, key_hash_ ^ id, j.conv.data(), j.conv.size());
}

void ConversationDisk::run() {
#ifdef _WIN32
    // below the engine's threads, for the CPU, the disk and the memory: the pages this thread writes are the first
    // the system lets go, not the model's
    SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
#endif
    for (;;) {
        Job j;
        bool skip;
        {
            std::unique_lock<std::mutex> l(m_);
            cv_.wait(l, [this] { return stop_ || !q_.empty(); });
            if (q_.empty()) return;
            j = std::move(q_.front());
            q_.pop_front();
            busy_ = true;
            skip = broken_ && (j.kind == Job::Blob || j.kind == Job::Conv);
            if (!in_batch_) in_batch_ = true, batch_t0_ = Clock::now(), batch_from_ = totals_;
        }
        const Clock::time_point t0 = Clock::now();
        bool ok = true;
        std::error_code ec;
        if (!skip) {
            switch (j.kind) {
            case Job::Blob: ok = write_blob(j); break;
            case Job::Conv: ok = write_conv(j); break;
            case Job::DropBlob: fs::remove(blob_path(j.id), ec); break;
            case Job::DropConv: fs::remove(conv_path(j.id), ec); break;
            }
        }
        const double ms = ms_since(t0);
        bool failed = false, done = false;
        Totals from, now;
        Clock::time_point since;
        {
            std::lock_guard<std::mutex> l(m_);
            busy_ = false;
            if (j.kind == Job::Blob) queued_ -= j.bytes;
            if (!ok && !broken_) broken_ = failed = true;
            if (ok && !skip && (j.kind == Job::Blob || j.kind == Job::Conv)) {
                totals_.written += j.kind == Job::Blob ? j.bytes : j.conv.size();
                ++totals_.files;
                totals_.write_ms += ms;
            } else if (j.kind == Job::DropBlob || j.kind == Job::DropConv) {
                ++totals_.deleted;
            }
            if (q_.empty()) {   // what the last commit() asked for is done
                idle_.notify_all();
                done = true;
                in_batch_ = false;
                from = batch_from_, now = totals_, since = batch_t0_;
            }
        }
        if (failed)
            say("cannot write %s (the disk full?); nothing more is written until the engine starts again",
                (j.kind == Job::Blob ? blob_path(j.id) : conv_path(j.id)).string().c_str());
        if (done && (now.files > from.files || now.deleted > from.deleted))
            say("wrote %.1f MiB in %llu files and deleted %llu in %.0f ms", (double) (now.written - from.written) / (1 << 20),
                (unsigned long long) (now.files - from.files), (unsigned long long) (now.deleted - from.deleted),
                ms_since(since));
    }
}

void ConversationDisk::flush() {
    if (!th_.joinable()) return;
    std::unique_lock<std::mutex> l(m_);
    idle_.wait(l, [this] { return q_.empty() && !busy_; });
}

bool ConversationDisk::read_blob(uint64_t file, size_t bytes, uint8_t* dst) {
    if (!open_) return false;
    const Clock::time_point t0 = Clock::now();
    std::FILE* f = open_file(blob_path(file), "rb");
    if (!f) return false;
    uint8_t head[kHead];
    bool ok = std::fread(head, 1, kHead, f) == kHead && head_is(head, kBlobMagic, key_hash_, bytes);
    strata::core::SessionHasher sum(key_hash_ ^ file);
    for (size_t at = 0; ok && at < bytes; at += kChunk) {
        const size_t n = std::min(kChunk, bytes - at);
        ok = std::fread(dst + at, 1, n, f) == n;
        if (ok) sum.update(dst + at, n);
    }
    uint64_t want = 0;
    ok = ok && std::fread(&want, 1, kTail, f) == kTail && want == sum.digest();
    std::fclose(f);
    std::lock_guard<std::mutex> l(m_);
    if (ok) {
        totals_.read += bytes;
        totals_.read_ms += ms_since(t0);
    }
    return ok;
}

uint64_t ConversationDisk::queued_bytes() const {
    std::lock_guard<std::mutex> l(m_);
    return queued_;
}

ConversationDisk::Totals ConversationDisk::totals() const {
    std::lock_guard<std::mutex> l(m_);
    return totals_;
}

}  // namespace strata::poly
