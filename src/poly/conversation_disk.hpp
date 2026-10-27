// src/poly/conversation_disk.hpp - the conversation cache's copy on disk (--conversation-disk), so that the
// conversations an engine had parked are there for the next one.  The cache is in the process's RAM, and a server
// that is ended takes it along: the FastStrata Gateway ends the server's process tree (taskkill /F) when it swaps
// the model or closes, and a PC that restarts ends everything.
//
// What the cache holds is written once and never changed (conversations.hpp), and so are the files: one for every
// blob (a checkpoint's state, a run of attention rows) and one for every conversation, which lists its tokens and
// the blobs it is made of, blobs that conversations share once.  A thread of its own writes them, in the order they
// were asked for, while the engine answers or waits.  A file is written under another name and renamed when it is
// whole, and a conversation's after the blobs it names, so an engine ended at any point leaves the last whole set.
// The next start takes the conversations whose files are all there and of the right size; a blob's checksum is
// checked when the blob is first read, and a conversation that has a bad one is dropped then.
//
// The folder given holds a subfolder for each model file.  Its key.txt says what the files there are valid for:
// the model's file, the engine's executable and the options that change what a state or a row is or how it was
// computed.  Where the key differs the subfolder's files are deleted.  A lock file keeps a second engine out of a
// subfolder that one uses.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata::poly {

/// A conversation as its file has it: where it is in the cache's order, the tokens it can be come back to (up to
/// its last checkpoint), its checkpoints and its rows, by the blob files they are in.
struct StoredConversation {
    struct Mark {
        int64_t pos;
        uint64_t file, bytes;
    };
    struct Rows {
        uint64_t file, bytes;
        int64_t first, count, keep;   // as Conversations' Rows: the file holds `count` rows from `first`
    };
    uint64_t id = 0;      ///< its file, c-<id>.conv
    uint64_t stamp = 0;   ///< when it was last used, as the cache counts: higher is later
    int64_t prompt = 0;
    std::vector<int32_t> seq;
    std::vector<Mark> marks;
    std::vector<Rows> rows;
};

class ConversationDisk {
public:
    struct Settings {
        std::filesystem::path dir;   ///< the subfolder (subfolder())
        std::string key;             ///< what key.txt holds
        uint64_t limit = 0;          ///< bytes the files may take
        uint64_t min_free = 0;       ///< bytes the volume keeps free besides them
        std::FILE* log = nullptr;
    };

    /// A blob a conversation is made of, for commit(): its file, and its bytes while they are in RAM (null when they
    /// are on disk only).  `keep` holds them until they are written.
    struct Source {
        uint64_t file = 0;
        size_t bytes = 0;
        std::shared_ptr<const void> keep;
        const uint8_t* data = nullptr;
    };
    struct Want {
        StoredConversation c;
        std::vector<Source> blobs;
    };

    /// The subfolder of `dir` for a model file: its name and a hash of its full path.
    static std::filesystem::path subfolder(const std::filesystem::path& dir, const std::filesystem::path& model);

    ConversationDisk() = default;
    ConversationDisk(const ConversationDisk&) = delete;
    ConversationDisk& operator=(const ConversationDisk&) = delete;
    /// Writes what is still to be written, and stops the thread.
    ~ConversationDisk();

    /// Takes the subfolder, deletes what is not valid there, and gives the conversations that are.  False, with a
    /// line on the log, when it cannot be used: the engine then runs without it.
    bool open(const Settings& s, std::vector<StoredConversation>& out);
    bool is_open() const { return open_; }

    uint64_t new_id() { return next_id_++; }

    /// The conversations the cache holds, the one it would drop last first.  As many of them go to disk as the
    /// limit and the volume's free space take; the files of the others, and the blobs no one names any more, are
    /// deleted after the new ones are written.  Returns how many of them are on disk then.
    size_t commit(std::vector<Want> want);

    /// Reads blob `file`, `bytes` long, into dst and checks it.  False: it is missing or not what was written.
    bool read_blob(uint64_t file, size_t bytes, uint8_t* dst);

    /// Waits until everything asked for is written.
    void flush();

    /// What went to disk so far, for the log.
    struct Totals {
        uint64_t written = 0, files = 0, deleted = 0, read = 0;
        double write_ms = 0, read_ms = 0;
    };
    Totals totals() const;
    uint64_t held_bytes() const { return held_; }   ///< the bytes of the blob files there or on their way
    uint64_t queued_bytes() const;                  ///< of them, those still to be written

private:
    struct Job {
        enum Kind { Blob, Conv, DropBlob, DropConv } kind;
        uint64_t id = 0;
        std::shared_ptr<const void> keep;
        const uint8_t* data = nullptr;
        size_t bytes = 0;
        std::vector<uint8_t> conv;   // a conversation's payload
    };

    void run();
    bool write_blob(const Job& j);
    bool write_conv(const Job& j);
    void put(Job j);
    std::filesystem::path blob_path(uint64_t id) const;
    std::filesystem::path conv_path(uint64_t id) const;
    bool parse(const std::vector<uint8_t>& file, StoredConversation& c) const;
    void say(const char* fmt, ...) const;

    Settings s_;
    bool open_ = false;
    uint64_t key_hash_ = 0;
    uint64_t next_id_ = 1;
    void* lock_ = nullptr;   // the lock file's handle (Windows) or descriptor + 1 (elsewhere)
    std::unordered_map<uint64_t, uint64_t> blobs_;   // the blob files there or on their way, and their bytes
    std::unordered_map<uint64_t, uint64_t> convs_;   // the conversation files, and a hash of what each holds
    uint64_t held_ = 0;                              // blobs_' bytes

    mutable std::mutex m_;
    std::condition_variable cv_, idle_;
    std::deque<Job> q_;
    uint64_t queued_ = 0;                            // the bytes of the blobs still to be written
    bool in_batch_ = false;                          // jobs since the queue was last empty: for the log
    std::chrono::steady_clock::time_point batch_t0_{};
    Totals batch_from_;
    bool busy_ = false, stop_ = false, broken_ = false;
    Totals totals_;
    std::thread th_;
};

}  // namespace strata::poly
