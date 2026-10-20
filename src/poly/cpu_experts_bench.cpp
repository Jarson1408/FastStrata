// src/poly/cpu_experts_bench.cpp - how fast the CPU's experts read the model's bytes, against what the RAM gives.
//
//   poly_cpu_experts_bench MODEL.gguf [--threads N] [--profile FILE] [--windows N] [--tokens 1,4,8]
//                          [--stream-gib N] [--skip-stream] [--multi] [--gu-rows N] [--down-rows N] [--check]
//                          [--warm N] [--copy normal|large] [--copy-gib N]
//
// No card is used.  Two figures, each over the same threads the engine would use:
//   - stream: a plain read of a buffer of the process's own (4 KB pages, with and without software prefetch, then
//     2 MB pages if the account may lock pages in memory), and of the routed experts in the mapped file, every thread
//     its own part: what the RAM gives;
//   - experts: cpu_experts() over windows of a few tokens, eight experts a token drawn by the expert profile's counts
//     (uniformly without one), every MoE layer in turn as an answer goes: the experts' bytes over the time it took.
// A window's tokens here share fewer experts than an answer's do (they are drawn independently), so the bytes a
// window are a little more than in the engine.  --multi and the options after it: CpuKernelOptions (the default is
// the plain path).  --check: before timing, every window's output of the timed path against the plain path's, bit
// for bit (it leaves the experts in the caches, so the times of a --check run are not the RAM's).  --warm N: the
// call's first N experts are read into the caches, untimed, before each call: what a perfect guess of them would
// give.  --copy normal|large: the experts (of as many whole layers as --copy-gib holds, all without it) copied into the
// process's own memory in 4 KB pages or, with large, in 2 MB pages when the account holds "Lock pages in memory" (it
// says which it got, and falls back to 4 KB pages without the right), and timed there instead of in the mapped file.
#include "cpu_experts.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include "ggml.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

struct MoeLayer {
    const uint8_t *gate, *up, *down;
    size_t gu_bytes, d_bytes;   // one expert's
    int gu_type, d_type;
};

// Reads [p, p + n) with 32-byte loads, four at a time, prefetching `ahead` bytes on (0: none); the xor keeps the
// loads.
uint64_t read_span(const uint8_t* p, size_t n, size_t ahead) {
    __m256i a = _mm256_setzero_si256(), b = a, c = a, d = a;
    size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        if (ahead > 0 && i + ahead + 128 <= n) {
            _mm_prefetch((const char*) p + i + ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) p + i + ahead + 64, _MM_HINT_T0);
        }
        a = _mm256_xor_si256(a, _mm256_loadu_si256((const __m256i*) (p + i)));
        b = _mm256_xor_si256(b, _mm256_loadu_si256((const __m256i*) (p + i + 32)));
        c = _mm256_xor_si256(c, _mm256_loadu_si256((const __m256i*) (p + i + 64)));
        d = _mm256_xor_si256(d, _mm256_loadu_si256((const __m256i*) (p + i + 96)));
    }
    a = _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d));
    uint64_t s = (uint64_t) _mm256_extract_epi64(a, 0) ^ (uint64_t) _mm256_extract_epi64(a, 3);
    for (; i < n; ++i) s += p[i];
    return s;
}

// Every thread reads its share of the spans; GB/s of the best of `reps` passes.
double stream(const std::vector<std::pair<const uint8_t*, size_t>>& spans, int threads, int reps, size_t ahead = 0) {
    size_t total = 0;
    for (const auto& s : spans) total += s.second;
    double best = 0.0;
    std::atomic<uint64_t> sink{0};
    for (int r = 0; r < reps; ++r) {
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t)
            th.emplace_back([&, t] {
                ready.fetch_add(1);
                while (!go.load()) _mm_pause();
                uint64_t s = 0;
                for (const auto& sp : spans) {
                    const size_t per = (sp.second / (size_t) threads + 4095) / 4096 * 4096;
                    const size_t lo = std::min(sp.second, per * (size_t) t), hi = std::min(sp.second, lo + per);
                    s ^= read_span(sp.first + lo, hi - lo, ahead);
                }
                sink.fetch_xor(s);
            });
        while (ready.load() < threads) _mm_pause();
        const Clock::time_point t0 = Clock::now();
        go.store(true);
        for (auto& t : th) t.join();
        best = std::max(best, (double) total / secs(t0) / 1e9);
    }
    if (sink.load() == 42) std::printf(" ");
    return best;
}

#ifdef _WIN32
// SeLockMemoryPrivilege, which 2 MB pages need: only an account the policy gives it ("Lock pages in memory") has it.
bool may_lock_pages(std::string& why) {
    HANDLE tok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
        why = "OpenProcessToken failed (" + std::to_string(GetLastError()) + ")";
        return false;
    }
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid) &&
              AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr);
    const DWORD err = GetLastError();   // ERROR_NOT_ALL_ASSIGNED when the account lacks it
    CloseHandle(tok);
    if (!ok || err != ERROR_SUCCESS) {
        why = err == ERROR_NOT_ALL_ASSIGNED ? "the account does not hold SeLockMemoryPrivilege (Lock pages in memory)"
                                            : "AdjustTokenPrivileges failed (" + std::to_string(err) + ")";
        return false;
    }
    return true;
}
#endif

// `bytes` of the process's own memory, in 2 MB pages when `large` is asked and the account may lock pages (else in
// 4 KB pages); `got` says which, and why not the large ones.
uint8_t* own_pages(size_t bytes, bool large, size_t& reserved, std::string& got) {
#ifdef _WIN32
    if (large) {
        std::string why;
        if (may_lock_pages(why)) {
            const size_t page = GetLargePageMinimum();
            reserved = (bytes + page - 1) / page * page;
            if (void* p = VirtualAlloc(nullptr, reserved, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE)) {
                got = std::to_string(page >> 10) + " KB pages";
                return (uint8_t*) p;
            }
            why = "VirtualAlloc with MEM_LARGE_PAGES failed (" + std::to_string(GetLastError()) +
                  "; the RAM may be too fragmented for that many 2 MB pages)";
        }
        got = "4 KB pages, as the large ones were not had: " + why;
    } else {
        got = "4 KB pages";
    }
    reserved = (bytes + 4095) / 4096 * 4096;
    return (uint8_t*) VirtualAlloc(nullptr, reserved, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    got = large ? "4 KB pages (2 MB pages are only tried on Windows here)" : "4 KB pages";
    reserved = (bytes + 4095) / 4096 * 4096;
    return (uint8_t*) std::aligned_alloc(4096, reserved);
#endif
}

void free_pages(uint8_t* p) {
#ifdef _WIN32
    VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

std::vector<double> read_profile(const std::string& path, int n_layer, int n_expert) {
    std::vector<double> w((size_t) n_layer * n_expert, 1.0);
    if (path.empty()) return w;
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        std::fprintf(stderr, "no profile %s: experts drawn uniformly\n", path.c_str());
        return w;
    }
    int il = 0, e = 0;
    long long n = 0;
    while (std::fscanf(f, "%d %d %lld", &il, &e, &n) == 3)
        if (il >= 0 && il < n_layer && e >= 0 && e < n_expert) w[(size_t) il * n_expert + e] = 1.0 + (double) n;
    std::fclose(f);
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: poly_cpu_experts_bench MODEL.gguf [--threads N] [--profile FILE] [--windows N] "
                             "[--tokens 1,4,8] [--stream-gib N] [--skip-stream] [--multi] [--gu-rows N] "
                             "[--down-rows N] [--check] [--warm N] [--copy normal|large] [--copy-gib N]\n");
        return 2;
    }
    int threads = 8, windows = 24, warm = 0;
    double stream_gib = 2.0, copy_gib = 0.0;
    std::string copy;   // "", "normal" or "large"
    bool skip_stream = false, check = false;
    strata::poly::CpuKernelOptions ko;
    std::string profile;
    std::vector<int> token_counts = {1, 2, 4, 8};
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--profile") profile = val();
        else if (a == "--windows") windows = std::atoi(val().c_str());
        else if (a == "--stream-gib") stream_gib = std::atof(val().c_str());
        else if (a == "--skip-stream") skip_stream = true;
        else if (a == "--multi") ko.multi = true;
        else if (a == "--gu-rows") ko.gu_rows = std::atoi(val().c_str());
        else if (a == "--down-rows") ko.down_rows = std::atoi(val().c_str());
        else if (a == "--check") check = true;
        else if (a == "--warm") warm = std::atoi(val().c_str());
        else if (a == "--copy") copy = val();
        else if (a == "--copy-gib") copy_gib = std::atof(val().c_str());
        else if (a == "--tokens") {
            token_counts.clear();
            const std::string s = val();
            for (size_t p = 0; p < s.size();) {
                token_counts.push_back(std::atoi(s.c_str() + p));
                p = s.find(',', p);
                if (p == std::string::npos) break;
                ++p;
            }
        }
    }
    using strata::GgufFile;
    GgufFile g(argv[1]);
    std::vector<MoeLayer> layers;
    int n_embd = 0, n_ff = 0, n_expert = 0;
    for (int il = 0;; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        if (!g.find(b + "attn_norm.weight") && !g.find(b + "ffn_gate_exps.weight")) break;
        const strata::TensorInfo* tg = g.find(b + "ffn_gate_exps.weight");
        const strata::TensorInfo* tu = g.find(b + "ffn_up_exps.weight");
        const strata::TensorInfo* td = g.find(b + "ffn_down_exps.weight");
        if (!tg || !tu || !td) continue;
        n_embd = (int) tg->shape[0];
        n_ff = (int) tg->shape[1];
        n_expert = (int) tg->shape[2];
        MoeLayer L;
        L.gate = g.tensor_data(*tg);
        L.up = g.tensor_data(*tu);
        L.down = g.tensor_data(*td);
        L.gu_type = (int) tg->type;
        L.d_type = (int) td->type;
        L.gu_bytes = ggml_row_size((ggml_type) tg->type, n_embd) * (size_t) n_ff;
        L.d_bytes = ggml_row_size((ggml_type) td->type, n_ff) * (size_t) n_embd;
        layers.push_back(L);
    }
    if (layers.empty()) {
        std::fprintf(stderr, "no routed experts (ffn_gate_exps / ffn_up_exps / ffn_down_exps) in %s\n", argv[1]);
        return 1;
    }
    const size_t one_expert = 2 * layers[0].gu_bytes + layers[0].d_bytes;
    std::printf("%zu MoE layers, %d experts of %.2f MiB (n_embd %d, n_ff %d), %d threads\n", layers.size(), n_expert,
                (double) one_expert / (1 << 20), n_embd, n_ff, threads);

    std::vector<std::pair<const uint8_t*, size_t>> mapped;
    for (const MoeLayer& L : layers) {
        mapped.emplace_back(L.gate, (size_t) n_expert * L.gu_bytes);
        mapped.emplace_back(L.up, (size_t) n_expert * L.gu_bytes);
        mapped.emplace_back(L.down, (size_t) n_expert * L.d_bytes);
    }
    // every page into the process once, as the engine's map_pages() does at a start
    {
        const Clock::time_point t0 = Clock::now();
        const double gbs = stream(mapped, threads, 1);
        std::printf("first read of the mapped experts (page faults; from the disk what the file cache lacks): %.1f GB/s, "
                    "%.1f s\n", gbs, secs(t0));
    }
    if (!skip_stream) {
        const size_t bytes = (size_t) (stream_gib * (1 << 30)) / 4096 * 4096;
        uint8_t* own = nullptr;
#ifdef _WIN32
        own = (uint8_t*) VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        own = (uint8_t*) std::aligned_alloc(4096, bytes);
#endif
        std::memset(own, 1, bytes);
        std::printf("stream, own buffer of %.1f GiB, 4 KB pages: %.1f GB/s; prefetching 512 / 2048 / 8192 bytes "
                    "ahead: %.1f / %.1f / %.1f GB/s\n",
                    stream_gib, stream({{own, bytes}}, threads, 5), stream({{own, bytes}}, threads, 5, 512),
                    stream({{own, bytes}}, threads, 5, 2048), stream({{own, bytes}}, threads, 5, 8192));
#ifdef _WIN32
        VirtualFree(own, 0, MEM_RELEASE);
        std::string why;
        if (may_lock_pages(why)) {
            const size_t large = GetLargePageMinimum();
            const size_t lbytes = (bytes + large - 1) / large * large;
            uint8_t* big = (uint8_t*) VirtualAlloc(nullptr, lbytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                                                   PAGE_READWRITE);
            if (big) {
                std::memset(big, 1, lbytes);
                std::printf("stream, own buffer, %zu KB pages: %.1f GB/s\n", large >> 10,
                            stream({{big, lbytes}}, threads, 5));
                VirtualFree(big, 0, MEM_RELEASE);
            } else {
                std::printf("stream, own buffer, large pages: VirtualAlloc failed (%lu)\n", GetLastError());
            }
        } else {
            std::printf("stream, own buffer, large pages: not tried, %s\n", why.c_str());
        }
#else
        std::free(own);
#endif
        size_t all = 0;
        for (const auto& s : mapped) all += s.second;
        std::printf("stream, the routed experts in the mapped file (%.1f GiB): %.1f GB/s; prefetching 2048 bytes "
                    "ahead: %.1f GB/s\n",
                    (double) all / (1 << 30), stream(mapped, threads, 3), stream(mapped, threads, 3, 2048));
    }

    // --copy: the experts of as many whole layers as --copy-gib holds (all without it) copied into the process's own
    // memory, and everything below reads the copy: what a copy for the CPU would give, in 4 KB or 2 MB pages
    uint8_t* copied = nullptr;
    if (copy == "normal" || copy == "large") {
        // a layer's own sizes: the down experts' type differs between layers (Q5_K and Q6_K in UD-Q4_K_M)
        const auto layer_bytes = [&](const MoeLayer& L) { return (size_t) n_expert * (2 * L.gu_bytes + L.d_bytes); };
        const size_t most = copy_gib > 0.0 ? (size_t) (copy_gib * (1 << 30)) : SIZE_MAX;
        size_t keep = 0, total = 0;
        while (keep < layers.size() && total + layer_bytes(layers[keep]) <= most) total += layer_bytes(layers[keep++]);
        if (keep == 0) {
            std::fprintf(stderr, "--copy-gib %.2f holds no layer (%.2f GiB a layer)\n", copy_gib,
                         (double) layer_bytes(layers[0]) / (1 << 30));
            return 1;
        }
        layers.resize(keep);
        size_t reserved = 0;
        std::string got;
        const Clock::time_point t0 = Clock::now();
        copied = own_pages(total, copy == "large", reserved, got);
        if (!copied) {
            std::fprintf(stderr, "could not allocate %.2f GiB for the copy\n", (double) total / (1 << 30));
            return 1;
        }
        std::vector<std::pair<const uint8_t*, size_t>> from;
        std::vector<uint8_t*> to;
        size_t at = 0;
        for (MoeLayer& L : layers) {
            for (const auto& [src, len, dst] : {std::tuple<const uint8_t*, size_t, const uint8_t**>{L.gate, (size_t) n_expert * L.gu_bytes, &L.gate},
                                                {L.up, (size_t) n_expert * L.gu_bytes, &L.up},
                                                {L.down, (size_t) n_expert * L.d_bytes, &L.down}}) {
                for (size_t o = 0; o < len; o += size_t(64) << 20) {
                    from.emplace_back(src + o, std::min(size_t(64) << 20, len - o));
                    to.push_back(copied + at + o);
                }
                *dst = copied + at;
                at += len;
            }
        }
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t)
            th.emplace_back([&] {
                for (size_t i; (i = next.fetch_add(1)) < from.size();) std::memcpy(to[i], from[i].first, from[i].second);
            });
        for (auto& t : th) t.join();
        std::printf("copy: %zu of the layers' experts, %.2f GiB, in %s; allocated and copied in %.1f s\n", keep,
                    (double) total / (1 << 30), got.c_str(), secs(t0));
        std::vector<std::pair<const uint8_t*, size_t>> spans;
        for (const MoeLayer& L : layers) {
            spans.emplace_back(L.gate, (size_t) n_expert * L.gu_bytes);
            spans.emplace_back(L.up, (size_t) n_expert * L.gu_bytes);
            spans.emplace_back(L.down, (size_t) n_expert * L.d_bytes);
        }
        if (!skip_stream)
            std::printf("stream, the copy: %.1f GB/s; prefetching 2048 bytes ahead: %.1f GB/s\n", stream(spans, threads, 3),
                        stream(spans, threads, 3, 2048));
    } else if (!copy.empty()) {
        std::fprintf(stderr, "--copy takes normal or large\n");
        return 2;
    }

    const std::vector<double> weights = read_profile(profile, (int) layers.size() + 8, n_expert);
    strata::poly::Workers workers(threads);
    if (ko.multi && !strata::poly::kq::usable()) std::printf("--multi: this CPU has no AVX2, FMA or F16C\n");
    strata::poly::set_cpu_kernel_options(ko);
    std::printf("path: %s", ko.multi ? "multi" : "plain");
    if (ko.multi) std::printf(", gate/up tasks of %d rows, down tasks of %d rows", ko.gu_rows, ko.down_rows);
    if (warm > 0) std::printf("; the first %d experts of each call read into the caches before it", warm);
    std::printf("\n");
    long long checked = 0, differ = 0;
    std::mt19937 rng(1234);
    std::vector<float> x, out, want;
    for (const int m : token_counts) {
        x.assign((size_t) m * n_embd, 0.f);
        out.assign((size_t) m * n_embd, 0.f);
        want.assign((size_t) m * n_embd, 0.f);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float& v : x) v = nd(rng);
        double sec = 0.0, bytes = 0.0;
        long long uses = 0, experts = 0;
        for (int w = -2; w < windows; ++w) {   // two windows to warm up
            for (size_t li = 0; li < layers.size(); ++li) {
                const MoeLayer& L = layers[li];
                // the file's layer index is not kept here: the profile's layers are the MoE layers in order
                std::discrete_distribution<int> pick(weights.begin() + (ptrdiff_t) (li * (size_t) n_expert),
                                                     weights.begin() + (ptrdiff_t) ((li + 1) * (size_t) n_expert));
                std::vector<strata::poly::CpuExpert> ce;
                std::vector<int> seen;
                for (int t = 0; t < m; ++t) {
                    std::vector<int> mine;
                    while ((int) mine.size() < 8) {
                        const int e = pick(rng);
                        if (std::find(mine.begin(), mine.end(), e) == mine.end()) mine.push_back(e);
                    }
                    for (const int e : mine) {
                        ce.push_back({L.gate + (size_t) e * L.gu_bytes, L.up + (size_t) e * L.gu_bytes,
                                      L.down + (size_t) e * L.d_bytes, 0.125f, t});
                        if (std::find(seen.begin(), seen.end(), e) == seen.end()) seen.push_back(e);
                    }
                }
                if (check) {   // the plain path's output, then the timed path's below against it
                    strata::poly::CpuKernelOptions plain = ko;
                    plain.multi = false;
                    strata::poly::set_cpu_kernel_options(plain);
                    for (size_t k = 0; k < ce.size(); ++k) ce[k].weight = 0.05f + 0.1f * (float) (k % 8);
                    strata::poly::cpu_experts(workers, ce.data(), (int) ce.size(), m, L.gu_type, L.d_type, n_embd,
                                              n_ff, x.data(), 7.f, false, want.data());
                    strata::poly::set_cpu_kernel_options(ko);
                }
                if (warm > 0) {   // the first `warm` of the call's experts into the caches, untimed
                    std::vector<std::pair<const uint8_t*, size_t>> pieces;
                    for (size_t i = 0; i < seen.size() && (int) i < warm; ++i) {
                        const size_t e = (size_t) seen[i];
                        for (const auto& [p, len] : {std::pair<const uint8_t*, size_t>{L.gate + e * L.gu_bytes, L.gu_bytes},
                                                     {L.up + e * L.gu_bytes, L.gu_bytes},
                                                     {L.down + e * L.d_bytes, L.d_bytes}})
                            for (size_t at = 0; at < len; at += 65536)
                                pieces.emplace_back(p + at, std::min<size_t>(65536, len - at));
                    }
                    std::atomic<uint64_t> sink{0};
                    workers.run((int) pieces.size(), [&](int i) {
                        sink.fetch_xor(read_span(pieces[(size_t) i].first, pieces[(size_t) i].second, 0));
                    });
                    if (sink.load() == 42) std::printf(" ");
                }
                const Clock::time_point t0 = Clock::now();
                strata::poly::cpu_experts(workers, ce.data(), (int) ce.size(), m, L.gu_type, L.d_type, n_embd, n_ff,
                                          x.data(), check ? 7.f : 1e30f, false, out.data());
                const double s = secs(t0);
                if (check) {
                    checked += (long long) out.size();
                    for (size_t k = 0; k < out.size(); ++k) differ += std::memcmp(&out[k], &want[k], 4) != 0;
                }
                if (w < 0) continue;
                sec += s;
                bytes += (double) seen.size() * (double) (2 * L.gu_bytes + L.d_bytes);
                uses += (long long) ce.size();
                experts += (long long) seen.size();
            }
        }
        std::printf("experts, windows of %d tokens: %.2f experts a token-layer distinct (%.1f a layer), %.2f ms a "
                    "layer, %.1f ms a window of %zu layers, %.1f GB/s\n",
                    m, (double) experts / (double) uses * 8.0, (double) experts / (windows * (double) layers.size()),
                    1e3 * sec / (windows * (double) layers.size()), 1e3 * sec / windows, layers.size(),
                    bytes / sec / 1e9);
    }
    if (copied) free_pages(copied);
    if (check) {
        std::printf("check: %lld of %lld output values differ from the plain path's bits\n", differ, checked);
        return differ == 0 ? 0 : 1;
    }
    return 0;
}
