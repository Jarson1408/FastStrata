// src/poly/cpu_experts.cpp - see the header.
#include "cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <numeric>

namespace strata::poly {
namespace {

// Rows of gate and up, and of down, a task.  With fewer tasks than several a thread, the last round left threads
// idle at the end of every layer: 64 rows gave Tiel's 512-row experts 8 tasks each, and its down products 32 tasks.
constexpr int kRowsPerTask = 32;
constexpr int kRowsPerDownTask = 16;
constexpr int kSpinsBeforeSleep = 20000;   // about a millisecond of pause instructions

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

thread_local bool in_task = false;   // this thread is running a task of a batch

// STRATA_CPU_STATS=1: what the CPU's experts read and how long they took, printed at exit, the calls of a few
// tokens (an answer's windows) apart from a prompt's batches.  The bytes are the experts' own, each read once a
// call however many of its tokens use it.
struct CallStats {
    std::atomic<int64_t> calls{0}, uses{0}, experts{0}, bytes{0}, ns{0}, ns_act{0}, ns_gu{0}, ns_hq{0}, ns_down{0};
};
struct Stats {
    bool on = std::getenv("STRATA_CPU_STATS") != nullptr;
    CallStats few, batch;
    ~Stats() {
        if (!on) return;
        for (const auto& [what, s] : {std::pair<const char*, const CallStats*>{"windows", &few}, {"batches", &batch}}) {
            const double sec = 1e-9 * (double) s->ns.load(), gib = (double) s->bytes.load() / (1 << 30);
            if (s->calls.load() == 0) continue;
            std::fprintf(stderr,
                         "strata-poly: CPU experts, %s: %lld calls, %lld uses of %lld experts, %.2f GiB in %.3f s = "
                         "%.1f GB/s; activations %.3f s, gate/up %.3f s, h %.3f s, down %.3f s\n",
                         what, (long long) s->calls.load(), (long long) s->uses.load(), (long long) s->experts.load(),
                         gib, sec, sec > 0 ? (double) s->bytes.load() / sec / 1e9 : 0.0, 1e-9 * (double) s->ns_act.load(),
                         1e-9 * (double) s->ns_gu.load(), 1e-9 * (double) s->ns_hq.load(),
                         1e-9 * (double) s->ns_down.load());
        }
    }
};
Stats g_stats;
constexpr int kFewRows = 16;   // a call of up to this many tokens counts as a window's
using StatClock = std::chrono::steady_clock;
int64_t ns_between(StatClock::time_point a, StatClock::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

CpuKernelOptions g_kernels;

// What cpu_experts() hands its products: the uses sorted by expert (by, run) and the buffers.
struct Products {
    const CpuExpert* experts;
    int n, rows, gu_type, d_type, n_embd, n_ff;
    size_t gu_row, d_row;
    float limit;
    bool gelu;
    const uint8_t* act;   // the tokens' Q8_K inputs, a_bytes apart
    size_t a_bytes;
    uint8_t* hq;          // the uses' Q8_K hidden values, h_bytes apart
    size_t h_bytes;
    const int *by, *run;
    int n_run;
    float* h;             // the uses' hidden values, n_ff apart
    float* out;
};

// CpuKernelOptions::multi: the products of the plain path in cpu_experts() with kq::rows() in place of vec_dot, an
// expert's row read once for all the uses of it, and wider down tasks.  The SwiGLU and the sums are the plain path's
// expressions in the plain path's order, so a token gets the same bits.
void multi_products(Workers& workers, const Products& p, const std::function<void()>* meanwhile,
                    StatClock::time_point* c2, StatClock::time_point* c3) {
    constexpr int kT = kq::kMaxTokens;
    const CpuKernelOptions& o = g_kernels;
    const CpuExpert* experts = p.experts;
    const int n = p.n, n_ff = p.n_ff, n_embd = p.n_embd;
    const float limit = p.limit;
    const int gu_step = std::max(1, o.gu_rows);
    const int gu_tasks = (n_ff + gu_step - 1) / gu_step;
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    workers.run(first + p.n_run * gu_tasks, [&](int t) {
        if (t < first) return (*meanwhile)();
        t -= first;
        const int i0 = p.run[t / gu_tasks], i1 = p.run[t / gu_tasks + 1];
        const CpuExpert& e = experts[p.by[i0]];
        const int r0 = (t % gu_tasks) * gu_step, r1 = std::min(n_ff, r0 + gu_step);
        static thread_local std::vector<float> gu;   // a task's gate and up values: kT uses' rows each
        gu.resize(2 * (size_t) kT * (size_t) gu_step);
        for (int c = i0; c < i1; c += kT) {
            const int nt = std::min(kT, i1 - c);
            const void* a[kT];
            float* g[kT];
            float* u[kT];
            for (int k = 0; k < nt; ++k) {
                a[k] = p.act + (size_t) experts[p.by[c + k]].row * p.a_bytes;
                g[k] = gu.data() + (size_t) k * gu_step;
                u[k] = gu.data() + (size_t) (kT + k) * gu_step;
            }
            kq::rows(p.gu_type, e.gate, p.gu_row, n_embd, a, nt, g, r0, r1);
            kq::rows(p.gu_type, e.up, p.gu_row, n_embd, a, nt, u, r0, r1);
            for (int r = r0; r < r1; ++r) {
                for (int k = 0; k < nt; ++k) {
                    const int j = p.by[c + k];
                    float gv = g[k][r - r0], uv = u[k][r - r0];
                    if (p.gelu) {
                        p.h[(size_t) j * n_ff + r] =
                            0.5f * gv * (1.f + std::tanh(0.7978845608f * gv * (1.f + 0.044715f * gv * gv))) * uv;
                        continue;
                    }
                    gv = std::min(gv, limit);
                    uv = std::clamp(uv, -limit, limit);
                    p.h[(size_t) j * n_ff + r] = gv / (1.f + std::exp(-gv)) * uv;
                }
            }
        }
    });
    if (c2) *c2 = StatClock::now();
    const auto hq_from = traits(GGML_TYPE_Q8_K)->from_float;
    workers.run(n, [&](int j) { hq_from(p.h + (size_t) j * n_ff, p.hq + (size_t) j * p.h_bytes, n_ff); });
    if (c3) *c3 = StatClock::now();

    const int d_step = std::max(1, o.down_rows);
    const int d_tasks = (n_embd + d_step - 1) / d_step;
    workers.run(d_tasks, [&](int t) {
        static thread_local std::vector<float> sv;   // the task's output rows of every use
        sv.resize((size_t) n * (size_t) d_step);
        const int r0 = t * d_step, r1 = std::min(n_embd, r0 + d_step);
        for (int u = 0; u < p.n_run; ++u) {
            const uint8_t* down = experts[p.by[p.run[u]]].down;
            for (int c = p.run[u]; c < p.run[u + 1]; c += kT) {
                const int nt = std::min(kT, p.run[u + 1] - c);
                const void* a[kT];
                float* s[kT];
                for (int k = 0; k < nt; ++k) {
                    a[k] = p.hq + (size_t) p.by[c + k] * p.h_bytes;
                    s[k] = sv.data() + (size_t) p.by[c + k] * d_step;
                }
                kq::rows(p.d_type, down, p.d_row, n_ff, a, nt, s, r0, r1);
            }
        }
        // a token's sum in the order its experts were given: the order it has in a call of its own
        const float* s = sv.data();
        for (int r = r0; r < r1; ++r) {
            int j = 0;
            for (int row = 0; row < p.rows; ++row) {
                float acc = 0.f;
                for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * s[(size_t) j * d_step + (r - r0)];
                p.out[(size_t) row * n_embd + r] = acc;
            }
        }
    });
}

}  // namespace

void set_cpu_kernel_options(const CpuKernelOptions& o) { g_kernels = o; }

Workers::Workers(int threads) {
    for (int i = 1; i < threads; ++i) th_.emplace_back([this] { loop(); });
}

Workers::~Workers() {
    quit_.store(true);
    posted_.fetch_add(1);
    posted_.notify_all();
    for (auto& t : th_) t.join();
}

// Whether it ran a task.  next is read before it is counted up, so it stops near n however often a worker comes
// back to a batch whose tasks are taken.
bool Workers::drain(Batch& b) {
    bool ran = false;
    while (b.next.load(std::memory_order_relaxed) < b.n) {
        const int i = b.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= b.n) break;
        in_task = true;
        (*b.fn)(i);
        in_task = false;
        b.left.fetch_sub(1, std::memory_order_release);
        ran = true;
    }
    return ran;
}

// A worker with nothing to do only reads posted_: a write by each of them, every turn of the loop, kept run() from
// ever seeing users_ at zero for long.  When a batch has been posted since the worker last looked, it counts itself
// in users_ and reads cur_ again before it uses it, so run() can tell when nobody holds its Batch any more.
//
// A batch stays posted after its last task is taken, until run() takes it down, and run() cannot while it has no
// CPU.  That happens with more threads than the system runs: a laptop that parks cores under load makes that of
// any count.  So a worker that found no task in what is posted leaves it alone: it pauses, then sleeps, until
// posted_ moves.  Going round the finished batch instead, the workers kept run() off the CPU for the rest of
// their time slices, every batch.
void Workers::loop() {
    int idle = 0;
    uint32_t seen = posted_.load() - 1;   // the count at which this worker last found nothing to do
    while (!quit_.load(std::memory_order_relaxed)) {
        const uint32_t now = posted_.load(std::memory_order_acquire);
        if (now != seen) {
            // cur_ is set before posted_ counts: the batch of `now` is there, or a later one, or it is done
            bool ran = false;
            if (cur_.load(std::memory_order_acquire)) {
                users_.fetch_add(1);
                Batch* b = cur_.load();
                if (b) ran = drain(*b);
                users_.fetch_sub(1);
            }
            if (ran) {
                idle = 0;
                continue;
            }
            seen = now;
        }
        if (++idle < kSpinsBeforeSleep) {
            _mm_pause();
            continue;
        }
        // run() counts posted_ up before it reads sleepers_, and wait() returns at once when posted_ has moved
        sleepers_.fetch_add(1);
        if (!quit_.load()) posted_.wait(seen);
        sleepers_.fetch_sub(1);
        idle = 0;
    }
}

void Workers::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    // One task is nothing to share out.  And a task that shares out work of its own does it alone: there is one
    // batch at a time, and a worker that waited below for the others to let go of theirs would wait for itself.
    if (n == 1 || in_task) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    const std::lock_guard<std::mutex> one(one_);
    Batch b;
    b.fn = &fn;
    b.n = n;
    b.left.store(n);
    cur_.store(&b);
    posted_.fetch_add(1);
    if (sleepers_.load() > 0) posted_.notify_all();
    drain(b);
    cur_.store(nullptr);   // every task is taken: a worker that comes now has nothing to find
    while (b.left.load(std::memory_order_acquire) != 0) _mm_pause();
    while (users_.load() != 0) _mm_pause();
}

// An MXFP4 value is kvalues_mxfp4[nibble] times the block's power of two halved; Q5_0's is (q - 16) times its half.
bool mxfp4_block_to_q5_0(const uint8_t* src, uint8_t* dst) {
    constexpr int kLow = -24, kHigh = 15;   // the powers of two a half holds, subnormals included
    static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    static const auto half = [] {
        std::array<uint16_t, kHigh - kLow + 1> h{};
        for (int p = kLow; p <= kHigh; ++p) h[(size_t) (p - kLow)] = ggml_fp32_to_fp16(std::ldexp(1.0f, p));
        return h;
    }();
    const int p = (int) src[0] - 128;
    if (p < kLow || p > kHigh) return false;
    std::memcpy(dst, &half[(size_t) (p - kLow)], 2);
    uint32_t qh = 0;
    for (int j = 0; j < 16; ++j) {   // nibble j's low half is value j, its high half value j + 16, in both types
        const int q0 = kv[src[1 + j] & 15] + 16, q1 = kv[src[1 + j] >> 4] + 16;
        dst[6 + j] = (uint8_t) ((q0 & 15) | (q1 & 15) << 4);
        qh |= (uint32_t) (q0 >> 4) << j | (uint32_t) (q1 >> 4) << (j + 16);
    }
    std::memcpy(dst + 2, &qh, 4);
    return true;
}

bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err) {
    init_once();
    for (const int type : {gu_type, d_type}) {
        const ggml_type_traits_cpu* t = traits(type);
        const ggml_type_traits_cpu* a = t && t->vec_dot ? traits(t->vec_dot_type) : nullptr;
        if (a == nullptr || a->from_float == nullptr) {
            err = "ggml-cpu has no dot product for tensor type " + std::to_string(type);
            return false;
        }
        const int n = type == gu_type ? n_embd : n_ff;
        if (n % ggml_blck_size((ggml_type) type) || n % ggml_blck_size(t->vec_dot_type)) {
            err = "a row of " + std::to_string(n) + " values is not whole blocks of tensor type " + std::to_string(type);
            return false;
        }
    }
    return true;
}

void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int d_type, int n_embd,
                 int n_ff, const float* x, float limit, bool gelu, float* out,
                 const std::function<void()>* meanwhile) {
    init_once();
    const bool timed = g_stats.on;
    StatClock::time_point c0, c1, c2, c3, c4;
    if (timed) c0 = StatClock::now();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    const size_t gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    const size_t d_row = ggml_row_size((ggml_type) d_type, n_ff);
    const size_t a_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, n_ff);

    // The workers get these through the pointers below: a thread_local named in their code would be their own.
    static thread_local std::vector<uint8_t> act, hq;
    static thread_local std::vector<float> h;
    static thread_local std::vector<int> by, run;
    act.resize((size_t) rows * a_bytes);
    hq.resize((size_t) n * h_bytes);
    h.resize((size_t) n * n_ff);
    // ggml's quantizers for the activations are scalar: microseconds a row, which on one thread kept the others
    // waiting, here and before the down products
    uint8_t* actp = act.data();
    const auto act_from = traits(tg->vec_dot_type)->from_float;
    workers.run(rows, [&](int t) { act_from(x + (size_t) t * n_embd, actp + (size_t) t * a_bytes, n_embd); });

    // The uses of one expert next to each other (by), and where each expert's run of them starts (run).
    by.resize((size_t) n);
    std::iota(by.begin(), by.end(), 0);
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return experts[a].gate < experts[b].gate; });
    run.clear();
    for (int i = 0; i < n; ++i)
        if (i == 0 || experts[by[(size_t) i]].gate != experts[by[(size_t) i - 1]].gate) run.push_back(i);
    const int n_run = (int) run.size();
    run.push_back(n);

    float* hp = h.data();
    const int* byp = by.data();
    const int* runp = run.data();
    if (timed) c1 = StatClock::now();
    const auto record = [&] {
        c4 = StatClock::now();
        CallStats& s = rows <= kFewRows ? g_stats.few : g_stats.batch;
        s.calls += 1;
        s.uses += n;
        s.experts += n_run;
        s.bytes += (int64_t) n_run * (int64_t) (2 * gu_row * (size_t) n_ff + d_row * (size_t) n_embd);
        s.ns += ns_between(c0, c4);
        s.ns_act += ns_between(c0, c1);
        s.ns_gu += ns_between(c1, c2);
        s.ns_hq += ns_between(c2, c3);
        s.ns_down += ns_between(c3, c4);
    };
    if (g_kernels.multi && kq::usable() && kq::has(gu_type) && kq::has(d_type) &&
        tg->vec_dot_type == GGML_TYPE_Q8_K && td->vec_dot_type == GGML_TYPE_Q8_K) {
        Products p{experts, n, rows, gu_type, d_type, n_embd, n_ff, gu_row, d_row, limit, gelu, actp, a_bytes,
                   hq.data(), h_bytes, byp, runp, n_run, h.data(), out};
        multi_products(workers, p, meanwhile, timed ? &c2 : nullptr, timed ? &c3 : nullptr);
        if (timed) record();
        return;
    }
    const int gu_tasks = (n_ff + kRowsPerTask - 1) / kRowsPerTask;
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    workers.run(first + n_run * gu_tasks, [&](int t) {
        if (t < first) return (*meanwhile)();
        t -= first;
        const int i0 = runp[t / gu_tasks], i1 = runp[t / gu_tasks + 1];
        const CpuExpert& e = experts[byp[i0]];
        const int r0 = (t % gu_tasks) * kRowsPerTask, r1 = std::min(n_ff, r0 + kRowsPerTask);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* gate = e.gate + (size_t) r * gu_row;
            const uint8_t* up = e.up + (size_t) r * gu_row;
            for (int i = i0; i < i1; ++i) {
                const int j = byp[i];
                const uint8_t* a = actp + (size_t) experts[j].row * a_bytes;
                float g = 0.f, u = 0.f;
                tg->vec_dot(n_embd, &g, 0, gate, 0, a, 0, 1);
                tg->vec_dot(n_embd, &u, 0, up, 0, a, 0, 1);
                if (gelu) {
                    hp[(size_t) j * n_ff + r] =
                        0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
                    continue;
                }
                g = std::min(g, limit);
                u = std::clamp(u, -limit, limit);
                hp[(size_t) j * n_ff + r] = g / (1.f + std::exp(-g)) * u;
            }
        }
    });
    if (timed) c2 = StatClock::now();
    uint8_t* hqp = hq.data();
    const auto hq_from = traits(td->vec_dot_type)->from_float;
    workers.run(n, [&](int j) { hq_from(hp + (size_t) j * n_ff, hqp + (size_t) j * h_bytes, n_ff); });
    if (timed) c3 = StatClock::now();

    const int d_tasks = (n_embd + kRowsPerDownTask - 1) / kRowsPerDownTask;
    workers.run(d_tasks, [&](int t) {
        static thread_local std::vector<float> one;   // an output value of every use
        one.resize((size_t) n);
        float* s = one.data();
        const int r0 = t * kRowsPerDownTask, r1 = std::min(n_embd, r0 + kRowsPerDownTask);
        for (int r = r0; r < r1; ++r) {
            for (int u = 0; u < n_run; ++u) {
                const uint8_t* down = experts[byp[runp[u]]].down + (size_t) r * d_row;
                for (int i = runp[u]; i < runp[u + 1]; ++i)
                    td->vec_dot(n_ff, s + byp[i], 0, down, 0, hqp + (size_t) byp[i] * h_bytes, 0, 1);
            }
            // a token's sum in the order its experts were given: the order it has in a call of its own
            int j = 0;
            for (int row = 0; row < rows; ++row) {
                float acc = 0.f;
                for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * s[j];
                out[(size_t) row * n_embd + r] = acc;
            }
        }
    });
    if (timed) record();
}

}  // namespace strata::poly
