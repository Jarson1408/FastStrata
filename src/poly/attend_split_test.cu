// src/poly/attend_split_test.cu - attend() and attend_q8() with --attn-split against the other kernels and against
// the same attention in doubles on the host, on random caches in Qwen3.6's shape (16 heads of 256 values, 2 groups)
// and one of 128 values; a token's result with split alone against the same token beside others, and with the
// position on the card against it on the host, bit for bit; then the time a call takes both ways.
//
// usage: poly_attend_split_test [--time]   (exit 0: every result close to the doubles and every repeat the same)
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace strata::poly;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// a value from the hash of i, about normal: a sum of four uniforms, times `scale`
__device__ __host__ inline float noise(unsigned long long i, unsigned seed, float scale) {
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) {
        unsigned long long x = i * 4 + k + 0x9E3779B97F4A7C15ull * (seed + 1);
        x ^= x >> 33, x *= 0xff51afd7ed558ccdull, x ^= x >> 33, x *= 0xc4ceb9fe1a85ec53ull, x ^= x >> 33;
        s += float(x >> 40) / float(1 << 24) - 0.5f;
    }
    return s * 1.7320508f * scale;
}

__global__ void fill_kernel(float* x, size_t n, unsigned long long first, unsigned seed, float scale) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = noise(first + i, seed, scale);
}

struct Shape {
    int dim, n_head, kv_heads;
};

struct Caches {
    Shape sh;
    bool q8;
    int rows;
    void* k = nullptr;
    void* v = nullptr;
    std::vector<float> hk, hv;   // the values as the kernels read them, for the host's sums
};

// rows of keys and values from the noise, stored as the engine stores them, and read back as the kernels read them
Caches make_caches(Shape sh, bool q8, int rows, bool host_copy) {
    Caches c{sh, q8, rows};
    const size_t width = size_t(sh.kv_heads) * sh.dim, row_bytes = q8 ? width + width / 16 : width * 2;
    ck(cudaMalloc(&c.k, row_bytes * rows), "cache");
    ck(cudaMalloc(&c.v, row_bytes * rows), "cache");
    const int batch = 4096;
    float* x = nullptr;
    ck(cudaMalloc(&x, sizeof(float) * width * batch), "batch");
    for (int which = 0; which < 2; ++which) {
        void* cache = which ? c.v : c.k;
        for (int r0 = 0; r0 < rows; r0 += batch) {
            const int n = std::min(batch, rows - r0);
            const size_t m = width * n;
            fill_kernel<<<unsigned((m + 255) / 256), 256>>>(x, m, size_t(r0) * width, 11 + which, 1.0f);
            if (q8) store_q8(x, (uint8_t*) cache, (int) width, r0, nullptr, n, nullptr);
            else store_f16(x, (uint16_t*) cache, (int) width, r0, n, nullptr);
        }
    }
    ck(cudaDeviceSynchronize(), "fill");
    cudaFree(x);
    if (!host_copy) return c;
    for (int which = 0; which < 2; ++which) {
        std::vector<uint8_t> raw(row_bytes * rows);
        ck(cudaMemcpy(raw.data(), which ? c.v : c.k, raw.size(), cudaMemcpyDeviceToHost), "copy");
        std::vector<float>& h = which ? c.hv : c.hk;
        h.resize(width * rows);
        for (int r = 0; r < rows; ++r) {
            const uint8_t* src = raw.data() + row_bytes * r;
            for (size_t i = 0; i < width; ++i) {
                uint16_t bits;
                float val;
                if (q8) {
                    std::memcpy(&bits, src + width + 2 * (i / 32), 2);
                    __half hb;
                    std::memcpy(&hb, &bits, 2);
                    val = __half2float(hb) * float((signed char) src[i]);
                } else {
                    std::memcpy(&bits, src + 2 * i, 2);
                    __half hb;
                    std::memcpy(&hb, &bits, 2);
                    val = __half2float(hb);
                }
                h[size_t(r) * width + i] = val;
            }
        }
    }
    return c;
}

// token t of a call from pos, head h, in doubles
void host_attend(const Caches& c, const std::vector<float>& q, int64_t pos, int t, int h, std::vector<double>& out) {
    const Shape sh = c.sh;
    const size_t width = size_t(sh.kv_heads) * sh.dim;
    const int g = h / (sh.n_head / sh.kv_heads);
    const float* qr = q.data() + (size_t(t) * sh.n_head + h) * sh.dim;
    const int64_t n = pos + t + 1;
    std::vector<double> s(n);
    double mx = -1e300;
    for (int64_t r = 0; r < n; ++r) {
        double acc = 0.0;
        const float* kr = c.hk.data() + size_t(r) * width + size_t(g) * sh.dim;
        for (int d = 0; d < sh.dim; ++d) acc += double(qr[d]) * kr[d];
        s[r] = acc / std::sqrt(double(sh.dim));
        mx = std::max(mx, s[r]);
    }
    double sum = 0.0;
    for (int64_t r = 0; r < n; ++r) sum += s[r] = std::exp(s[r] - mx);
    out.assign(sh.dim, 0.0);
    for (int64_t r = 0; r < n; ++r) {
        const float* vr = c.hv.data() + size_t(r) * width + size_t(g) * sh.dim;
        for (int d = 0; d < sh.dim; ++d) out[d] += s[r] / sum * vr[d];
    }
}

struct Run {
    Caches* c;
    float* q;          // rows * n_head * dim
    float* score;
    size_t score_floats;
    int chunk;
    float* out;
    long long* at;     // on the card
};

void call(const Run& r, int64_t pos, int rows, bool split, bool on_card, int room, cudaStream_t st = nullptr,
          bool set_at = true) {
    const Shape sh = r.c->sh;
    const float scale = 1.0f / std::sqrt(float(sh.dim));
    const long long* at = on_card ? r.at : nullptr;
    if (on_card && set_at) {
        const long long p = pos;
        ck(cudaMemcpyAsync(r.at, &p, sizeof(p), cudaMemcpyHostToDevice, st), "at");
        at = r.at;
    }
    if (r.c->q8)
        attend_q8(r.q, (const uint8_t*) r.c->k, (const uint8_t*) r.c->v, pos, rows, r.score, r.chunk, r.out, sh.dim,
                  sh.n_head, sh.kv_heads, scale, st, 0, 0, false, at, on_card ? room : 0, split, r.score_floats);
    else
        attend(r.q, (const uint16_t*) r.c->k, (const uint16_t*) r.c->v, pos, rows, r.score, r.chunk, r.out, sh.dim,
               sh.n_head, sh.kv_heads, scale, st, 0, 0, false, at, on_card ? room : 0, split, r.score_floats);
}

int room_of(int64_t last, int most) {
    int64_t r = 1024;
    while (r < last) r *= 2;
    return (int) std::min<int64_t>(r, most);
}

std::vector<float> fetch(const float* p, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost), "fetch");
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    const bool timing = argc > 1 && std::string(argv[1]) == "--time";
    const int max_rows = 147456, chunk = 7, most_tokens = 8;
    // --one POS TOKENS: that call alone, ten times each way, for a profiler
    const bool one = argc > 3 && std::string(argv[1]) == "--one";
    if (one) {
        const int64_t pos = std::atoll(argv[2]);
        const int t_n = std::atoi(argv[3]);
        const Shape sh{256, 16, 2};
        Caches c = make_caches(sh, true, (int) std::min<int64_t>(max_rows, pos + t_n + 1024), false);
        const size_t qn = size_t(most_tokens) * sh.n_head * sh.dim, score_floats = size_t(chunk) * sh.n_head * c.rows;
        float *q = nullptr, *score = nullptr, *out = nullptr;
        long long* at = nullptr;
        ck(cudaMalloc(&q, qn * sizeof(float)), "q");
        ck(cudaMalloc(&score, score_floats * sizeof(float)), "score");
        ck(cudaMalloc(&out, qn * sizeof(float)), "out");
        ck(cudaMalloc(&at, sizeof(long long)), "at");
        fill_kernel<<<unsigned((qn + 255) / 256), 256>>>(q, qn, 0, 7, 2.5f);
        Run r{&c, q, score, score_floats, chunk, out, at};
        const int room = room_of(pos + t_n, c.rows);
        for (int split = 0; split < 2; ++split)
            for (int i = 0; i < 10; ++i) call(r, pos, t_n, split, true, room);
        ck(cudaDeviceSynchronize(), "one");
        std::printf("done\n");
        return 0;
    }
    int bad = 0;
    struct Case {
        Shape sh;
        bool q8;
        int rows;
    };
    const Case cases[] = {{{256, 16, 2}, true, timing ? max_rows : 40000},
                          {{256, 16, 2}, false, 9000},
                          {{128, 16, 4}, true, 9000}};
    for (const Case& cs : cases) {
        const Shape sh = cs.sh;
        Caches c = make_caches(sh, cs.q8, cs.rows, !timing || cs.rows <= 40000);
        const size_t qn = size_t(most_tokens) * sh.n_head * sh.dim;
        float *q = nullptr, *score = nullptr, *out_a = nullptr, *out_b = nullptr;
        long long* at = nullptr;
        const size_t score_floats = size_t(chunk) * sh.n_head * cs.rows;
        ck(cudaMalloc(&q, qn * sizeof(float)), "q");
        ck(cudaMalloc(&score, score_floats * sizeof(float)), "score");
        ck(cudaMalloc(&out_a, qn * sizeof(float)), "out");
        ck(cudaMalloc(&out_b, qn * sizeof(float)), "out");
        ck(cudaMalloc(&at, sizeof(long long)), "at");
        // queries of two sizes: the larger give a softmax with a few rows well above the rest
        fill_kernel<<<unsigned((qn + 255) / 256), 256>>>(q, qn, 0, 7, 2.5f);
        std::vector<float> hq = fetch(q, qn);
        Run ra{&c, q, score, score_floats, chunk, out_a, at}, rb{&c, q, score, score_floats, chunk, out_b, at};
        std::printf("== %s cache, %d heads of %d values in %d groups, %d rows\n", cs.q8 ? "q8_0" : "fp16", sh.n_head,
                    sh.dim, sh.kv_heads, cs.rows);
        if (!timing) {
            const int64_t positions[] = {0, 5, 255, 256, 300, 4100, cs.rows - 9};
            for (const int64_t pos : positions)
                for (const int t_n : {1, 3, 8}) {
                    if (pos + t_n > cs.rows) continue;
                    const int room = room_of(pos + t_n, cs.rows);
                    call(ra, pos, t_n, false, false, 0);
                    call(rb, pos, t_n, true, true, room);
                    ck(cudaDeviceSynchronize(), "attend");
                    const std::vector<float> a = fetch(out_a, size_t(t_n) * sh.n_head * sh.dim);
                    const std::vector<float> b = fetch(out_b, a.size());
                    // the doubles for the first and last token, every head
                    double ea = 0.0, eb = 0.0, top = 0.0, eab = 0.0;
                    for (const int t : {0, t_n - 1})
                        for (int h = 0; h < sh.n_head; ++h) {
                            std::vector<double> ref;
                            host_attend(c, hq, pos, t, h, ref);
                            for (int d = 0; d < sh.dim; ++d) {
                                const size_t i = (size_t(t) * sh.n_head + h) * sh.dim + d;
                                top = std::max(top, std::fabs(ref[d]));
                                ea = std::max(ea, std::fabs(a[i] - ref[d]));
                                eb = std::max(eb, std::fabs(b[i] - ref[d]));
                                eab = std::max(eab, (double) std::fabs(a[i] - b[i]));
                            }
                        }
                    // split alone at each token's position, and with the position on the host: the same bits
                    int moved = 0;
                    for (int t = 0; t < t_n; ++t) {
                        Run alone = ra;
                        alone.q = q + size_t(t) * sh.n_head * sh.dim;   // token t's queries
                        call(alone, pos + t, 1, true, false, 0);
                        ck(cudaDeviceSynchronize(), "attend");
                        const std::vector<float> one = fetch(out_a, size_t(sh.n_head) * sh.dim);
                        if (std::memcmp(one.data(), b.data() + size_t(t) * sh.n_head * sh.dim,
                                        one.size() * sizeof(float)))
                            ++moved;
                    }
                    const bool close = eb <= 2e-3 * top + 1e-6 && eb <= 4.0 * ea + 2e-3 * top;
                    std::printf("pos %6lld tokens %d: max |old - doubles| %.2e, |split - doubles| %.2e, "
                                "|old - split| %.2e of %.3f; split alone differs for %d of %d tokens%s\n",
                                (long long) pos, t_n, ea, eb, eab, top, moved, t_n,
                                close && !moved ? "" : "   <-- FAIL");
                    if (!close || moved) ++bad;
                }
        } else {
            cudaStream_t st;
            ck(cudaStreamCreate(&st), "stream");
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            const int64_t positions[] = {2048, 37433, 100000, 147000};
            for (const int64_t pos : positions)
                for (const int t_n : {1, 2, 4, 8}) {
                    if (pos + t_n > cs.rows) continue;
                    const int room = room_of(pos + t_n, cs.rows);
                    double ms[2];
                    for (int split = 0; split < 2; ++split) {
                        // as a recorded layer runs: the position on the card, launches sized for the room
                        const Run& r = split ? rb : ra;
                        for (int i = 0; i < 3; ++i) call(r, pos, t_n, split, true, room, st);
                        std::vector<float> each;
                        for (int i = 0; i < 20; ++i) {
                            cudaEventRecord(e0, st);
                            call(r, pos, t_n, split, true, room, st, false);   // the position is there already
                            cudaEventRecord(e1, st);
                            cudaEventSynchronize(e1);
                            float t = 0;
                            cudaEventElapsedTime(&t, e0, e1);
                            each.push_back(t);
                        }
                        std::sort(each.begin(), each.end());
                        ms[split] = each[each.size() / 2];
                    }
                    const size_t width = size_t(sh.kv_heads) * sh.dim;
                    const double bytes = 2.0 * double(pos + t_n) * (cs.q8 ? width + width / 16 : 2 * width);
                    std::printf("pos %6lld tokens %d room %6d: old %8.1f us, split %7.1f us (%5.1fx; split reads the "
                                "caches at %5.1f GB/s)\n",
                                (long long) pos, t_n, room, 1000 * ms[0], 1000 * ms[1], ms[0] / ms[1],
                                bytes / (ms[1] * 1e6));
                }
        }
        cudaFree(q), cudaFree(score), cudaFree(out_a), cudaFree(out_b), cudaFree(at), cudaFree(c.k), cudaFree(c.v);
        if (timing) break;   // the time of Qwen3.6's shape only
    }
    std::printf(bad ? "%d cases FAILED\n" : "all cases passed\n", bad);
    return bad ? 1 : 0;
}
