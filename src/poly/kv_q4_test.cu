// src/poly/kv_q4_test.cu - the --kv-q4 caches: store_q4() against ggml's Q4_0 arithmetic on the host, code for code
// and scale for scale; then attend_q4() (the other kernels and --attn-split) and attend_batch_q4() for both layouts
// (keys Q4_0, keys Q8_0) against the same attention in doubles on the host over the values the rows give back, on
// random caches in Qwen3.6's shape (16 heads of 256 values, 2 groups) and one of 128 values; a token's result with
// split alone against the same token beside others, and with the position on the card against it on the host, bit
// for bit; then the time a call takes, beside the q8 cache's.
//
// usage: poly_kv_q4_test [--time]   (exit 0: every code the host's, every result close to the doubles and every
//                                    repeat the same)
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
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

float half_bits(const uint8_t* p) {
    __half h;
    std::memcpy(&h, p, 2);
    return __half2float(h);
}

// where value k of a group of 8 lies in its 32-bit word (store_q4())
int nibble(int k) { return k / 2 + 4 * (k % 2); }

// ggml's quantize_row_q4_0 for one block of 32: the scale's float and the codes
float host_q4(const float* x, int* code) {
    float amax = 0.0f, mx = 0.0f;
    for (int j = 0; j < 32; ++j)
        if (amax < std::fabs(x[j])) amax = std::fabs(x[j]), mx = x[j];
    const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    for (int j = 0; j < 32; ++j) {
        volatile float t = x[j] * id;   // ggml's two roundings, not one fused
        code[j] = std::min(15, (int) (t + 8.5f));
    }
    return d;
}

// A row's values as the kernels read them back.  bits: 4 (Q4_0), 8 (Q8_0), 16 (fp16).
float value_of(const uint8_t* row, size_t width, int bits, size_t i) {
    if (bits == 16) return half_bits(row + 2 * i);
    if (bits == 8) return half_bits(row + width + 2 * (i / 32)) * float((signed char) row[i]);
    const int nib = nibble(int(i % 8));
    const int code = row[i / 8 * 4 + nib / 2] >> 4 * (nib % 2) & 15;
    return half_bits(row + width / 2 + 2 * (i / 32)) * float(code - 8);
}

size_t row_bytes(size_t width, int bits) {
    return bits == 4 ? width / 2 + width / 16 : bits == 8 ? width + width / 16 : 2 * width;
}

// store_q4() on rows of noise with blocks made to be awkward (zeros, a tie of + and - at the largest, one large
// value), against host_q4(): every code and every scale's bits the same.  Returns the blocks that differ.
int check_store() {
    const int width = 512, rows = 300;
    std::vector<float> h(size_t(width) * rows);
    for (size_t i = 0; i < h.size(); ++i) h[i] = noise(i, 3, 1.0f + float(i / width % 7));
    for (int j = 0; j < 32; ++j) h[j] = 0.0f;                    // a block of zeros
    h[32] = 2.5f, h[40] = -2.5f;                                 // a tie: the first is the scale's
    h[64 + 7] = -40.0f;                                          // one value far out
    h[96] = 3.0f, h[97] = -3.0f, h[98] = 2.99999f;               // + first
    float* x = nullptr;
    uint8_t* cache = nullptr;
    const size_t rb = row_bytes(width, 4);
    ck(cudaMalloc(&x, h.size() * sizeof(float)), "x");
    ck(cudaMalloc(&cache, rb * (rows + 10)), "cache");
    ck(cudaMemcpy(x, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "x");
    long long* at = nullptr;
    ck(cudaMalloc(&at, sizeof(long long)), "at");
    const long long later = rows / 2 + 10;   // the second half's first row
    ck(cudaMemcpy(at, &later, sizeof(later), cudaMemcpyHostToDevice), "at");
    store_q4(x, cache, width, 0, nullptr, rows / 2, nullptr);                          // rows 0 to 149 at pos
    store_q4(x + size_t(width) * (rows / 2), cache, width, 999, at, rows - rows / 2, nullptr);   // the rest at *at
    ck(cudaDeviceSynchronize(), "store");
    std::vector<uint8_t> got(rb * (rows + 10));
    ck(cudaMemcpy(got.data(), cache, got.size(), cudaMemcpyDeviceToHost), "copy");
    int bad = 0;
    double err2 = 0.0, all2 = 0.0;
    for (int r = 0; r < rows; ++r) {
        const int crow = r < rows / 2 ? r : r + 10;
        const uint8_t* row = got.data() + rb * crow;
        for (int b = 0; b < width / 32; ++b) {
            int code[32];
            const float* xb = h.data() + size_t(r) * width + 32 * b;
            const float d = host_q4(xb, code);
            const __half hd = __float2half(d);
            bool same = std::memcmp(&hd, row + width / 2 + 2 * b, 2) == 0;
            for (int j = 0; j < 32; ++j) {
                const int c = 32 * b + j, nib = nibble(c % 8);
                if ((row[c / 8 * 4 + nib / 2] >> 4 * (nib % 2) & 15) != code[j]) same = false;
                const double back = value_of(row, width, 4, size_t(c));
                err2 += (back - xb[j]) * (back - xb[j]);
                all2 += double(xb[j]) * xb[j];
            }
            if (!same) ++bad;
        }
    }
    std::printf("store_q4: %d of %d blocks differ from ggml's Q4_0 on the host; the values read back are %.1f%% off "
                "(RMS)%s\n",
                bad, rows * width / 32, 100.0 * std::sqrt(err2 / all2), bad ? "   <-- FAIL" : "");
    cudaFree(x), cudaFree(cache), cudaFree(at);
    return bad;
}

struct Shape {
    int dim, n_head, kv_heads;
};

struct Caches {
    Shape sh;
    int kbits, vbits;   // 4, 8 or 16
    int rows;
    void* k = nullptr;
    void* v = nullptr;
    std::vector<float> hk, hv;   // the values as the kernels read them, for the host's sums
};

// rows of keys and values from the noise, stored as the engine stores them, and read back as the kernels read them
Caches make_caches(Shape sh, int kbits, int vbits, int rows, bool host_copy) {
    Caches c{sh, kbits, vbits, rows};
    const size_t width = size_t(sh.kv_heads) * sh.dim;
    ck(cudaMalloc(&c.k, row_bytes(width, kbits) * rows), "cache");
    ck(cudaMalloc(&c.v, row_bytes(width, vbits) * rows), "cache");
    const int batch = 4096;
    float* x = nullptr;
    ck(cudaMalloc(&x, sizeof(float) * width * batch), "batch");
    for (int which = 0; which < 2; ++which) {
        void* cache = which ? c.v : c.k;
        const int bits = which ? vbits : kbits;
        for (int r0 = 0; r0 < rows; r0 += batch) {
            const int n = std::min(batch, rows - r0);
            const size_t m = width * n;
            fill_kernel<<<unsigned((m + 255) / 256), 256>>>(x, m, size_t(r0) * width, 11 + which, 1.0f);
            if (bits == 4) store_q4(x, (uint8_t*) cache, (int) width, r0, nullptr, n, nullptr);
            else if (bits == 8) store_q8(x, (uint8_t*) cache, (int) width, r0, nullptr, n, nullptr);
            else store_f16(x, (uint16_t*) cache, (int) width, r0, n, nullptr);
        }
    }
    ck(cudaDeviceSynchronize(), "fill");
    cudaFree(x);
    if (!host_copy) return c;
    for (int which = 0; which < 2; ++which) {
        const int bits = which ? vbits : kbits;
        const size_t rb = row_bytes(width, bits);
        std::vector<uint8_t> raw(rb * rows);
        ck(cudaMemcpy(raw.data(), which ? c.v : c.k, raw.size(), cudaMemcpyDeviceToHost), "copy");
        std::vector<float>& h = which ? c.hv : c.hk;
        h.resize(width * rows);
        for (int r = 0; r < rows; ++r)
            for (size_t i = 0; i < width; ++i) h[size_t(r) * width + i] = value_of(raw.data() + rb * r, width, bits, i);
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
    }
    if (r.c->vbits == 4)
        attend_q4(r.q, (const uint8_t*) r.c->k, (const uint8_t*) r.c->v, r.c->kbits == 8, pos, rows, r.score, r.chunk,
                  r.out, sh.dim, sh.n_head, sh.kv_heads, scale, st, 0, 0, false, at, on_card ? room : 0, split,
                  r.score_floats);
    else
        attend_q8(r.q, (const uint8_t*) r.c->k, (const uint8_t*) r.c->v, pos, rows, r.score, r.chunk, r.out, sh.dim,
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

const char* name_of(int kbits, int vbits) {
    return vbits == 8 ? "q8_0" : kbits == 8 ? "keys q8_0, values q4_0" : "keys and values q4_0";
}

// The largest difference from the doubles of the first and last token's every head, and the largest value.
void against_doubles(const Caches& c, const std::vector<float>& hq, const std::vector<float>& got, int64_t pos,
                     int t_n, double* err, double* top) {
    const Shape sh = c.sh;
    *err = *top = 0.0;
    for (const int t : {0, t_n - 1})
        for (int h = 0; h < sh.n_head; ++h) {
            std::vector<double> ref;
            host_attend(c, hq, pos, t, h, ref);
            for (int d = 0; d < sh.dim; ++d) {
                const size_t i = (size_t(t) * sh.n_head + h) * sh.dim + d;
                *top = std::max(*top, std::fabs(ref[d]));
                *err = std::max(*err, std::fabs(got[i] - ref[d]));
            }
        }
}

}  // namespace

int main(int argc, char** argv) {
    const bool timing = argc > 1 && std::string(argv[1]) == "--time";
    const int max_rows = 147456, chunk = 7, most_tokens = 8;
    int bad = timing ? 0 : int(check_store() > 0);
    struct Case {
        Shape sh;
        int kbits, vbits, rows;
    };
    const Case cases[] = {{{256, 16, 2}, 4, 4, 20000}, {{256, 16, 2}, 8, 4, 20000}, {{128, 16, 4}, 4, 4, 9000},
                          {{128, 16, 4}, 8, 4, 9000}};
    const Case timed[] = {{{256, 16, 2}, 8, 8, max_rows}, {{256, 16, 2}, 8, 4, max_rows}, {{256, 16, 2}, 4, 4, max_rows}};
    for (const Case& cs : timing ? std::vector<Case>(timed, timed + 3) : std::vector<Case>(cases, cases + 4)) {
        const Shape sh = cs.sh;
        Caches c = make_caches(sh, cs.kbits, cs.vbits, cs.rows, !timing);
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
        std::printf("== %s, %d heads of %d values in %d groups, %d rows\n", name_of(cs.kbits, cs.vbits), sh.n_head,
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
                    double ea, eb, top;
                    against_doubles(c, hq, a, pos, t_n, &ea, &top);
                    against_doubles(c, hq, b, pos, t_n, &eb, &top);
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
                    // the other kernels sum floats of the exact values; split takes halves, as in attend_split_test
                    const bool close = ea <= 1e-4 * top + 1e-6 && eb <= 2e-3 * top + 1e-6;
                    std::printf("pos %6lld tokens %d: max |old - doubles| %.2e, |split - doubles| %.2e of %.3f; split "
                                "alone differs for %d of %d tokens%s\n",
                                (long long) pos, t_n, ea, eb, top, moved, t_n, close && !moved ? "" : "   <-- FAIL");
                    if (!close || moved) ++bad;
                }
            // a prompt batch through cuBLAS: from the cache's start, and from further on
            void* blas = blas_create(nullptr);
            const size_t width = size_t(sh.kv_heads) * sh.dim;
            const int deq_rows = 512;
            for (const auto& [pos, t_n] : {std::pair<int64_t, int>{0, 700}, {4100, 64}, {cs.rows - 40, 40}}) {
                float *bq = nullptr, *bout = nullptr, *work = nullptr, *bscore = nullptr;
                uint16_t* deq = nullptr;
                const size_t bqn = size_t(t_n) * sh.n_head * sh.dim, bscore_n = size_t(1) << 21;
                ck(cudaMalloc(&bq, bqn * sizeof(float)), "q");
                ck(cudaMalloc(&bout, bqn * sizeof(float)), "out");
                ck(cudaMalloc(&work, (size_t(t_n) * sh.n_head * (sh.dim + sh.dim / 2 + 3) + 1) * sizeof(float)), "w");
                ck(cudaMalloc(&bscore, bscore_n * sizeof(float)), "score");
                ck(cudaMalloc(&deq, 2 * size_t(deq_rows) * width * sizeof(uint16_t)), "deq");
                fill_kernel<<<unsigned((bqn + 255) / 256), 256>>>(bq, bqn, 99, 7, 2.5f);
                attend_batch_q4(blas, bq, (const uint8_t*) c.k, (const uint8_t*) c.v, cs.kbits == 8, deq, deq_rows,
                                pos, t_n, bscore, bscore_n, work, bout, sh.dim, sh.n_head, sh.kv_heads,
                                1.0f / std::sqrt(float(sh.dim)), nullptr);
                ck(cudaDeviceSynchronize(), "attend_batch");
                const std::vector<float> hbq = fetch(bq, bqn), got = fetch(bout, bqn);
                double e, top;
                against_doubles(c, hbq, got, pos, t_n, &e, &top);
                const bool close = e <= 2e-3 * top + 1e-6;
                std::printf("batch from pos %6lld, %d tokens: max |batch - doubles| %.2e of %.3f%s\n", (long long) pos,
                            t_n, e, top, close ? "" : "   <-- FAIL");
                if (!close) ++bad;
                cudaFree(bq), cudaFree(bout), cudaFree(work), cudaFree(bscore), cudaFree(deq);
            }
            blas_destroy(blas);
        } else {
            cudaStream_t st;
            ck(cudaStreamCreate(&st), "stream");
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            const int64_t positions[] = {2048, 37433, 100000, 139056};
            for (const int64_t pos : positions)
                for (const int t_n : {1, 4, 8}) {
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
                    const double bytes =
                        double(pos + t_n) * double(row_bytes(width, cs.kbits) + row_bytes(width, cs.vbits));
                    std::printf("pos %6lld tokens %d: old %8.1f us, split %7.1f us (reads the caches at %5.1f GB/s)\n",
                                (long long) pos, t_n, 1000 * ms[0], 1000 * ms[1], bytes / (ms[1] * 1e6));
                }
        }
        cudaFree(q), cudaFree(score), cudaFree(out_a), cudaFree(out_b), cudaFree(at), cudaFree(c.k), cudaFree(c.v);
    }
    std::printf(bad ? "%d cases FAILED\n" : "all cases passed\n", bad);
    return bad ? 1 : 0;
}
