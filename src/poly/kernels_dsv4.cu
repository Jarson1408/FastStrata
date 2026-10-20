// src/poly/kernels_dsv4.cu - DeepSeek V4's kernels (kernels.hpp).  The reference for every formula is llama.cpp's
// deepseek4 graph (src/models/deepseek4.cpp) and its compressed caches (src/llama-kv-cache-dsv4.cpp).  Where a kernel
// takes several tokens, blockIdx.y is the token unless it says otherwise.
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int SELECT_THREADS = 1024;
constexpr int ATTEND_THREADS = 256;
constexpr int SCORE_KEYS = 32;      // dsv4_idx_score: the keys a block takes
constexpr int IDX_DIM = 128;        // an indexer key's values
constexpr int GROUP_HEADS = 8;      // attend_rows_kernel: the heads of a token a block takes, a warp each
constexpr int GROUP_KEYS = WARP;    // the keys it holds in shared memory at a time, a lane each
constexpr int GROUP_ROWS = 32;      // dsv4_attend takes attend_rows_kernel from this many tokens a call
constexpr int GROUP_DIM = 512;      // the longest head it takes

cudaStream_t cs(void* s) { return (cudaStream_t) s; }
int grid(int n, int b) { return (n + b - 1) / b; }

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Lets `kernel` take as much dynamic shared memory as the current card gives a block, once a card; the most it may
// then take, in bytes.
template <class K>
size_t allow_smem(K kernel, unsigned& done) {
    int dev = 0, most = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess || most <= 1024)
        throw std::runtime_error("dsv4: the card's shared memory a block is unknown");
    const size_t room = size_t(most) - 1024;   // the kernels' own static shared memory
    if (dev < 32 && !(done & (1u << dev))) {
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(room)) != cudaSuccess)
            throw std::runtime_error("dsv4: a kernel cannot be given the card's shared memory");
        done |= 1u << dev;
    }
    return room;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    return x;
}

// One value per thread summed (or the largest of them) over the block; `part` has a float a warp.
__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();
    return s;
}

__device__ __forceinline__ float block_max(float x, float* part) {
    x = warp_max(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float m = part[0];
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

__device__ __forceinline__ float half_at(const uint16_t* p, size_t i) {
    return __half2float(__ushort_as_half(p[i]));
}

// ggml-cuda's rope_yarn() and rope_yarn_ramp(), with the pair's index i0 counted from the first turned value.
__device__ __forceinline__ void yarn(float theta_extrap, const Rope& r, int i0, bool back, float& c, float& s) {
    const float theta_interp = r.freq_scale * theta_extrap;
    float theta = theta_interp, mscale = r.attn_factor;
    if (r.ext_factor != 0.0f) {
        const float y = (i0 / 2 - r.corr0) / fmaxf(0.001f, r.corr1 - r.corr0);
        const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * r.ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / r.freq_scale);
    }
    c = cosf(theta) * mscale;
    s = sinf(theta) * mscale;
    if (back) s = -s;
}

// Token t's position: what the card holds at `at` for it when the launch runs (a recorded launch), else
// pos + t * step.
__device__ __forceinline__ long long position(const long long* at, long long pos, int step, int t) {
    return at ? at[2 * t] : pos + (long long) t * step;
}

// A block a head of a token; a thread a pair of values (dim <= 2 * blockDim.x).
__global__ void rope_kernel(const float* __restrict__ x, float* y, int dim, int off, int n_rot, Rope r,
                            long long pos, int step, bool norm, float eps, bool back, const long long* at) {
    __shared__ float part[32];
    const size_t base = (size_t(blockIdx.y) * gridDim.x + blockIdx.x) * dim;
    const int i0 = 2 * threadIdx.x;
    float x0 = 0.0f, x1 = 0.0f;
    if (i0 < dim) {
        x0 = x[base + i0];
        x1 = x[base + i0 + 1];
    }
    float scale = 1.0f;
    if (norm) scale = 1.0f / sqrtf(block_sum(x0 * x0 + x1 * x1, part) / float(dim) + eps);
    if (i0 >= dim) return;
    x0 *= scale;
    x1 *= scale;
    if (i0 >= off && i0 < off + n_rot) {
        const int iw = i0 - off;
        const float p = float(position(at, pos, step, blockIdx.y));
        float c, s;
        yarn(p * powf(r.theta_scale, iw / 2.0f), r, iw, back, c, s);
        const float a = x0, b = x1;
        x0 = a * c - b * s;
        x1 = a * s + b * c;
    }
    y[base + i0] = x0;
    y[base + i0 + 1] = x1;
}

// A thread a value of a token's row.
__global__ void state_kernel(const float* __restrict__ kv, const float* __restrict__ score,
                             const float* __restrict__ ape, float* __restrict__ ring_kv, float* __restrict__ ring_score,
                             int ring, int width, int ratio, long long pos, const long long* at) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const long long p = position(at, pos, 1, blockIdx.y);
    const size_t src = size_t(blockIdx.y) * width + i, dst = size_t(p % ring) * width + i;
    ring_kv[dst] = kv[src];
    ring_score[dst] = score[src] + ape[size_t(p % ratio) * width + i];
}

// A block a token, a thread a value: nothing unless the token ends a block of ratio positions, else the block's
// compressed row.  Its values come in the order llama.cpp concatenates them, with overlap the rows before the block
// first; the weights are the softmax's, then the sum of weight times value in that order, as ggml's soft_max, mul
// and sum_rows give it.  Then the row is RMS-normed with its weights, turned by the block's first position on its
// last n_rot values, and stored as fp16 at the block's index.
__global__ void squeeze_kernel(const float* __restrict__ ring_kv, const float* __restrict__ ring_score, int ring,
                               int dim, int ratio, bool overlap, const float* __restrict__ norm, float eps, Rope r,
                               int n_rot, long long pos, const long long* at, uint16_t* __restrict__ cache) {
    __shared__ float part[32], vals[1024];
    const long long p = position(at, pos, 1, blockIdx.x);
    if ((p + 1) % ratio != 0) return;
    const int d = threadIdx.x;
    const long long start = p + 1 - ratio;
    const int width = overlap ? 2 * dim : dim, n = overlap ? 2 * ratio : ratio;
    // the j-th value: a ring row and a column, or nothing for a row before the sequence
    auto src = [&](int j, size_t& idx) {
        long long row;
        int col = d;
        if (!overlap) {
            row = start + j;
        } else if (j < ratio) {
            row = start - ratio + j;
        } else {
            row = start + j - ratio;
            col = dim + d;
        }
        if (row < 0) return false;
        idx = size_t(row % ring) * width + col;
        return true;
    };
    float mx = -INFINITY;
    for (int j = 0; j < n; ++j) {
        size_t idx;
        if (src(j, idx)) mx = fmaxf(mx, ring_score[idx]);
    }
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        size_t idx;
        if (src(j, idx)) sum += expf(ring_score[idx] - mx);
    }
    const float inv = 1.0f / sum;
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        size_t idx;
        if (src(j, idx)) acc += expf(ring_score[idx] - mx) * inv * ring_kv[idx];
    }
    const float scale = 1.0f / sqrtf(block_sum(acc * acc, part) / float(dim) + eps);
    vals[d] = acc * scale * norm[d];
    __syncthreads();
    const int off = dim - n_rot;
    float v = vals[d];
    if (d >= off) {
        const int iw = (d - off) & ~1;
        float c, s;
        yarn(float(start) * powf(r.theta_scale, iw / 2.0f), r, iw, false, c, s);
        const float a = vals[off + iw], b = vals[off + iw + 1];
        v = (d - off) & 1 ? a * s + b * c : a * c - b * s;
    }
    cache[size_t(start / ratio) * dim + d] = __half_as_ushort(__float2half(v));
}

// A block a token and SCORE_KEYS keys; a warp a key at a time, a lane two heads of it.  The queries are in shared
// memory a head a row, one value longer than a row so the lanes' rows lie in different banks.
__global__ void idx_score_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                 const uint16_t* __restrict__ keys, float* __restrict__ score, int score_stride,
                                 int n_head, int ratio, long long pos, float scale, const long long* at) {
    extern __shared__ float sq[];   // n_head * (IDX_DIM + 1) queries, n_head weights, SCORE_KEYS * IDX_DIM keys
    float* sw = sq + n_head * (IDX_DIM + 1);
    float* sk = sw + n_head;
    const int t = blockIdx.y;
    const long long visible = (position(at, pos, 1, t) + 1) / ratio;
    const long long k0 = (long long) blockIdx.x * SCORE_KEYS;
    if (k0 >= visible) return;
    const float* qt = q + size_t(t) * n_head * IDX_DIM;
    for (int i = threadIdx.x; i < n_head * IDX_DIM; i += blockDim.x)
        sq[(i / IDX_DIM) * (IDX_DIM + 1) + i % IDX_DIM] = qt[i];
    for (int h = threadIdx.x; h < n_head; h += blockDim.x) sw[h] = w[size_t(t) * n_head + h];
    const int nk = (int) min((long long) SCORE_KEYS, visible - k0);
    for (int i = threadIdx.x; i < nk * IDX_DIM; i += blockDim.x) sk[i] = half_at(keys, size_t(k0) * IDX_DIM + i);
    __syncthreads();
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, warps = blockDim.x / WARP;
    for (int k = wid; k < nk; k += warps) {
        const float* key = sk + k * IDX_DIM;
        float acc = 0.0f;
        for (int h = lane; h < n_head; h += WARP) {
            const float* qh = sq + h * (IDX_DIM + 1);
            float dot = 0.0f;
#pragma unroll 8
            for (int i = 0; i < IDX_DIM; ++i) dot += qh[i] * key[i];
            acc += fmaxf(dot, 0.0f) * sw[h];
        }
        acc = warp_sum(acc);
        if (lane == 0) score[size_t(t) * score_stride + k0 + k] = acc * scale;
    }
}

__device__ __forceinline__ uint32_t order_key(float s) {
    const uint32_t b = __float_as_uint(s + 0.0f);
    return b & 0x80000000u ? ~b : b | 0x80000000u;
}

// x summed over the threads before this one, and over all of them in *all.  part has a slot a warp.
__device__ __forceinline__ int block_before(int x, int* part, int* all) {
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, warps = blockDim.x / WARP;
    int in = x;
#pragma unroll
    for (int o = 1; o < WARP; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffff, in, o, WARP);
        if (lane >= o) in += v;
    }
    if (lane == WARP - 1) part[wid] = in;
    __syncthreads();
    int before = 0, total = 0;
    for (int k = 0; k < warps; ++k) {
        if (k < wid) before += part[k];
        total += part[k];
    }
    __syncthreads();
    *all = total;
    return before + in - x;
}

// A block a token.  The n_top blocks with the highest scores by a radix search for the n_top-th score, eight bits a
// round, then the blocks above it and of those on it the lowest, written in the order of the blocks.  A token that
// sees n_top blocks or fewer takes them all.
__global__ void select_kernel(const float* __restrict__ score, int score_stride, int32_t* __restrict__ sel,
                              int sel_stride, int n_top, int ratio, long long pos, const long long* at) {
    __shared__ int hist[SELECT_THREADS / WARP][256], part[SELECT_THREADS / WARP], pick[2];
    const int n = int((position(at, pos, 1, blockIdx.x) + 1) / ratio);
    int32_t* out = sel + size_t(blockIdx.x) * sel_stride;
    if (n <= n_top) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = i;
        return;
    }
    const float* s = score + size_t(blockIdx.x) * score_stride;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    uint32_t cut = 0, mask = 0;
    int want = n_top;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int d = lane; d < 256; d += WARP) hist[wid][d] = 0;
        __syncwarp();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t k = order_key(s[i]);
            if ((k & mask) == cut) atomicAdd(&hist[wid][(k >> shift) & 255], 1);
        }
        __syncthreads();
        if (threadIdx.x < 256) {
            int t = 0;
            for (int k = 0; k < SELECT_THREADS / WARP; ++k) t += hist[k][threadIdx.x];
            hist[0][threadIdx.x] = t;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int d = 255, above = 0;
            for (; d > 0 && above + hist[0][d] < want; --d) above += hist[0][d];
            pick[0] = d;
            pick[1] = above;
        }
        __syncthreads();
        cut |= uint32_t(pick[0]) << shift;
        mask |= 255u << shift;
        want -= pick[1];
        __syncthreads();
    }
    const int per = (n + int(blockDim.x) - 1) / int(blockDim.x);
    const int lo = threadIdx.x * per, hi = min(lo + per, n);
    int g = 0, e = 0;
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = order_key(s[i]);
        g += k > cut;
        e += k == cut;
    }
    int g_all, e_all;
    const int g0 = block_before(g, part, &g_all);
    int e0 = block_before(e, part, &e_all);
    const int need = n_top - g_all;
    int o = g0 + min(e0, need);
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = order_key(s[i]);
        if (k > cut || (k == cut && e0++ < need)) out[o++] = i;
    }
}

// A block a head of a token.  Its keys are the raw rows of its window, oldest first, then its compressed rows; a
// key is its value too.  Scores, the softmax with the head's sink, then the weighted sum of the keys.
__global__ void attend_kernel(const float* __restrict__ q, const uint16_t* __restrict__ raw, int ring, int window,
                              const uint16_t* __restrict__ comp, int ratio, const int32_t* __restrict__ sel,
                              int sel_stride, int top, const float* __restrict__ sinks, float* __restrict__ out,
                              int dim, int n_head, float scale, long long pos, const long long* at) {
    extern __shared__ float smem[];   // dim query values, then a score a key, then a key's row
    __shared__ float part[32];
    float* sq = smem;
    float* sc = smem + dim;
    const int h = blockIdx.x, t = blockIdx.y;
    const long long p = position(at, pos, 1, t);
    const int n_raw = (int) min((long long) window, p + 1);
    const long long first_raw = p + 1 - n_raw;
    int n_comp = 0;
    if (comp) {
        const long long visible = (p + 1) / ratio;
        n_comp = sel ? (int) min(visible, (long long) top) : (int) visible;
    }
    const int n_key = n_raw + n_comp;
    int* row = (int*) (sc + n_key);
    const float* qh = q + (size_t(t) * n_head + h) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) sq[i] = qh[i];
    for (int k = threadIdx.x; k < n_key; k += blockDim.x)
        row[k] = k < n_raw ? int((first_raw + k) % ring) : sel ? sel[size_t(t) * sel_stride + k - n_raw] : k - n_raw;
    __syncthreads();
    auto key = [&](int k) { return (k < n_raw ? raw : comp) + size_t(row[k]) * dim; };
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, warps = blockDim.x / WARP;
    for (int k = wid; k < n_key; k += warps) {
        const uint16_t* kr = key(k);
        float dot = 0.0f;
        for (int i = lane; i < dim; i += WARP) dot += sq[i] * half_at(kr, i);
        dot = warp_sum(dot);
        if (lane == 0) sc[k] = dot * scale;
    }
    __syncthreads();
    const float sink = sinks[h];
    float mx = sink;
    for (int k = threadIdx.x; k < n_key; k += blockDim.x) mx = fmaxf(mx, sc[k]);
    mx = block_max(mx, part);
    float sum = 0.0f;
    for (int k = threadIdx.x; k < n_key; k += blockDim.x) {
        const float e = expf(sc[k] - mx);
        sc[k] = e;
        sum += e;
    }
    sum = block_sum(sum, part) + expf(sink - mx);
    const float inv = 1.0f / sum;
    float* oh = out + (size_t(t) * n_head + h) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < n_key; ++k) acc += sc[k] * half_at(key(k), i);
        oh[i] = acc * inv;
    }
}

// attend_kernel for a batch of tokens.  The heads share their one key row, so a block takes GROUP_HEADS heads of a
// token, a warp each, and reads the keys GROUP_KEYS at a time into shared memory once for all of them; the softmax
// runs along the keys, from the sink.  attend_kernel read every key twice for every head.
__global__ void attend_rows_kernel(const float* __restrict__ q, const uint16_t* __restrict__ raw, int ring,
                                   int window, const uint16_t* __restrict__ comp, int ratio,
                                   const int32_t* __restrict__ sel, int sel_stride, int top,
                                   const float* __restrict__ sinks, float* __restrict__ out, int dim, int n_head,
                                   float scale, long long pos, const long long* at) {
    // GROUP_HEADS rows of query values; GROUP_KEYS rows of keys, a word a pair of halves and four words more a row:
    // read 16 bytes at a time, the 8 lanes of a quarter warp then reach every bank once, along a row or down a
    // column; a probability a head and key
    extern __shared__ float smem[];
    __shared__ int row[GROUP_KEYS];
    const int pitch = dim / 2 + 4;
    float* sq = smem;
    uint32_t* sk = (uint32_t*) (sq + GROUP_HEADS * dim);
    float* sp = (float*) (sk + GROUP_KEYS * pitch);
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
    const int h = blockIdx.x * GROUP_HEADS + w, t = blockIdx.y;
    const long long p = position(at, pos, 1, t);
    const int n_raw = (int) min((long long) window, p + 1);
    const long long first_raw = p + 1 - n_raw;
    int n_comp = 0;
    if (comp) {
        const long long visible = (p + 1) / ratio;
        n_comp = sel ? (int) min(visible, (long long) top) : (int) visible;
    }
    const int n_key = n_raw + n_comp;
    for (int i = threadIdx.x; i < GROUP_HEADS * dim; i += blockDim.x) {
        const int hh = blockIdx.x * GROUP_HEADS + i / dim;
        sq[i] = hh < n_head ? q[(size_t(t) * n_head + hh) * dim + i % dim] : 0.0f;
    }
    // this lane's part of the head's output: values 8 lane to 8 lane + 7 of every 256
    float o[GROUP_DIM / WARP] = {};
    const int chunks = dim / (8 * WARP);
    float m = h < n_head ? sinks[h] : 0.0f, l = 1.0f;
    for (int k0 = 0; k0 < n_key; k0 += GROUP_KEYS) {
        const int nk = min(GROUP_KEYS, n_key - k0);
        __syncthreads();   // the keys before are read
        if (threadIdx.x < GROUP_KEYS) {
            const int k = k0 + threadIdx.x;
            row[threadIdx.x] = k >= k0 + nk ? 0
                               : k < n_raw  ? int((first_raw + k) % ring)
                               : sel        ? sel[size_t(t) * sel_stride + k - n_raw]
                                            : k - n_raw;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < GROUP_KEYS * (dim / 8); i += blockDim.x) {
            const int kk = i / (dim / 8), c = i % (dim / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (kk < nk) v = ((const uint4*) ((k0 + kk < n_raw ? raw : comp) + size_t(row[kk]) * dim))[c];
            *(uint4*) &sk[kk * pitch + 4 * c] = v;
        }
        __syncthreads();
        float s = -INFINITY;
        if (lane < nk) {
            const uint4* kr = (const uint4*) (sk + lane * pitch);
            const float4* qh = (const float4*) (sq + w * dim);
            float d0 = 0.0f, d1 = 0.0f;
            for (int c = 0; c < dim / 8; ++c) {
                const uint4 kv = kr[c];
                const float4 qa = qh[2 * c], qb = qh[2 * c + 1];
                const float2 a = __half22float2(*(const __half2*) &kv.x), b = __half22float2(*(const __half2*) &kv.y);
                const float2 e = __half22float2(*(const __half2*) &kv.z), f = __half22float2(*(const __half2*) &kv.w);
                d0 += qa.x * a.x + qa.y * a.y + qa.z * b.x + qa.w * b.y;
                d1 += qb.x * e.x + qb.y * e.y + qb.z * f.x + qb.w * f.y;
            }
            s = (d0 + d1) * scale;
        }
        const float mn = fmaxf(m, warp_max(s));
        const float alpha = expf(m - mn), e = lane < nk ? expf(s - mn) : 0.0f;
        l = l * alpha + warp_sum(e);
        m = mn;
        sp[w * GROUP_KEYS + lane] = e;
        __syncwarp();
#pragma unroll
        for (int j = 0; j < GROUP_DIM / WARP; ++j) o[j] *= alpha;
        for (int k = 0; k < nk; ++k) {
            const float pk = sp[w * GROUP_KEYS + k];
            const uint4* kr = (const uint4*) (sk + k * pitch) + lane;
#pragma unroll
            for (int j = 0; j < GROUP_DIM / (8 * WARP); ++j)   // whole, so that o stays in registers
                if (j < chunks) {
                    const uint4 v = kr[WARP * j];
                    const float2 a = __half22float2(*(const __half2*) &v.x), b = __half22float2(*(const __half2*) &v.y);
                    const float2 c = __half22float2(*(const __half2*) &v.z), d = __half22float2(*(const __half2*) &v.w);
                    float* oj = o + 8 * j;
                    oj[0] += pk * a.x, oj[1] += pk * a.y, oj[2] += pk * b.x, oj[3] += pk * b.y;
                    oj[4] += pk * c.x, oj[5] += pk * c.y, oj[6] += pk * d.x, oj[7] += pk * d.y;
                }
        }
        __syncwarp();   // sp is written again for the next keys
    }
    if (h >= n_head) return;
    float* oh = out + (size_t(t) * n_head + h) * dim + 8 * lane;
    const float inv = 1.0f / l;
#pragma unroll
    for (int j = 0; j < GROUP_DIM / (8 * WARP); ++j)
        if (j < chunks) {
            const float* oj = o + 8 * j;
            *(float4*) &oh[8 * WARP * j] = make_float4(oj[0] * inv, oj[1] * inv, oj[2] * inv, oj[3] * inv);
            *(float4*) &oh[8 * WARP * j + 4] = make_float4(oj[4] * inv, oj[5] * inv, oj[6] * inv, oj[7] * inv);
        }
}

__global__ void permute_kernel(const float* __restrict__ x, float* __restrict__ y, int n, int groups, int rows,
                               bool back) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int g = blockIdx.y % groups, t = blockIdx.y / groups;
    const size_t a = (size_t(t) * groups + g) * n + i, b = (size_t(g) * rows + t) * n + i;
    if (back) y[a] = x[b];
    else y[b] = x[a];
}

}  // namespace

void dsv4_rope(const float* x, float* y, int dim, int heads, int off, int n_rot, const Rope& r, int64_t pos,
               int step, int rows, bool norm, float eps, bool back, void* stream, const long long* at) {
    if (rows <= 0) return;
    if (dim % 2 || dim > 2048 || off % 2 || off + n_rot > dim)
        throw std::invalid_argument("dsv4_rope: the heads are not dim <= 2048 with the turned values inside");
    const int threads = std::max(WARP, (dim / 2 + WARP - 1) / WARP * WARP);
    rope_kernel<<<dim3(heads, rows), threads, 0, cs(stream)>>>(x, y, dim, off, n_rot, r, (long long) pos, step, norm,
                                                               eps, back, at);
    ck("dsv4_rope");
}

void dsv4_state(const float* kv, const float* score, const float* ape, float* ring_kv, float* ring_score, int ring,
                int width, int ratio, int64_t pos, int rows, void* stream, const long long* at) {
    if (rows <= 0) return;
    state_kernel<<<dim3(grid(width, 256), rows), 256, 0, cs(stream)>>>(kv, score, ape, ring_kv, ring_score, ring, width,
                                                                       ratio, (long long) pos, at);
    ck("dsv4_state");
}

void dsv4_squeeze(const float* ring_kv, const float* ring_score, int ring, int dim, int ratio, bool overlap,
                  const float* norm, float eps, const Rope& r, int n_rot, int64_t pos, int rows, uint16_t* cache,
                  void* stream, const long long* at) {
    if (rows <= 0) return;
    if (dim % WARP || dim > 1024 || n_rot % 2 || n_rot > dim)
        throw std::invalid_argument("dsv4_squeeze: a row of 32 to 1024 values, in 32s");
    squeeze_kernel<<<rows, dim, 0, cs(stream)>>>(ring_kv, ring_score, ring, dim, ratio, overlap, norm, eps, r, n_rot,
                                                 (long long) pos, at, cache);
    ck("dsv4_squeeze");
}

void dsv4_idx_score(const float* q, const float* w, const uint16_t* keys, float* score, int score_stride, int dim,
                    int n_head, int ratio, int64_t pos, int rows, float scale, void* stream, const long long* at,
                    int room) {
    if (dim != IDX_DIM)
        throw std::invalid_argument("dsv4_idx_score: the indexer's keys are not 128 values");
    const long long visible = (at ? room : pos + rows) / ratio;   // the last token sees the most
    if (rows <= 0 || visible == 0) return;
    const size_t smem = (size_t(n_head) * (IDX_DIM + 1) + n_head + SCORE_KEYS * IDX_DIM) * sizeof(float);
    static unsigned done = 0;
    if (smem > allow_smem(idx_score_kernel, done))
        throw std::runtime_error("dsv4_idx_score: the indexer's queries do not fit the card's shared memory");
    idx_score_kernel<<<dim3(grid((int) visible, SCORE_KEYS), rows), 256, smem, cs(stream)>>>(
        q, w, keys, score, score_stride, n_head, ratio, (long long) pos, scale, at);
    ck("dsv4_idx_score");
}

void dsv4_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int ratio,
                 int64_t pos, int rows, void* stream, const long long* at) {
    if (rows <= 0) return;
    select_kernel<<<rows, SELECT_THREADS, 0, cs(stream)>>>(score, score_stride, sel, sel_stride, n_top, ratio,
                                                           (long long) pos, at);
    ck("dsv4_select");
}

void dsv4_attend(const float* q, const uint16_t* raw, int ring, int window, const uint16_t* comp, int ratio,
                 const int32_t* sel, int sel_stride, int top, const float* sinks, float* out, int dim, int n_head,
                 float scale, int64_t pos, int rows, void* stream, const long long* at, int room) {
    if (rows <= 0) return;
    // a batch: its shared memory does not grow with the keys.  A few tokens keep a block a head, which spreads a
    // token over more of the card.
    if (rows >= GROUP_ROWS && dim % (8 * WARP) == 0 && dim <= GROUP_DIM) {
        const size_t smem = (size_t(GROUP_HEADS) * dim + GROUP_HEADS * GROUP_KEYS) * sizeof(float) +
                            size_t(GROUP_KEYS) * (dim / 2 + 4) * sizeof(uint32_t);
        static unsigned done = 0;
        if (smem <= allow_smem(attend_rows_kernel, done)) {
            attend_rows_kernel<<<dim3(grid(n_head, GROUP_HEADS), rows), GROUP_HEADS * WARP, smem, cs(stream)>>>(
                q, raw, ring, window, comp, ratio, sel, sel_stride, top, sinks, out, dim, n_head, scale,
                (long long) pos, at);
            ck("dsv4_attend");
            return;
        }
    }
    long long keys = window;
    if (comp) keys += sel ? top : (at ? room : pos + rows) / ratio;
    const size_t smem = (size_t(dim) + 2 * size_t(keys)) * sizeof(float);
    static unsigned done = 0;
    if (smem > allow_smem(attend_kernel, done))
        throw std::runtime_error("dsv4_attend: " + std::to_string(keys) + " keys a token do not fit the card's shared "
                                 "memory; lower --max-context");
    attend_kernel<<<dim3(n_head, rows), ATTEND_THREADS, smem, cs(stream)>>>(q, raw, ring, window, comp, ratio, sel,
                                                                           sel_stride, top, sinks, out, dim, n_head,
                                                                           scale, (long long) pos, at);
    ck("dsv4_attend");
}

void permute_groups(const float* x, float* y, int n, int groups, int rows, bool back, void* stream) {
    if (rows <= 0) return;
    permute_kernel<<<dim3(grid(n, 256), groups * rows), 256, 0, cs(stream)>>>(x, y, n, groups, rows, back);
    ck("permute_groups");
}

}  // namespace strata::poly
