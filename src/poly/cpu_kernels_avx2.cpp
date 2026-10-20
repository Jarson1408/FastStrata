// src/poly/cpu_kernels_avx2.cpp - the CPU experts' K-quant rows for several tokens at once, bit for bit ggml-cpu's.
//
// ggml-cpu is built for AVX2 here, so its vec_dot for Q4_K, Q5_K and Q6_K against Q8_K is the __AVX2__ branch of
// ggml-cpu/arch/x86/quants.c.  These are those branches with the loop over the tokens inside the loop over the
// blocks: what depends only on the weights (the nibbles, the fifth and sixth bits, the 6-bit scales and mins) is
// unpacked once a block, and every token then runs ggml's integer chain and ggml's float operations in ggml's order,
// so each token's result is ggml's to the bit (poly_cpu_kernels_test checks it on random blocks, the bench on the
// model's).  The same scheme as src/kernels/cpu/kq_avx2.cpp, which does it for the other engine's types.
//
// A row is read once for all the tokens of the call that use its expert, and a call of the rows loops over them
// itself: a Q5_K down row is 352 bytes, which a vec_dot call a row and token spent much of its time getting to.
// Prefetching 1 to 8 KB ahead of the rows was tried and measured as nothing.
#include "cpu_experts.hpp"

#include "ggml.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace strata::poly::kq {
namespace {

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
// the fp16 pair {d, dmin} a K-quant block starts with (a union in ggml-common.h's C++ declarations)
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

// ggml-cpu/arch/x86/quants.c's helpers, the AVX2 forms
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
alignas(32) const uint8_t k_shuffle_k4[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
};
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_load_si256((const __m256i*) k_shuffle_k4 + i); }
alignas(16) const uint8_t k_shuffle_q6[128] = {
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,  6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
     8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,
    12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13, 14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,
};
inline __m128i get_scale_shuffle(int i) { return _mm_load_si128((const __m128i*) k_shuffle_q6 + i); }

// The 6-bit scales and mins of a Q4_K / Q5_K block as ggml's AVX2 branch lays them out: 16-bit scales in the low
// half, mins in the high half.
inline __m256i k4_mins_and_scales(const uint8_t* scales12) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    std::memcpy(utmp, scales12, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    return _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
}

// ggml_vec_dot_q4_K_q8_K, nt tokens
void q4k_row(const block_q4_K* x, int nb, const block_q8_K* const* act, int nt, float* res) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[kMaxTokens];
    __m128 acc_m[kMaxTokens];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
        acc_m[t] = _mm_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        const __m256i mins_and_scales = k4_mins_and_scales(x[i].scales);
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i p16l = _mm256_madd_epi16(scale_l[j], _mm256_maddubs_epi16(q4l[j], q8l));
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i p16h = _mm256_madd_epi16(scale_h[j], _mm256_maddubs_epi16(q4h[j], q8h));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// ggml_vec_dot_q5_K_q8_K, nt tokens
void q5k_row(const block_q5_K* x, int nb, const block_q8_K* const* act, int nt, float* res) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);
    __m256 acc[kMaxTokens];
    float summs[kMaxTokens];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
        summs[t] = 0.f;
    }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        const __m256i mins_and_scales = k4_mins_and_scales(x[i].scales);
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);
        __m256i hmask = mone;
        __m256i scale_0[4], scale_1[4], q5_0[4], q5_1[4];
        const uint8_t* q5 = x[i].qs;
        int bit = 0;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_0[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_1[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) q5);
            q5 += 32;
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_0[j] = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_1[j] = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[t] += dmin * _mm_extract_epi32(hsum, 0);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i p16_0 = _mm256_madd_epi16(scale_0[j], _mm256_maddubs_epi16(q5_0[j], q8_0));
                const __m256i p16_1 = _mm256_madd_epi16(scale_1[j], _mm256_maddubs_epi16(q5_1[j], q8_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ggml_vec_dot_q6_K_q8_K, nt tokens
void q6k_row(const block_q6_K* x, int nb, const block_q8_K* const* act, int nt, float* res) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256 acc[kMaxTokens];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d);
        const __m128i scales = _mm_loadu_si128((const __m128i*) x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        __m256i q6[8], sc[8];
        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) qh);
            qh += 32;
            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);
            q6[4 * j + 0] = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            q6[4 * j + 1] = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            q6[4 * j + 2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            q6[4 * j + 3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
            for (int k = 0; k < 4; ++k)
                sc[4 * j + k] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(4 * j + k)));
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = act[t] + i;
            const float d = y->d * xd;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 128; ++j) {
                __m256i p16[4];
                for (int k = 0; k < 4; ++k) {
                    const __m256i q8k = _mm256_loadu_si256((const __m256i*) q8);
                    q8 += 32;
                    p16[k] = _mm256_madd_epi16(sc[4 * j + k], _mm256_maddubs_epi16(q6[4 * j + k], q8k));
                }
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16[0], p16[1]));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16[2], p16[3]));
            }
            sumi = _mm256_sub_epi32(sumi, q8sclsub);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

template <class Block, void (*Row)(const Block*, int, const block_q8_K* const*, int, float*)>
void rows_of(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
             int r1) {
    const block_q8_K* const* a = (const block_q8_K* const*) act;
    const int nb = n / QK_K;
    float res[kMaxTokens];
    for (int r = r0; r < r1; ++r) {
        Row((const Block*) (w + (size_t) r * row_bytes), nb, a, nt, res);
        for (int t = 0; t < nt; ++t) out[t][r - r0] = res[t];
    }
}

}  // namespace

bool usable() {
    static const bool ok = [] {
        unsigned r1[4] = {0, 0, 0, 0}, r7[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, 1, 0);
        for (int i = 0; i < 4; ++i) r1[i] = (unsigned) x[i];
        __cpuidex(x, 7, 0);
        for (int i = 0; i < 4; ++i) r7[i] = (unsigned) x[i];
#else
        __cpuid_count(1, 0, r1[0], r1[1], r1[2], r1[3]);
        __cpuid_count(7, 0, r7[0], r7[1], r7[2], r7[3]);
#endif
        // OSXSAVE, AVX, FMA, F16C, AVX2, and the OS saving the YMM state
        if (!((r1[2] >> 27) & 1u) || !((r1[2] >> 28) & 1u) || !((r1[2] >> 12) & 1u) || !((r1[2] >> 29) & 1u) ||
            !((r7[1] >> 5) & 1u))
            return false;
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        return (xcr0 & 0x6) == 0x6;
    }();
    return ok;
}

bool has(int type) { return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K; }

void rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
          int r0, int r1) {
    switch (type) {
        case GGML_TYPE_Q4_K: rows_of<block_q4_K, q4k_row>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case GGML_TYPE_Q5_K: rows_of<block_q5_K, q5k_row>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case GGML_TYPE_Q6_K: rows_of<block_q6_K, q6k_row>(w, row_bytes, n, act, nt, out, r0, r1); break;
        default: break;
    }
}

}  // namespace strata::poly::kq
