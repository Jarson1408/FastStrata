// src/poly/mxfp4_test.cpp - mxfp4_block_to_q5_0() against ggml's own arithmetic: random MXFP4 blocks with every
// exponent a half holds dequantize to the same floats as the Q5_0 blocks made of them, and their dot products with
// a Q8_0 row agree but for the order of their sums; an exponent out of that range is refused.  CMake builds this:
// poly_mxfp4_test.
#include "cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using strata::poly::mxfp4_block_to_q5_0;

int main() {
    ggml_cpu_init();
    constexpr int kBlocks = 40 * 256, kRow = 64 * 32;   // every exponent from 104 to 143; dot products of 64 blocks
    std::mt19937 rng(7);
    std::vector<uint8_t> mx((size_t) kBlocks * 17), q5((size_t) kBlocks * 22);
    for (int b = 0; b < kBlocks; ++b) {
        mx[(size_t) b * 17] = (uint8_t) (104 + b % 40);
        for (int j = 1; j < 17; ++j) mx[(size_t) b * 17 + j] = (uint8_t) rng();
        if (!mxfp4_block_to_q5_0(&mx[(size_t) b * 17], &q5[(size_t) b * 22])) {
            std::fprintf(stderr, "FAIL: block %d (exponent %d) refused\n", b, mx[(size_t) b * 17]);
            return 1;
        }
    }
    std::vector<float> a((size_t) kBlocks * 32), c((size_t) kBlocks * 32);
    ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(mx.data(), a.data(), (int64_t) a.size());
    ggml_get_type_traits(GGML_TYPE_Q5_0)->to_float(q5.data(), c.data(), (int64_t) c.size());
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &c[i], sizeof(float)) != 0) {
            std::fprintf(stderr, "FAIL: value %zu is %g as MXFP4 and %g as Q5_0\n", i, a[i], c[i]);
            return 1;
        }

    const ggml_type_traits_cpu* tm = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    const ggml_type_traits_cpu* tq = ggml_get_type_traits_cpu(GGML_TYPE_Q5_0);
    if (tm->vec_dot_type != tq->vec_dot_type) {
        std::fprintf(stderr, "FAIL: the two types meet the activations in different types\n");
        return 1;
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(kRow);
    for (float& v : x) v = nd(rng);
    std::vector<uint8_t> xq(ggml_row_size(tm->vec_dot_type, kRow));
    ggml_get_type_traits_cpu(tm->vec_dot_type)->from_float(x.data(), xq.data(), kRow);
    // the two sum in another order: measured against the sum of the products' sizes
    std::vector<float> xd(kRow);
    ggml_get_type_traits(tm->vec_dot_type)->to_float(xq.data(), xd.data(), kRow);
    double worst = 0;
    for (int r = 0; r < kBlocks / 64; ++r) {
        float dm = 0, dq = 0;
        tm->vec_dot(kRow, &dm, 0, &mx[(size_t) r * 64 * 17], 0, xq.data(), 0, 1);
        tq->vec_dot(kRow, &dq, 0, &q5[(size_t) r * 64 * 22], 0, xq.data(), 0, 1);
        double size = 0;
        for (int i = 0; i < kRow; ++i) size += std::fabs((double) a[(size_t) r * kRow + i] * xd[(size_t) i]);
        worst = std::fmax(worst, std::fabs((double) dm - dq) / std::fmax(1e-30, size));
    }
    if (worst > 1e-6) {
        std::fprintf(stderr, "FAIL: a dot product differs by %.3g of its products' sizes\n", worst);
        return 1;
    }

    uint8_t low[17] = {103}, high[17] = {144}, out[22];
    if (mxfp4_block_to_q5_0(low, out) || mxfp4_block_to_q5_0(high, out)) {
        std::fprintf(stderr, "FAIL: an exponent a half does not hold was converted\n");
        return 1;
    }
    std::printf("ok: %d blocks, the same values; dot products within %.3g\n", kBlocks, worst);
    return 0;
}
