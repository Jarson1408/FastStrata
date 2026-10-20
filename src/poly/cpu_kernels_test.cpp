// src/poly/cpu_kernels_test.cpp - --cpu-multi gives ggml-cpu's bits.  On random Q4_K, Q5_K and Q6_K rows (quantized
// by ggml from random floats, the scales and mins at all their sizes) kq::rows() for 1 to 8 tokens against
// ggml-cpu's vec_dot, row by row; then cpu_experts() with and without CpuKernelOptions::multi on random experts of
// the model's shapes (gate and up Q4_K, down Q5_K or Q6_K), tokens sharing some of them, 1 to 13 tokens, the
// SwiGLU's limit reached and not, and a GELU call: every output value the same bits.  No model; on a CPU without
// AVX2 it says so and passes.  CMake builds this: poly_cpu_kernels_test.
#include "cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::poly;

namespace {

std::vector<uint8_t> quantized(int type, int rows, int n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> scale(0.001f, 4.f);
    std::vector<float> x((size_t) rows * n);
    for (int r = 0; r < rows; ++r) {
        const float s = scale(rng);   // rows of very different sizes, and an offset for the mins
        const float off = (r % 3 == 0) ? 0.f : nd(rng) * s;
        for (int i = 0; i < n; ++i) x[(size_t) r * n + i] = nd(rng) * s + off;
    }
    std::vector<uint8_t> q(ggml_row_size((ggml_type) type, n) * (size_t) rows);
    const auto from = ggml_get_type_traits_cpu((ggml_type) type)->from_float;
    for (int r = 0; r < rows; ++r) from(&x[(size_t) r * n], &q[ggml_row_size((ggml_type) type, n) * (size_t) r], n);
    return q;
}

int rows_test(std::mt19937& rng) {
    constexpr int kRows = 37, kN = 2048, kTokens = 8;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) kTokens * kN);
    for (float& v : x) v = nd(rng) * (1.f + 3.f * (float) (&v - x.data()) / (float) x.size());
    const size_t a_bytes = ggml_row_size(GGML_TYPE_Q8_K, kN);
    std::vector<uint8_t> act(a_bytes * kTokens);
    for (int t = 0; t < kTokens; ++t)
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(&x[(size_t) t * kN], &act[a_bytes * t], kN);
    long long checked = 0;
    for (const int type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
        const std::vector<uint8_t> w = quantized(type, kRows, kN, rng);
        const size_t row_bytes = ggml_row_size((ggml_type) type, kN);
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) type);
        for (int nt = 1; nt <= kTokens; ++nt) {
            const int r0 = nt % 3, r1 = kRows - nt % 2;
            std::vector<float> got((size_t) kTokens * kRows, -1.f);
            const void* a[kTokens];
            float* out[kTokens];
            for (int t = 0; t < nt; ++t) {
                a[t] = &act[a_bytes * (size_t) ((t * 5) % kTokens)];   // in another order than they lie
                out[t] = &got[(size_t) t * kRows];
            }
            kq::rows(type, w.data(), row_bytes, kN, a, nt, out, r0, r1);
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < r1; ++r) {
                    float want = 0.f;
                    tr->vec_dot(kN, &want, 0, &w[row_bytes * (size_t) r], 0, a[t], 0, 1);
                    ++checked;
                    if (std::memcmp(&want, &out[t][r - r0], 4) != 0) {
                        std::fprintf(stderr, "FAIL: type %d, %d tokens, row %d, token %d: %.9g, ggml %.9g\n", type, nt,
                                     r, t, out[t][r - r0], want);
                        return 1;
                    }
                }
        }
    }
    std::printf("kq::rows: %lld products, ggml's bits\n", checked);
    return 0;
}

int experts_test(std::mt19937& rng) {
    constexpr int kEmbd = 2048, kFf = 512, kExperts = 12;
    Workers workers(6);
    long long checked = 0;
    for (const int d_type : {GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
        const std::vector<uint8_t> gate = quantized(GGML_TYPE_Q4_K, kExperts * kFf, kEmbd, rng);
        const std::vector<uint8_t> up = quantized(GGML_TYPE_Q4_K, kExperts * kFf, kEmbd, rng);
        const std::vector<uint8_t> down = quantized(d_type, kExperts * kEmbd, kFf, rng);
        const size_t gu = ggml_row_size(GGML_TYPE_Q4_K, kEmbd) * kFf, dn = ggml_row_size((ggml_type) d_type, kFf) * kEmbd;
        for (const int rows : {1, 2, 3, 5, 8, 13}) {
            for (const bool gelu : {false, true}) {
                std::normal_distribution<float> nd(0.f, 3.f);
                std::vector<float> x((size_t) rows * kEmbd);
                for (float& v : x) v = nd(rng);
                std::vector<CpuExpert> ce;
                std::uniform_int_distribution<int> pick(0, kExperts - 1);
                std::uniform_real_distribution<float> wt(0.01f, 0.5f);
                for (int t = 0; t < rows; ++t) {
                    std::vector<int> mine;
                    while ((int) mine.size() < 6) {   // six of twelve: tokens share experts
                        const int e = pick(rng);
                        if (std::find(mine.begin(), mine.end(), e) == mine.end()) mine.push_back(e);
                    }
                    for (const int e : mine)
                        ce.push_back({&gate[gu * (size_t) e], &up[gu * (size_t) e], &down[dn * (size_t) e], wt(rng), t});
                }
                for (const float limit : {1e30f, 2.f}) {
                    std::vector<float> want((size_t) rows * kEmbd), got((size_t) rows * kEmbd);
                    CpuKernelOptions o;
                    set_cpu_kernel_options(o);
                    cpu_experts(workers, ce.data(), (int) ce.size(), rows, GGML_TYPE_Q4_K, d_type, kEmbd, kFf, x.data(),
                                limit, gelu, want.data());
                    o.multi = true;
                    o.down_rows = 48 + rows;   // tasks that do not divide the rows
                    o.gu_rows = 20;
                    set_cpu_kernel_options(o);
                    cpu_experts(workers, ce.data(), (int) ce.size(), rows, GGML_TYPE_Q4_K, d_type, kEmbd, kFf, x.data(),
                                limit, gelu, got.data());
                    for (size_t i = 0; i < want.size(); ++i) {
                        ++checked;
                        if (std::memcmp(&want[i], &got[i], 4) != 0) {
                            std::fprintf(stderr, "FAIL: down type %d, %d tokens, gelu %d, limit %g, value %zu: %.9g, "
                                                 "plain %.9g\n", d_type, rows, (int) gelu, limit, i, got[i], want[i]);
                            return 1;
                        }
                    }
                }
            }
        }
    }
    set_cpu_kernel_options(CpuKernelOptions{});
    std::printf("cpu_experts: %lld outputs, the plain path's bits\n", checked);
    return 0;
}

}  // namespace

int main() {
    ggml_cpu_init();
    if (!kq::usable()) {
        std::printf("this CPU has no AVX2, FMA or F16C: --cpu-multi falls back to ggml-cpu, nothing to compare\n");
        return 0;
    }
    std::mt19937 rng(11);
    if (rows_test(rng) || experts_test(rng)) return 1;
    std::printf("ok\n");
    return 0;
}
