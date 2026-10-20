// src/poly/dsv4_test.cu - DeepSeek V4's kernels (kernels_dsv4.cu) against the same arithmetic on the host: the rope,
// the compressor, the indexer's scores and pick, and the attention with its window, its compressed rows and sinks.
// It checks the kernels do what their comments say; llama.cpp is the reference for what that should be
// (tools: a model's tokens against llama.cpp's).
//
// usage: poly_dsv4_test   (exit 0: every kernel within float rounding of the host)
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

std::mt19937 rng(7);

std::vector<float> random(size_t n, float scale = 1.f) {
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}

template <class T>
T* dev(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

std::vector<uint16_t> to_f16(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __half_as_ushort(__float2half(v[i]));
    return h;
}

float f16(uint16_t h) { return __half2float(__ushort_as_half(h)); }

int report(const char* what, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::abs((double) got[i] - want[i]) / (1e-3 + std::abs((double) want[i])));
    const bool ok = worst <= tol;
    std::printf("%-34s %zu values, worst %.2g: %s\n", what, got.size(), worst, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

void rope_host(float* x, int dim, int off, int n_rot, const Rope& r, double pos, bool back) {
    for (int i0 = off; i0 < off + n_rot; i0 += 2) {
        const int iw = i0 - off;
        const float te = (float) pos * std::pow(r.theta_scale, iw / 2.0f);
        float theta = r.freq_scale * te, mscale = r.attn_factor;
        if (r.ext_factor != 0.0f) {
            const float y = (iw / 2 - r.corr0) / std::max(0.001f, r.corr1 - r.corr0);
            const float ramp = (1.0f - std::min(1.0f, std::max(0.0f, y))) * r.ext_factor;
            theta = theta * (1 - ramp) + te * ramp;
            mscale *= 1.0f + 0.1f * std::log(1.0f / r.freq_scale);
        }
        const float c = std::cos(theta) * mscale, s = std::sin(theta) * mscale * (back ? -1.f : 1.f);
        const float a = x[i0], b = x[i0 + 1];
        x[i0] = a * c - b * s;
        x[i0 + 1] = a * s + b * c;
    }
}

int test_rope() {
    const int dim = 512, heads = 4, rows = 3, off = 448, n_rot = 64;
    Rope r;
    r.theta_scale = std::pow(160000.f, -2.f / n_rot);
    r.freq_scale = 1.f / 16;
    r.ext_factor = 1.f;
    r.attn_factor = 1.f / (1.f + 0.1f * std::log(16.f));
    r.corr0 = 10, r.corr1 = 40;
    auto x = random((size_t) dim * heads * rows);
    float* d = dev(x);
    dsv4_rope(d, d, dim, heads, off, n_rot, r, 1000, 4, rows, true, 1e-6f, false, nullptr);
    auto got = host(d, x.size());
    auto want = x;
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < heads; ++h) {
            float* v = want.data() + ((size_t) t * heads + h) * dim;
            double ss = 0;
            for (int i = 0; i < dim; ++i) ss += (double) v[i] * v[i];
            const float sc = 1.f / std::sqrt((float) (ss / dim) + 1e-6f);
            for (int i = 0; i < dim; ++i) v[i] *= sc;
            rope_host(v, dim, off, n_rot, r, 1000 + 4 * t, false);
        }
    int bad = report("rope, normed, YaRN", got, want, 1e-4);
    dsv4_rope(d, d, dim, heads, off, n_rot, r, 1000, 4, rows, false, 0.f, true, nullptr);   // turned back
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < heads; ++h) rope_host(want.data() + ((size_t) t * heads + h) * dim, dim, off, n_rot, r,
                                                  1000 + 4 * t, true);
    bad += report("rope back", host(d, x.size()), want, 1e-4);
    cudaFree(d);
    return bad;
}

int test_squeeze(bool overlap, int ratio, int dim) {
    const int width = overlap ? 2 * dim : dim, rows = 3 * ratio + 1, ring = rows + 2 * ratio, n_rot = 64;
    auto kv = random((size_t) rows * width), sc = random((size_t) rows * width), ape = random((size_t) ratio * width);
    auto norm = random(dim, 0.5f);
    std::vector<float> zeros((size_t) ring * width, 0.f);
    float *d_kv = dev(kv), *d_sc = dev(sc), *d_ape = dev(ape), *r_kv = dev(zeros), *r_sc = dev(zeros), *d_n = dev(norm);
    Rope r;
    r.theta_scale = std::pow(160000.f, -2.f / n_rot);
    r.freq_scale = 1.f / 16;
    r.ext_factor = 1.f;
    r.attn_factor = 1.f / (1.f + 0.1f * std::log(16.f));
    r.corr0 = 10, r.corr1 = 40;
    const int64_t pos = 0, count = (pos + rows) / ratio;
    std::vector<uint16_t> zc((size_t) count * dim, 0);
    uint16_t* d_c = dev(zc);
    dsv4_state(d_kv, d_sc, d_ape, r_kv, r_sc, ring, width, ratio, pos, rows, nullptr);
    dsv4_squeeze(r_kv, r_sc, ring, dim, ratio, overlap, d_n, 1e-6f, r, n_rot, pos, rows, d_c, nullptr);
    std::vector<float> got, want;
    for (uint16_t h : host(d_c, zc.size())) got.push_back(f16(h));
    for (int b = 0; b < count; ++b) {
        std::vector<float> rowv(dim);
        for (int d = 0; d < dim; ++d) {
            std::vector<float> s, v;
            const int start = b * ratio;
            for (int j = 0; j < (overlap ? 2 * ratio : ratio); ++j) {
                int row = start + j, col = d;
                if (overlap && j < ratio) row = start - ratio + j;
                else if (overlap) row = start + j - ratio, col = dim + d;
                if (row < 0) continue;
                s.push_back(sc[(size_t) row * width + col] + ape[(size_t) (row % ratio) * width + col]);
                v.push_back(kv[(size_t) row * width + col]);
            }
            const float mx = *std::max_element(s.begin(), s.end());
            float sum = 0, acc = 0;
            for (float x : s) sum += std::exp(x - mx);
            for (size_t j = 0; j < s.size(); ++j) acc += std::exp(s[j] - mx) / sum * v[j];
            rowv[d] = acc;
        }
        double ss = 0;
        for (float x : rowv) ss += (double) x * x;
        const float scale = 1.f / std::sqrt((float) (ss / dim) + 1e-6f);
        for (int d = 0; d < dim; ++d) rowv[d] *= scale * norm[d];
        rope_host(rowv.data(), dim, dim - n_rot, n_rot, r, b * ratio, false);
        for (float x : rowv) want.push_back(f16(__half_as_ushort(__float2half(x))));
    }
    char what[64];
    std::snprintf(what, sizeof what, "squeeze ratio %d%s", ratio, overlap ? ", overlap" : "");
    const int bad = report(what, got, want, 2e-3);
    for (float* p : {d_kv, d_sc, d_ape, r_kv, r_sc, d_n}) cudaFree(p);
    cudaFree(d_c);
    return bad;
}

// rows from GROUP_ROWS (32) take the batch kernel; 12 heads leave its second group of 8 short
int test_attend(bool with_sel, int ratio, int rows, int heads, int window) {
    const int dim = 512, top = 6;
    const int64_t pos = 60;
    const int ring = window + rows, n_comp = (int) ((pos + rows) / ratio);
    auto q = random((size_t) rows * heads * dim, 0.05f);
    auto rawf = random((size_t) ring * dim), compf = random((size_t) n_comp * dim), sinks = random(heads);
    auto raw = to_f16(rawf), comp = to_f16(compf);
    std::vector<int32_t> sel((size_t) rows * top);
    for (int t = 0; t < rows; ++t)
        for (int j = 0; j < top; ++j) sel[(size_t) t * top + j] = (j * 3 + t) % std::max(1, (int) ((pos + t + 1) / ratio));
    float *d_q = dev(q), *d_s = dev(sinks);
    uint16_t *d_raw = dev(raw), *d_comp = dev(comp);
    int32_t* d_sel = dev(sel);
    std::vector<float> zo((size_t) rows * heads * dim);
    float* d_out = dev(zo);
    dsv4_attend(d_q, d_raw, ring, window, d_comp, ratio, with_sel ? d_sel : nullptr, top, top, d_s, d_out, dim, heads,
                0.04f, pos, rows, nullptr);
    auto got = host(d_out, zo.size());
    std::vector<float> want(zo.size());
    for (int t = 0; t < rows; ++t) {
        const int64_t p = pos + t;
        std::vector<const uint16_t*> keys;
        for (int64_t k = std::max<int64_t>(0, p - window + 1); k <= p; ++k) keys.push_back(raw.data() + (k % ring) * dim);
        const int vis = (int) ((p + 1) / ratio);
        if (with_sel)
            for (int j = 0; j < std::min(vis, top); ++j) keys.push_back(comp.data() + (size_t) sel[(size_t) t * top + j] * dim);
        else
            for (int b = 0; b < vis; ++b) keys.push_back(comp.data() + (size_t) b * dim);
        for (int h = 0; h < heads; ++h) {
            const float* qh = q.data() + ((size_t) t * heads + h) * dim;
            std::vector<float> sc(keys.size());
            float mx = sinks[h];
            for (size_t k = 0; k < keys.size(); ++k) {
                float d = 0;
                for (int i = 0; i < dim; ++i) d += qh[i] * f16(keys[k][i]);
                sc[k] = d * 0.04f;
                mx = std::max(mx, sc[k]);
            }
            float sum = std::exp(sinks[h] - mx);
            for (float& s : sc) sum += s = std::exp(s - mx);
            for (int i = 0; i < dim; ++i) {
                float acc = 0;
                for (size_t k = 0; k < keys.size(); ++k) acc += sc[k] * f16(keys[k][i]);
                want[((size_t) t * heads + h) * dim + i] = acc / sum;
            }
        }
    }
    char what[64];
    std::snprintf(what, sizeof what, "attend ratio %d%s, %d tokens", ratio, with_sel ? ", picked" : "", rows);
    const int bad = report(what, got, want, 1e-3);
    for (void* p : {(void*) d_q, (void*) d_s, (void*) d_raw, (void*) d_comp, (void*) d_sel, (void*) d_out}) cudaFree(p);
    return bad;
}

int test_index() {
    const int heads = 64, dim = 128, ratio = 4, top = 16, rows = 3;
    const int64_t pos = 400;
    const int stride = (int) ((pos + rows) / ratio) + 1, n_keys = stride;
    auto q = random((size_t) rows * heads * dim, 0.1f), w = random((size_t) rows * heads);
    auto keysf = random((size_t) n_keys * dim);
    auto keys = to_f16(keysf);
    float *d_q = dev(q), *d_w = dev(w);
    uint16_t* d_k = dev(keys);
    std::vector<float> zs((size_t) rows * stride, -1e30f);
    float* d_s = dev(zs);
    std::vector<int32_t> zsel((size_t) rows * top);
    int32_t* d_sel = dev(zsel);
    dsv4_idx_score(d_q, d_w, d_k, d_s, stride, dim, heads, ratio, pos, rows, 0.01f, nullptr);
    dsv4_select(d_s, stride, d_sel, top, top, ratio, pos, rows, nullptr);
    auto got = host(d_s, zs.size());
    auto sel = host(d_sel, zsel.size());
    std::vector<float> g, want;
    int bad = 0;
    for (int t = 0; t < rows; ++t) {
        const int vis = (int) ((pos + t + 1) / ratio);
        std::vector<float> sc(vis);
        for (int b = 0; b < vis; ++b) {
            float acc = 0;
            for (int h = 0; h < heads; ++h) {
                float d = 0;
                for (int i = 0; i < dim; ++i) d += q[((size_t) t * heads + h) * dim + i] * f16(keys[(size_t) b * dim + i]);
                acc += std::max(d, 0.f) * w[(size_t) t * heads + h];
            }
            sc[b] = acc * 0.01f;
            g.push_back(got[(size_t) t * stride + b]);
            want.push_back(sc[b]);
        }
        // the pick: the top scores the kernel computed, in the order of the rows
        std::vector<int> order(vis);
        std::iota(order.begin(), order.end(), 0);
        const float* mine = got.data() + (size_t) t * stride;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return mine[a] > mine[b]; });
        order.resize(top);
        std::sort(order.begin(), order.end());
        for (int j = 0; j < top; ++j) bad += sel[(size_t) t * top + j] != order[j];
    }
    bad = bad ? 1 : 0;
    std::printf("%-34s %d tokens: %s\n", "indexer pick", rows, bad ? "FAIL" : "ok");
    bad += report("indexer scores", g, want, 1e-3);
    for (void* p : {(void*) d_q, (void*) d_w, (void*) d_k, (void*) d_s, (void*) d_sel}) cudaFree(p);
    return bad;
}

}  // namespace

int main() {
    int bad = test_rope();
    bad += test_squeeze(true, 4, 512);
    bad += test_squeeze(true, 4, 128);
    bad += test_squeeze(false, 128, 512);
    bad += test_attend(false, 128, 5, 8, 16);
    bad += test_attend(true, 4, 5, 8, 16);
    bad += test_attend(false, 4, 5, 8, 16);
    bad += test_attend(false, 128, 40, 12, 40);
    bad += test_attend(true, 4, 40, 12, 40);
    bad += test_attend(false, 4, 40, 12, 40);
    bad += test_index();
    if (cudaGetLastError() != cudaSuccess) {
        std::printf("CUDA error\n");
        return 2;
    }
    return bad ? 1 : 0;
}
