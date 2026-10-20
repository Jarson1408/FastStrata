// src/poly/expert_list_test.cu - native_mmvq_list() against native_mmvq_slots() in Qwen3.6's shape (experts of 2048
// inputs and 512 outputs, gate and up in Q4_K, down in Q5_K or Q6_K), on random blocks: every entry of a list gives
// the bits the slots kernel gives it, for lists of 1 to 32 entries; then the time a list takes both ways.
//
// usage: poly_expert_list_test [--time]   (exit 0: every output the same bits)
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace strata::kernels;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

uint64_t mix(uint64_t x) {
    x ^= x >> 33, x *= 0xff51afd7ed558ccdull, x ^= x >> 33, x *= 0xc4ceb9fe1a85ec53ull, x ^= x >> 33;
    return x;
}

// Random blocks of a K-quant whose scales are small finite halves: `scale_at` is where the block's d (and dmin
// after it, where `two`) lies in its bytes.
std::vector<uint8_t> blocks(size_t n_blocks, size_t block_bytes, size_t scale_at, bool two, uint64_t seed) {
    std::vector<uint8_t> b(n_blocks * block_bytes);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (uint8_t) (mix(seed * 0x9E3779B97F4A7C15ull + i) >> 24);
    for (size_t k = 0; k < n_blocks; ++k) {
        for (int h = 0; h < (two ? 2 : 1); ++h) {
            const float v = 0.001f + 0.01f * float(mix(seed + k * 2 + h) % 1000) / 1000.f;
            const __half x = __float2half(v);
            std::memcpy(&b[k * block_bytes + scale_at + 2 * h], &x, 2);
        }
    }
    return b;
}

struct Expert {
    const uint8_t* slot;   // gate | up | down, as the engine's cache holds an expert
};

}  // namespace

int main(int argc, char** argv) {
    const bool timing = argc > 1 && std::strcmp(argv[1], "--time") == 0;
    constexpr int kEmbd = 2048, kFf = 512, kExperts = 32, kTokens = 4;
    const int gu_type = 12;   // Q4_K
    int fails = 0;
    for (const int d_type : {13, 14}) {   // Q5_K, Q6_K
        const size_t gu_bytes = native_mmvq_weight_bytes(gu_type, kEmbd, kFf);
        const size_t d_bytes = native_mmvq_weight_bytes(d_type, kFf, kEmbd);
        const size_t slot_bytes = (2 * gu_bytes + d_bytes + 255) / 256 * 256;
        std::vector<uint8_t> all(slot_bytes * kExperts + 256, 0);
        for (int e = 0; e < kExperts; ++e) {
            uint8_t* s = all.data() + e * slot_bytes;
            const std::vector<uint8_t> gu = blocks(2 * gu_bytes / 144, 144, 0, true, 7 + e);
            std::memcpy(s, gu.data(), gu.size());
            const std::vector<uint8_t> dn = d_type == 13 ? blocks(d_bytes / 176, 176, 0, true, 1000 + e)
                                                         : blocks(d_bytes / 210, 210, 208, false, 1000 + e);
            std::memcpy(s + 2 * gu_bytes, dn.data(), dn.size());
        }
        uint8_t* d_slots = nullptr;
        ck(cudaMalloc(&d_slots, all.size()), "cudaMalloc");
        ck(cudaMemcpy(d_slots, all.data(), all.size(), cudaMemcpyHostToDevice), "cudaMemcpy");

        cudaStream_t st = nullptr;
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate");
        // the tokens' inputs, and a hidden layer for each of 32 entries, quantized
        constexpr int kMost = 32;
        std::vector<float> hx((size_t) kTokens * kEmbd), hh((size_t) kMost * kFf);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = float((int) (mix(i + 77) % 2001) - 1000) / 400.f;
        for (size_t i = 0; i < hh.size(); ++i) hh[i] = float((int) (mix(i + 99) % 2001) - 1000) / 900.f;
        float *dx = nullptr, *dh = nullptr, *y_slots = nullptr, *y_list = nullptr;
        void *xq = nullptr, *hq = nullptr;
        const size_t xq_row = native_q8_1_bytes(kEmbd), hq_row = native_q8_1_bytes(kFf);
        ck(cudaMalloc(&dx, hx.size() * 4), "cudaMalloc");
        ck(cudaMalloc(&dh, hh.size() * 4), "cudaMalloc");
        ck(cudaMalloc(&xq, xq_row * kTokens), "cudaMalloc");
        ck(cudaMalloc(&hq, hq_row * kMost), "cudaMalloc");
        const size_t y_n = (size_t) kMost * std::max(2 * kFf, kEmbd);
        ck(cudaMalloc(&y_slots, y_n * 4), "cudaMalloc");
        ck(cudaMalloc(&y_list, y_n * 4), "cudaMalloc");
        ck(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "cudaMemcpy");
        ck(cudaMemcpy(dh, hh.data(), hh.size() * 4, cudaMemcpyHostToDevice), "cudaMemcpy");
        native_quantize_q8_1(dx, xq, kEmbd * kTokens, 1, st);
        native_quantize_q8_1(dh, hq, kFf * kMost, 1, st);

        // the list on the card: count, the slots, the tokens
        struct List {
            int count;
            const void* w[kMost];
            int row[kMost];
        };
        List* dl = nullptr;
        ck(cudaMalloc(&dl, sizeof(List)), "cudaMalloc");
        for (int count : {1, 2, 3, 5, 8, 12, 17, 24, 32}) {
            List l{};
            l.count = count;
            for (int j = 0; j < count; ++j) {
                l.w[j] = d_slots + (size_t) ((j * 7 + count) % kExperts) * slot_bytes;
                l.row[j] = j % kTokens;
            }
            ck(cudaMemcpy(dl, &l, sizeof(List), cudaMemcpyHostToDevice), "cudaMemcpy");
            for (int part = 0; part < 2; ++part) {   // gate and up, then down
                const int n_in = part == 0 ? kEmbd : kFf, n_out = part == 0 ? 2 * kFf : kEmbd;
                const int type = part == 0 ? gu_type : d_type;
                const size_t offset = part == 0 ? 0 : 2 * gu_bytes;
                const auto slots_way = [&] {
                    for (int i0 = 0; i0 < count; i0 += 8) {
                        const int m = std::min(8, count - i0);
                        MmvqSlots s{};
                        for (int j = 0; j < m; ++j) {
                            s.w[j] = (const uint8_t*) l.w[i0 + j] + offset;
                            s.x[j] = part == 0 ? (const uint8_t*) xq + (size_t) l.row[i0 + j] * xq_row
                                               : (const uint8_t*) hq + (size_t) (i0 + j) * hq_row;
                            s.y[j] = y_slots + (size_t) (i0 + j) * n_out;
                        }
                        if (!(m > 1 && native_mmvq_slots(type, s, m, n_in, n_out, st)))
                            for (int j = 0; j < m; ++j) native_mmvq(type, s.w[j], s.x[j], s.y[j], n_in, n_out, 1, st);
                    }
                };
                const MmvqList list{&dl->count, dl->w, offset, part == 0 ? dl->row : nullptr,
                                    part == 0 ? xq : hq, part == 0 ? xq_row : hq_row, y_list, (size_t) n_out};
                const auto list_way = [&] {
                    if (!native_mmvq_list(type, list, kMost, n_in, n_out, st)) {
                        std::fprintf(stderr, "no list kernel for type %d\n", type);
                        std::exit(2);
                    }
                };
                ck(cudaMemsetAsync(y_slots, 0xff, y_n * 4, st), "memset");
                ck(cudaMemsetAsync(y_list, 0xee, y_n * 4, st), "memset");
                slots_way();
                list_way();
                ck(cudaStreamSynchronize(st), "run");
                std::vector<uint32_t> a((size_t) count * n_out), b(a.size());
                ck(cudaMemcpy(a.data(), y_slots, a.size() * 4, cudaMemcpyDeviceToHost), "cudaMemcpy");
                ck(cudaMemcpy(b.data(), y_list, b.size() * 4, cudaMemcpyDeviceToHost), "cudaMemcpy");
                size_t differ = 0;
                for (size_t i = 0; i < a.size(); ++i) differ += a[i] != b[i];
                fails += differ != 0;
                std::printf("down %s, %2d entries, %s: %s", d_type == 13 ? "Q5_K" : "Q6_K", count,
                            part == 0 ? "gate+up" : "down   ", differ ? "DIFFER" : "same bits");
                if (differ) std::printf(" (%zu of %zu)", differ, a.size());
                if (timing) {
                    cudaEvent_t e0, e1;
                    ck(cudaEventCreate(&e0), "event");
                    ck(cudaEventCreate(&e1), "event");
                    float ms[2] = {};
                    for (int way = 0; way < 2; ++way) {
                        for (int r = 0; r < 20; ++r) way ? list_way() : slots_way();
                        ck(cudaEventRecord(e0, st), "event");
                        for (int r = 0; r < 200; ++r) way ? list_way() : slots_way();
                        ck(cudaEventRecord(e1, st), "event");
                        ck(cudaEventSynchronize(e1), "event");
                        ck(cudaEventElapsedTime(&ms[way], e0, e1), "event");
                    }
                    std::printf("; slots %.1f us, list %.1f us", 1000.f * ms[0] / 200, 1000.f * ms[1] / 200);
                }
                std::printf("\n");
            }
        }
        cudaFree(dl), cudaFree(d_slots), cudaFree(dx), cudaFree(dh), cudaFree(xq), cudaFree(hq);
        cudaFree(y_slots), cudaFree(y_list);
        cudaStreamDestroy(st);
    }
    std::printf(fails ? "FAILED\n" : "all the same bits\n");
    return fails ? 1 : 0;
}
