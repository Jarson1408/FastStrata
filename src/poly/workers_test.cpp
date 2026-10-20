// src/poly/workers_test.cpp - Workers: every task of a batch runs once, and a task that shares out work of its
// own comes back.  The rest of cpu_experts.cpp needs ggml, so CMake builds this: poly_workers_test.
#include "cpu_experts.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using strata::poly::Workers;

int main() {
    // a pool that waits for itself never comes back: the test says so instead of standing still
    std::atomic<bool> done{false};
    std::thread watch([&] {
        for (int i = 0; i < 300 && !done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!done.load()) {
            std::fprintf(stderr, "FAIL: the pool did not come back in 30 s\n");
            std::_Exit(1);
        }
    });
    Workers pool(8);
    int checks = 0;
    for (int round = 0; round < 2000; ++round) {
        std::atomic<int> outer{0}, inner{0};
        pool.run(16, [&](int) {
            outer.fetch_add(1);
            pool.run(4, [&](int) { inner.fetch_add(1); });
        });
        if (outer.load() != 16 || inner.load() != 64) {
            std::fprintf(stderr, "FAIL: round %d ran %d of 16 tasks and %d of 64 of theirs\n", round, outer.load(),
                         inner.load());
            done.store(true);
            watch.join();
            return 1;
        }
        checks += 2;
    }
    done.store(true);
    watch.join();
    std::printf("workers_test: %d checks passed\n", checks);
    return 0;
}
