#include <gtest/gtest.h>

#include <omp.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "Utilities/TarFile/Crc32.h"
#include "Utilities/TarFile/Crc32c.h"

using namespace std;

static double bench_gbs(const char* name,
                        uint32_t (*fn)(const uint32_t crc_accum, const uint8_t*, const size_t),
                        const vector<uint8_t>& buf,
                        int iters) {
    // warmup
    volatile uint32_t sink = 0;
    sink = fn(sink, buf.data(), buf.size());

    auto t0 = chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        sink = fn(sink, buf.data(), buf.size());
    }
    auto t1 = chrono::steady_clock::now();
    chrono::duration<double> dt = t1 - t0;

    const double bytes = double(buf.size()) * double(iters);
    const double gbs = bytes / dt.count() / 1e9;
    printf("%s: %.2f GB/s (sink=%08x)\n", name, gbs, (unsigned)sink);
    return gbs;
}

static double bench_gbs_omp_independent(const char* name,
                                        uint32_t (*fn)(const uint32_t crc_accum, const uint8_t*, const size_t),
                                        size_t per_thread_bytes,
                                        int iters,
                                        int num_threads) {
#ifndef _OPENMP
    printf("%s: OpenMP not enabled; skipping MT benchmark.\n", name);
    return 0.0;
#else
    // Build per-thread buffers deterministically (different seed per thread)
    vector<vector<uint8_t> > bufs(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        bufs[t].resize(per_thread_bytes);
        mt19937_64 rng(123 + 1000ull * (uint64_t)t);
        for (auto& b : bufs[t])
            b = static_cast<uint8_t>(rng());
    }

    // Warmup (parallel)
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        volatile uint32_t sink = 0;
        sink = fn(sink, bufs[tid].data(), bufs[tid].size());
    }

    auto t0 = chrono::steady_clock::now();

    for (int it = 0; it < iters; ++it) {
#pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            volatile uint32_t sink = 0;
            sink = fn(sink, bufs[tid].data(), bufs[tid].size());
        }
    }

    auto t1 = chrono::steady_clock::now();
    chrono::duration<double> dt = t1 - t0;

    const double bytes = double(per_thread_bytes) * double(iters) * double(num_threads);
    const double gbs = bytes / dt.count() / 1e9;

    printf("%s (OpenMP %d threads, %.2f MiB/thread): %.2f GB/s\n", name, num_threads, double(per_thread_bytes) / (1024.0 * 1024.0), gbs);
    return gbs;
#endif
}

static double bench_memcpy_gbs(const char* name, vector<uint8_t>& dst, const vector<uint8_t>& src, int iters) {
    if (dst.size() != src.size()) {
        printf("%s: dst/src size mismatch\n", name);
        return 0.0;
    }

    // warmup
    memcpy(dst.data(), src.data(), src.size());

    auto t0 = chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        memcpy(dst.data(), src.data(), src.size());
    }
    auto t1 = chrono::steady_clock::now();

    const double bytes = double(src.size()) * double(iters);
    const double sec = chrono::duration<double>(t1 - t0).count();
    const double gbs = bytes / sec / 1e9;

    // touch one byte so compiler can't “optimize away” (memcpy won't, but keep consistent)
    volatile uint8_t sink = dst[dst.size() / 2];
    (void)sink;

    printf("%s: %.2f GB/s\n", name, gbs);
    return gbs;
}

static double bench_memread_gbs(const char* name, const vector<uint8_t>& buf, int iters) {
    const size_t n64 = buf.size() / 8;
    const uint64_t* p = reinterpret_cast<const uint64_t*>(buf.data());
    volatile uint64_t sink = 0;

    auto t0 = chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        size_t i = 0;
        for (; i + 4 <= n64; i += 4) {
            s0 += p[i + 0];
            s1 += p[i + 1];
            s2 += p[i + 2];
            s3 += p[i + 3];
        }
        for (; i < n64; ++i)
            s0 += p[i];
        sink ^= (s0 ^ s1 ^ s2 ^ s3);
    }
    auto t1 = chrono::steady_clock::now();

    const double bytes = double(n64 * 8ull) * double(iters);
    const double sec = chrono::duration<double>(t1 - t0).count();
    const double gbs = bytes / sec / 1e9;

    printf("%s: %.2f GB/s (sink=%llx)\n", name, gbs, (unsigned long long)sink);
    return gbs;
}

static double bench_memcpy_gbs_omp_independent(const char* name, size_t per_thread_bytes, int iters, int num_threads) {
#ifndef _OPENMP
    printf("%s: OpenMP not enabled; skipping MT benchmark.\n", name);
    return 0.0;
#else
    vector<vector<uint8_t> > src(num_threads), dst(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        src[t].resize(per_thread_bytes);
        dst[t].resize(per_thread_bytes);

        mt19937_64 rng(123 + 1000ull * (uint64_t)t);
        for (auto& b : src[t])
            b = static_cast<uint8_t>(rng());
    }

    // warmup
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        memcpy(dst[tid].data(), src[tid].data(), per_thread_bytes);
    }

    auto t0 = chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
#pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            memcpy(dst[tid].data(), src[tid].data(), per_thread_bytes);
        }
    }
    auto t1 = chrono::steady_clock::now();

    const double bytes = double(per_thread_bytes) * double(iters) * double(num_threads);
    const double sec = chrono::duration<double>(t1 - t0).count();
    const double gbs = bytes / sec / 1e9;

    volatile uint8_t sink = dst[0][per_thread_bytes / 2];
    (void)sink;

    printf("%s (OpenMP %d threads, %.2f MiB/thread): %.2f GB/s\n", name, num_threads, double(per_thread_bytes) / (1024.0 * 1024.0), gbs);
    return gbs;
#endif
}

static double bench_memread_gbs_omp_independent(const char* name, size_t per_thread_bytes, int iters, int num_threads) {
#ifndef _OPENMP
    printf("%s: OpenMP not enabled; skipping MT benchmark.\n", name);
    return 0.0;
#else
    vector<vector<uint8_t> > bufs(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        bufs[t].resize(per_thread_bytes);
        mt19937_64 rng(123 + 1000ull * (uint64_t)t);
        for (auto& b : bufs[t])
            b = static_cast<uint8_t>(rng());
    }

    volatile uint64_t sink = 0;

    // warmup
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const uint64_t* p = reinterpret_cast<const uint64_t*>(bufs[tid].data());
        const size_t n64 = bufs[tid].size() / 8;
        uint64_t s = 0;
        for (size_t i = 0; i < n64; ++i)
            s += p[i];
#pragma omp atomic
        sink ^= s;
    }

    auto t0 = chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
#pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            const uint64_t* p = reinterpret_cast<const uint64_t*>(bufs[tid].data());
            const size_t n64 = bufs[tid].size() / 8;
            uint64_t s = 0;
            for (size_t i = 0; i < n64; ++i)
                s += p[i];
#pragma omp atomic
            sink ^= s;
        }
    }
    auto t1 = chrono::steady_clock::now();

    const double bytes = double(per_thread_bytes) * double(iters) * double(num_threads);
    const double sec = chrono::duration<double>(t1 - t0).count();
    const double gbs = bytes / sec / 1e9;

    printf("%s (OpenMP %d threads, %.2f MiB/thread): %.2f GB/s (sink=%llx)\n",
           name,
           num_threads,
           double(per_thread_bytes) / (1024.0 * 1024.0),
           gbs,
           (unsigned long long)sink);
    return gbs;
#endif
}

TEST(Crc32Ieee_ZLibNg, DISABLED_BenchmarkVsRawMemoryRead) {
    const size_t sz = 1ull * 1024ull * 1024ull * 1024ull;  // 1 GiB
    vector<uint8_t> buf(sz);
    vector<uint8_t> buf2(sz);

    mt19937_64 rng(123);
    for (auto& b : buf)
        b = static_cast<uint8_t>(rng());

    const int iters = 8;
    const int threads = 8;                                        // match your CRC run
    const size_t per_thread_bytes = 1024ull * 1024ull * 1024ull;  // 1 GiB/thread (adjust if RAM tight)
    const int mt_iters = 8;

    // ---- Memory bandwidth baselines (single-thread) ----
    bench_memread_gbs("MEMREAD (single-thread)", buf, iters);
    bench_memcpy_gbs("MEMCPY  (single-thread)", buf2, buf, iters);

    // 1-thread baselines
    bench_gbs("CRC32c HW support", &thor_file::Crc32c::update, buf, iters);
    bench_gbs("CRC32-B (zlib-ng)", &crc32_ieee, buf, iters);

    // ---- Memory bandwidth baselines (multi-thread, independent buffers) ----

    bench_memread_gbs_omp_independent("MEMREAD (independent)", per_thread_bytes, mt_iters, threads);
    bench_memcpy_gbs_omp_independent("MEMCPY  (independent)", per_thread_bytes, mt_iters, threads);

    // Multi-threaded comparison: 4 threads, independent buffers
    // Don't allocate 4x4GiB unless you really want to; pick something that stays in RAM.
    // Example: 512 MiB per thread (2 GiB total)

    bench_gbs_omp_independent("CRC32c HW support", &thor_file::Crc32c::update, sz, mt_iters, threads);
    bench_gbs_omp_independent("CRC32-B (zlib-ng)", &crc32_ieee, sz, mt_iters, threads);
}
