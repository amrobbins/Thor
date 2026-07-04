#include <gtest/gtest.h>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TarFile/UringDirect.h"

#include <cstdint>
#include <deque>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"

namespace fs = std::filesystem;
using namespace ThorImplementation;
using namespace std;

static std::string makeTmpPrefix(const std::string& stem) {
    static int counter = 0;
    ++counter;

    int pid = 0;
    pid = getpid();

    char buf[256];
    std::snprintf(buf, sizeof(buf), "/tmp/%s_%d_%d", stem.c_str(), pid, counter);
    return std::string(buf);
}

static void fillTestPatterns(uint8_t* a, uint8_t* b, uint64_t nBytes) {
    // Use all but 2 processors, but never fewer than 1 thread.
    int maxThreads = omp_get_num_procs();
    int threads = std::max(1, maxThreads - 2);
    omp_set_num_threads(threads);

    // 512-cycle patterns
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < nBytes; ++i) {
        // 0..511 repeating
        a[i] = static_cast<uint8_t>(i & 0x1FF);  // mod 512
        // 511..0 repeating
        b[i] = static_cast<uint8_t>(511 - (i & 0x1FF));  // 511 - (mod 512)
    }
}

static bool checkTestPattern(const uint8_t* fileMem, uint64_t bufferBytes, uint32_t numBuffers, uint64_t* firstBadIndexOut = nullptr) {
    const uint64_t total = numBuffers * bufferBytes;

    // Use all but 2 processors, but at least 1 thread.
    int threads = std::max(1, omp_get_num_procs() - 2);

    // We'll find the minimum failing index across threads.
    uint64_t globalFirstBad = total;  // "no failure" sentinel

#pragma omp parallel num_threads(threads)
    {
        uint64_t localFirstBad = total;

#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(total); ++i) {
            bool countUp = (i / bufferBytes) % 2 == 0;
            const uint8_t expected = countUp
                                         ? static_cast<uint8_t>(static_cast<uint64_t>(i) & 0x1FF)                           // 0..511 repeat
                                         : static_cast<uint8_t>(511 - ((static_cast<uint64_t>(i) - bufferBytes) & 0x1FF));  // 511..0 repeat

            if (fileMem[static_cast<uint64_t>(i)] != expected) {
                // printf("%ld: %d  expected %d\n", i, (int)fileMem[static_cast<uint64_t>(i)], (uint32_t)expected);
                //  Track earliest failure this thread sees.
                localFirstBad = std::min(localFirstBad, static_cast<uint64_t>(i));
            }
        }

        // Reduce to global min failing index.
#pragma omp critical
        {
            globalFirstBad = std::min(globalFirstBad, localFirstBad);
        }
    }

    if (firstBadIndexOut) {
        *firstBadIndexOut = (globalFirstBad == total) ? static_cast<uint64_t>(-1) : globalFirstBad;
    }
    return globalFirstBad == total;
}

static void readEntireFileInto(void* dst, uint64_t bytes, const string& path) {
    ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("ifstream open failed: " + path);
    }

    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!in) {
        // If it failed because the file was shorter, gcount() tells you how many were read.
        throw std::runtime_error("ifstream read failed/short: " + path + " read=" + std::to_string(in.gcount()) +
                                 " expected=" + std::to_string(bytes));
    }
}

static uint64_t fileSizeBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("ifstream open failed: " + path);
    auto sz = in.tellg();
    if (sz < 0)
        throw std::runtime_error("tellg failed: " + path);
    return static_cast<uint64_t>(sz);
}

static inline bool isAlignedPtr(const void* p, std::size_t a) { return (reinterpret_cast<std::uintptr_t>(p) % a) == 0; }

static void readEntireFileIntoDirect(void* dst, uint64_t bytes, const std::string& path) {
    constexpr uint64_t kAlign = 4096;

    if (!dst)
        throw std::runtime_error("readEntireFileIntoDirect: dst is null");
    if (!isAlignedPtr(dst, kAlign)) {
        throw std::runtime_error("readEntireFileIntoDirect: dst not 4k aligned");
    }
    if ((bytes % kAlign) != 0) {
        throw std::runtime_error("readEntireFileIntoDirect: bytes not multiple of 4k");
    }

    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0) {
        throw std::runtime_error("open(O_DIRECT) failed: " + path + ": " + std::strerror(errno));
    }

    uint64_t off = 0;
    while (off < bytes) {
        // Read in moderately-sized aligned chunks (e.g., 8 MiB) to avoid huge single syscalls.
        uint64_t chunk = std::min<uint64_t>(bytes - off, 8ull * 1024 * 1024);
        chunk = (chunk / kAlign) * kAlign;
        if (chunk == 0)
            break;

        ssize_t n = ::pread(fd, static_cast<char*>(dst) + off, static_cast<std::size_t>(chunk), static_cast<off_t>(off));
        if (n < 0) {
            int e = errno;
            close(fd);
            throw std::runtime_error("pread(O_DIRECT) failed: " + path + ": " + std::strerror(e));
        }
        if (n == 0) {  // EOF early
            close(fd);
            throw std::runtime_error("pread(O_DIRECT) short read (EOF): " + path + " at off=" + std::to_string(off) +
                                     " expected=" + std::to_string(bytes));
        }

        // With O_DIRECT, you typically get full aligned reads, but handle partials anyway.
        off += static_cast<uint64_t>(n);
    }

    close(fd);

    if (off != bytes) {
        throw std::runtime_error("readEntireFileIntoDirect: short read: " + path + " read=" + std::to_string(off) +
                                 " expected=" + std::to_string(bytes));
    }
}

TEST(UringDirect, AlignmentRequired) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    uint64_t bufferSize = (uint64_t(1) << 29) + 4096;  // 2^29 + 4096
    TensorDescriptor bufferDescriptor(DataType::UINT8, {bufferSize});

    Tensor buffers[2];
    buffers[0] = Tensor(cpuPlacement, bufferDescriptor, 512);
    buffers[1] = Tensor(cpuPlacement, bufferDescriptor, 512);

    string filename = makeTmpPrefix("prefetch_write_loop");

    UringDirect uringDirect;
    uringDirect.registerDumpFile(filename);
    vector<void*> bufferMem = {buffers[0].getMemPtr(), buffers[1].getMemPtr()};
    ASSERT_THROW(uringDirect.registerReusableBuffers(bufferMem, {bufferSize, bufferSize}), std::runtime_error);
}

TEST(UringDirect, PrefetchWriteLoop) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    const uint64_t sixteenMegs = (uint64_t(1) << 24);
    uint64_t bufferSize = sixteenMegs;
    TensorDescriptor bufferDescriptor(DataType::UINT8, {bufferSize});

    Tensor buffers[4];
    buffers[0] = Tensor(cpuPlacement, bufferDescriptor, 4096);
    buffers[1] = Tensor(cpuPlacement, bufferDescriptor, 4096);
    buffers[2] = Tensor(cpuPlacement, bufferDescriptor, 4096);
    buffers[3] = Tensor(cpuPlacement, bufferDescriptor, 4096);
    uint8_t* b0Mem = buffers[0].getMemPtr<uint8_t>();
    uint8_t* b1Mem = buffers[1].getMemPtr<uint8_t>();
    uint8_t* b2Mem = buffers[2].getMemPtr<uint8_t>();
    uint8_t* b3Mem = buffers[3].getMemPtr<uint8_t>();

    fillTestPatterns(b0Mem, b1Mem, sixteenMegs);
    fillTestPatterns(b2Mem, b3Mem, sixteenMegs);

    string filename = makeTmpPrefix("prefetch_write_loop");

    UringDirect uringDirect;
    uringDirect.registerDumpFile(filename);
    vector<void*> bufferMem = {buffers[0].getMemPtr(), buffers[1].getMemPtr(), buffers[2].getMemPtr(), buffers[3].getMemPtr()};
    uringDirect.registerReusableBuffers(bufferMem, {bufferSize, bufferSize, bufferSize, bufferSize});

    uringDirect.submitWriteFixed(0, 0, sixteenMegs, 0);
    uringDirect.submitWriteFixed(1, sixteenMegs, sixteenMegs, 0);
    uringDirect.submitWriteFixed(2, 2 * sixteenMegs, sixteenMegs, 0);
    uringDirect.submitWriteFixed(3, 3 * sixteenMegs, sixteenMegs, 0);
    uringDirect.submit();
    auto comps = uringDirect.waitCompletionsInOrder(4);

    for (auto& c : comps) {
        ASSERT_GE(c.responseCode, 0) << "write failed: userData=" << c.userData << " res=" << c.responseCode
                                     << " errno=" << -c.responseCode;
        // For your case each should be exactly fiveHundredMB
        ASSERT_EQ(static_cast<uint64_t>(c.responseCode), sixteenMegs) << "short write: userData=" << c.userData;
    }
    ASSERT_EQ(fileSizeBytes(filename), 4 * sixteenMegs);

    uringDirect.finishDumpedFile(false);

    Tensor verifyBuffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {4 * sixteenMegs}));
    readEntireFileInto(verifyBuffer.getMemPtr(), 4 * sixteenMegs, filename);
    // readEntireFileIntoDirect(verifyBuffer.getMemPtr(), 4 * fiveHundredKB, filename);

    // uint32_t start = 2 * sixteenMegs - 10;
    // for (uint32_t i = start; i < start + 1024; ++i) {
    //     printf("%d %d\n", i, uint32_t(verifyBuffer.getMemPtr<uint8_t>()[i]));
    // }

    uint64_t firstBadIndexOut;
    bool checkPassed = checkTestPattern(verifyBuffer.getMemPtr<uint8_t>(), sixteenMegs, 4, &firstBadIndexOut);
    EXPECT_TRUE(checkPassed);
    if (!checkPassed) {
        printf("first bad index %ld\n", firstBadIndexOut);
    }

    fs::remove(filename);
}

TEST(UringDirect, PrefetchReadLoop) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    const uint64_t sixteenMegs = (uint64_t(1) << 24);
    const uint64_t bufferSize = sixteenMegs;
    TensorDescriptor bufferDescriptor(DataType::UINT8, {bufferSize});

    // 4 registered buffers
    Tensor buffers[4];
    for (int i = 0; i < 4; ++i) {
        buffers[i] = Tensor(cpuPlacement, bufferDescriptor, 4096);
        std::memset(buffers[i].getMemPtr<void>(), 0, bufferSize);  // start clean (optional)
    }

    // We'll write known patterns into the file first using plain buffered IO (simplest),
    // OR you can reuse your existing write test setup and just open the produced file.
    // Here I'll reuse your existing io_uring writer to create the file.

    uint8_t* b0Mem = buffers[0].getMemPtr<uint8_t>();
    uint8_t* b1Mem = buffers[1].getMemPtr<uint8_t>();
    uint8_t* b2Mem = buffers[2].getMemPtr<uint8_t>();
    uint8_t* b3Mem = buffers[3].getMemPtr<uint8_t>();

    fillTestPatterns(b0Mem, b1Mem, sixteenMegs);
    fillTestPatterns(b2Mem, b3Mem, sixteenMegs);

    std::string filename = makeTmpPrefix("prefetch_read_loop");

    {
        UringDirect writer;
        writer.registerDumpFile(filename);

        std::vector<void*> bufferMem = {buffers[0].getMemPtr(), buffers[1].getMemPtr(), buffers[2].getMemPtr(), buffers[3].getMemPtr()};
        writer.registerReusableBuffers(bufferMem, {bufferSize, bufferSize, bufferSize, bufferSize});

        writer.submitWriteFixed(0, 0, sixteenMegs, 0);
        writer.submitWriteFixed(1, sixteenMegs, sixteenMegs, 0);
        writer.submitWriteFixed(2, 2 * sixteenMegs, sixteenMegs, 0);
        writer.submitWriteFixed(3, 3 * sixteenMegs, sixteenMegs, 0);
        writer.submit();
        auto comps = writer.waitCompletionsInOrder(4);

        for (auto& c : comps) {
            ASSERT_GE(c.responseCode, 0) << "write failed: userData=" << c.userData << " res=" << c.responseCode
                                         << " errno=" << -c.responseCode;
            ASSERT_EQ(static_cast<uint64_t>(c.responseCode), sixteenMegs) << "short write: userData=" << c.userData;
        }

        ASSERT_EQ(fileSizeBytes(filename), 4 * sixteenMegs);
        writer.finishDumpedFile(false);
    }

    // Now do the read side using io_uring. We'll read into fresh buffers to be sure.
    Tensor readBuffers[4];
    for (int i = 0; i < 4; ++i) {
        readBuffers[i] = Tensor(cpuPlacement, bufferDescriptor, 4096);
        std::memset(readBuffers[i].getMemPtr<void>(), 0xCD, bufferSize);  // poison (optional)
    }

    UringDirect reader;
    reader.registerLoadFile(filename);

    std::vector<void*> readMem = {
        readBuffers[0].getMemPtr(), readBuffers[1].getMemPtr(), readBuffers[2].getMemPtr(), readBuffers[3].getMemPtr()};
    reader.registerReusableBuffers(readMem, {bufferSize, bufferSize, bufferSize, bufferSize});

    reader.submitReadFixed(0, 0, sixteenMegs, 0);
    reader.submitReadFixed(1, sixteenMegs, sixteenMegs, 0);
    reader.submitReadFixed(2, 2 * sixteenMegs, sixteenMegs, 0);
    reader.submitReadFixed(3, 3 * sixteenMegs, sixteenMegs, 0);
    reader.submit();
    auto rcomps = reader.waitCompletionsInOrder(4);

    for (auto& c : rcomps) {
        ASSERT_GE(c.responseCode, 0) << "read failed: userData=" << c.userData << " res=" << c.responseCode << " errno=" << -c.responseCode;
        ASSERT_EQ(static_cast<uint64_t>(c.responseCode), sixteenMegs) << "short read: userData=" << c.userData;
    }

    // Verify each 16MiB region matches the expected patterns.
    // Regions 0 and 2 are "0..511 repeating"; regions 1 and 3 are "511..0 repeating"
    //
    // We can reuse your checkTestPattern by copying into one contiguous buffer,
    // or add a small checker that checks per-region. Here's the simplest: contiguous verify buffer.
    Tensor verifyBuffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {4 * sixteenMegs}), 4096);

    // Concatenate readBuffers into verifyBuffer (memcpy is fine in a test)
    uint8_t* out = verifyBuffer.getMemPtr<uint8_t>();
    std::memcpy(out + 0 * sixteenMegs, readBuffers[0].getMemPtr<uint8_t>(), sixteenMegs);
    std::memcpy(out + 1 * sixteenMegs, readBuffers[1].getMemPtr<uint8_t>(), sixteenMegs);
    std::memcpy(out + 2 * sixteenMegs, readBuffers[2].getMemPtr<uint8_t>(), sixteenMegs);
    std::memcpy(out + 3 * sixteenMegs, readBuffers[3].getMemPtr<uint8_t>(), sixteenMegs);

    uint64_t firstBadIndexOut = 0;
    bool checkPassed = checkTestPattern(verifyBuffer.getMemPtr<uint8_t>(),
                                        sixteenMegs,
                                        /*numChunks=*/4,
                                        &firstBadIndexOut);
    EXPECT_TRUE(checkPassed);
    if (!checkPassed) {
        printf("first bad index %lu\n", (unsigned long)firstBadIndexOut);
    }

    fs::remove(filename);
}

static uint64_t env_u64(const char* name, uint64_t def) {
    const char* v = std::getenv(name);
    if (!v || !*v)
        return def;
    char* end = nullptr;
    unsigned long long x = std::strtoull(v, &end, 10);
    if (end == v)
        return def;
    return static_cast<uint64_t>(x);
}

static bool env_bool(const char* name, bool def = false) {
    const char* v = std::getenv(name);
    if (!v || !*v)
        return def;
    if (!std::strcmp(v, "1"))
        return true;
    if (!std::strcmp(v, "true"))
        return true;
    if (!std::strcmp(v, "TRUE"))
        return true;
    return def;
}

class ScopedUnlink {
   public:
    explicit ScopedUnlink(std::string path) : path_(std::move(path)) {}
    ~ScopedUnlink() {
        if (path_.empty())
            return;
        if (::unlink(path_.c_str()) != 0) {
            std::fprintf(stderr, "ScopedUnlink: unlink('%s') failed: %s\n", path_.c_str(), std::strerror(errno));
        }
    }
    void release() { path_.clear(); }
    const std::string& path() const { return path_; }

   private:
    std::string path_;
};

// Fast-ish fill so the buffers aren't all zeros (helps catch "oops I wrote zeros")
// Uses a simple xorshift pattern per buffer.
static void fillBuffer(uint8_t* p, uint64_t nBytes, uint64_t seed) {
    uint64_t x = seed ? seed : 0x9e3779b97f4a7c15ull;
    for (uint64_t i = 0; i < nBytes; ++i) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        p[i] = static_cast<uint8_t>(x);
    }
}

static double secondsSince(const std::chrono::steady_clock::time_point& t0) {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now() - t0).count();
}

static double gibPerSec(uint64_t bytes, double seconds) {
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    return seconds > 0 ? gib / seconds : 0.0;
}

TEST(UringDirectPerf, SequentialWriteRead) {
    if (!env_bool("RUN_URING_PERF", false)) {
        GTEST_SKIP() << "Set RUN_URING_PERF=1 to run io_uring perf test";
    }

    const uint64_t totalGB = env_u64("URING_PERF_TOTAL_GB", 16);
    const uint64_t chunkMB = env_u64("URING_PERF_CHUNK_MB", 16);
    const uint64_t qd = env_u64("URING_PERF_QD", 32);
    const uint64_t numBufs = env_u64("URING_PERF_NUM_BUFS", 64);
    const uint64_t ringDepth = env_u64("URING_PERF_RING_DEPTH", 512);

    const uint64_t chunkBytes = chunkMB * 1024ull * 1024ull;
    ASSERT_EQ(chunkBytes % 4096ull, 0ull) << "chunk must be 4k multiple";

    const uint64_t totalBytes = totalGB * 1024ull * 1024ull * 1024ull;
    ASSERT_EQ(totalBytes % chunkBytes, 0ull) << "TOTAL_GB must be multiple of chunk size";
    const uint64_t numChunks = totalBytes / chunkBytes;

    // Put the temp file on your fast volume by ensuring makeTmpPrefix picks that dir,
    // or replace with a hard path under your RAID0 mount.
    std::string filename = makeTmpPrefix("uring_perf");
    ScopedUnlink cleanup(filename);

    std::printf("=== UringDirectPerf ===\n");
    std::printf("file: %s\n", filename.c_str());
    std::printf("total: %lu GiB, chunk: %lu MiB, chunks: %lu\n", (unsigned long)totalGB, (unsigned long)chunkMB, (unsigned long)numChunks);
    std::printf("QD: %lu, bufs: %lu, ringDepth: %lu\n", (unsigned long)qd, (unsigned long)numBufs, (unsigned long)ringDepth);
    std::fflush(stdout);

    // Allocate buffers as CPU tensors aligned to 4k (for O_DIRECT).
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    TensorDescriptor bufDesc(DataType::UINT8, {chunkBytes});

    std::vector<Tensor> bufs;
    bufs.reserve(numBufs);
    for (uint64_t i = 0; i < numBufs; ++i) {
        bufs.emplace_back(cpuPlacement, bufDesc, 4096);
        fillBuffer(bufs.back().getMemPtr<uint8_t>(), chunkBytes, 0x1234ull + i);
    }

    std::vector<void*> bufPtrs;
    bufPtrs.reserve(numBufs);
    for (uint64_t i = 0; i < numBufs; ++i)
        bufPtrs.push_back(bufs[i].getMemPtr());

    // -------------------------
    // WRITE BENCH
    // -------------------------
    {
        UringDirect ur(static_cast<unsigned>(ringDepth));
        ur.registerDumpFile(filename, /*truncate=*/true);
        ur.registerReusableBuffers(bufPtrs, std::vector<std::size_t>(numBufs, static_cast<std::size_t>(chunkBytes)));

        // preallocate to avoid filesystem extent growth cost during the run.
        // Best-effort: ignore failures (some FS / quotas / permissions might block).
        (void)::posix_fallocate(ur.fd(), 0, static_cast<off_t>(totalBytes));

        // Warm-up: a few chunks to stabilize CPU freq / NVMe state
        {
            const uint64_t warm = std::min<uint64_t>(numChunks, 8);
            for (uint64_t i = 0; i < warm; ++i) {
                while (!ur.submitWriteFixed(static_cast<unsigned>(i % numBufs), i * chunkBytes, static_cast<uint32_t>(chunkBytes), 0)) {
                    ur.submit();
                    (void)ur.waitCompletionInOrder();
                }
            }
            ur.submit();
            (void)ur.waitCompletionsInOrder(static_cast<std::size_t>(warm));
            ur.finishDumpedFile(true);
        }

        // Main run
        std::deque<uint64_t> inflight;  // tokens in flight
        uint64_t nextChunk = 0;
        uint64_t nextToken = 1;

        auto t0 = std::chrono::steady_clock::now();

        while (nextChunk < numChunks || !inflight.empty()) {
            // Fill up to QD
            while (nextChunk < numChunks && inflight.size() < qd) {
                unsigned bufIndex = static_cast<unsigned>(nextChunk % numBufs);
                uint64_t off = nextChunk * chunkBytes;

                bool queued = ur.submitWriteFixed(bufIndex, off, static_cast<uint32_t>(chunkBytes), 0);
                if (!queued)
                    break;  // SQ full, submit/drain below

                inflight.push_back(nextToken);
                ++nextToken;
                ++nextChunk;
            }

            ur.submit();

            // Drain at least 1 completion when we have inflight and either:
            // - we couldn't queue more, or
            // - we're full
            if (!inflight.empty() && (inflight.size() >= qd || nextChunk >= numChunks)) {
                auto c = ur.waitCompletionInOrder();
                ASSERT_GE(c.responseCode, 0) << "write failed: res=" << c.responseCode << " errno=" << -c.responseCode;
                ASSERT_EQ(static_cast<uint64_t>(c.responseCode), chunkBytes) << "short write";
                inflight.pop_front();
            } else {
                // Opportunistic polling
                auto v = ur.pollCompletionsInOrder(32);
                for (auto& c : v) {
                    ASSERT_GE(c.responseCode, 0) << "write failed: res=" << c.responseCode << " errno=" << -c.responseCode;
                    ASSERT_EQ(static_cast<uint64_t>(c.responseCode), chunkBytes) << "short write";
                    if (!inflight.empty())
                        inflight.pop_front();
                }
            }
        }

        ur.finishDumpedFile(true);

        double sec = secondsSince(t0);
        std::printf("[WRITE] %.2f GiB/s (%.3f s for %lu GiB)\n", gibPerSec(totalBytes, sec), sec, (unsigned long)totalGB);
        std::fflush(stdout);
    }

    // Ensure file size looks right
    {
        struct stat st{};
        ASSERT_EQ(::stat(filename.c_str(), &st), 0);
        ASSERT_EQ(static_cast<uint64_t>(st.st_size), totalBytes);
    }

    // -------------------------
    // READ BENCH
    // -------------------------
    {
        // Fresh buffers (avoid any weird “hot in cache” assumptions about memory)
        std::vector<Tensor> rbufs;
        rbufs.reserve(numBufs);
        for (uint64_t i = 0; i < numBufs; ++i) {
            rbufs.emplace_back(cpuPlacement, bufDesc, 4096);
            std::memset(rbufs.back().getMemPtr<void>(), 0, static_cast<size_t>(chunkBytes));
        }
        std::vector<void*> rPtrs;
        rPtrs.reserve(numBufs);
        for (uint64_t i = 0; i < numBufs; ++i)
            rPtrs.push_back(rbufs[i].getMemPtr());

        UringDirect ur(static_cast<unsigned>(ringDepth));
        ur.registerLoadFile(filename);
        ur.registerReusableBuffers(rPtrs, std::vector<std::size_t>(numBufs, static_cast<std::size_t>(chunkBytes)));

        // Warm-up: a few chunks
        {
            const uint64_t warm = std::min<uint64_t>(numChunks, 8);
            for (uint64_t i = 0; i < warm; ++i) {
                while (!ur.submitReadFixed(static_cast<unsigned>(i % numBufs), i * chunkBytes, static_cast<uint32_t>(chunkBytes), 0)) {
                    ur.submit();
                    (void)ur.waitCompletionInOrder();
                }
            }
            ur.submit();
            (void)ur.waitCompletionsInOrder(static_cast<std::size_t>(warm));
        }

        std::deque<uint64_t> inflight;
        uint64_t nextChunk = 0;
        uint64_t nextToken = 1;

        auto t0 = std::chrono::steady_clock::now();

        while (nextChunk < numChunks || !inflight.empty()) {
            while (nextChunk < numChunks && inflight.size() < qd) {
                unsigned bufIndex = static_cast<unsigned>(nextChunk % numBufs);
                uint64_t off = nextChunk * chunkBytes;

                bool queued = ur.submitReadFixed(bufIndex, off, static_cast<uint32_t>(chunkBytes), 0);
                if (!queued)
                    break;

                inflight.push_back(nextToken);
                ++nextToken;
                ++nextChunk;
            }

            ur.submit();

            if (!inflight.empty() && (inflight.size() >= qd || nextChunk >= numChunks)) {
                auto c = ur.waitCompletionInOrder();
                ASSERT_GE(c.responseCode, 0) << "read failed: res=" << c.responseCode << " errno=" << -c.responseCode;
                ASSERT_EQ(static_cast<uint64_t>(c.responseCode), chunkBytes) << "short read";
                inflight.pop_front();
            } else {
                auto v = ur.pollCompletionsInOrder(32);
                for (auto& c : v) {
                    ASSERT_GE(c.responseCode, 0) << "read failed: res=" << c.responseCode << " errno=" << -c.responseCode;
                    ASSERT_EQ(static_cast<uint64_t>(c.responseCode), chunkBytes) << "short read";
                    if (!inflight.empty())
                        inflight.pop_front();
                }
            }
        }

        double sec = secondsSince(t0);
        std::printf("[READ ] %.2f GiB/s (%.3f s for %lu GiB)\n", gibPerSec(totalBytes, sec), sec, (unsigned long)totalGB);
        std::fflush(stdout);

        // Optional correctness spot-check: compare a few bytes from the first buffer.
        // (Full verification costs time; keep perf tests mostly perf.)
        // uint8_t* p = rbufs[0].getMemPtr<uint8_t>();
        // std::printf("read sample: %u %u %u %u\n", p[0], p[1], p[2], p[3]);
    }

    std::printf("=== done ===\n");
}

TEST(UringDirect, FixedBuffer_SubOffsetsWriteDifferentBlocks) {
    using namespace ThorImplementation;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    constexpr uint32_t kAlign = 4096;
    constexpr uint32_t kBlocks = 2;
    constexpr uint32_t kBytes = kBlocks * kAlign;

    TensorDescriptor desc(DataType::UINT8, {kBytes});
    Tensor buf(cpuPlacement, desc, kAlign);
    uint8_t* p = buf.getMemPtr<uint8_t>();
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p) % kAlign, 0u);

    // Fill 2 distinct 4KB blocks
    for (uint32_t i = 0; i < kAlign; ++i)
        p[i] = 0xAA;
    for (uint32_t i = 0; i < kAlign; ++i)
        p[kAlign + i] = 0x55;

    std::string filename = makeTmpPrefix("uring_fixed_suboffsets");
    ScopedUnlink cleanup(filename);

    UringDirect uring(64);
    uring.registerDumpFile(filename);
    uring.registerReusableBuffers({buf.getMemPtr()}, {kBytes});

    // Two writes at different file offsets, but from different sub-offsets of the same fixed buffer.
    // This test catches the classic bug: writing always from iovecs_[bufIndex].iov_base (no buf offset).
    ASSERT_TRUE(uring.submitWriteFixed(/*bufIndex=*/0,
                                       /*fileOffsetBytes=*/0,
                                       /*lenBytes=*/kAlign,
                                       /*bufOffsetBytes=*/0));
    ASSERT_TRUE(uring.submitWriteFixed(/*bufIndex=*/0,
                                       /*fileOffsetBytes=*/kAlign,
                                       /*lenBytes=*/kAlign,
                                       /*bufOffsetBytes=*/kAlign));

    uring.submit();
    auto comps = uring.waitCompletionsInOrder(2);

    ASSERT_EQ(comps.size(), 2u);
    for (auto& c : comps) {
        ASSERT_GE(c.responseCode, 0) << "write failed: userData=" << c.userData << " res=" << c.responseCode
                                     << " errno=" << -c.responseCode;
        ASSERT_EQ(static_cast<uint32_t>(c.responseCode), kAlign) << "short write: userData=" << c.userData;
    }

    ASSERT_EQ(fileSizeBytes(filename), static_cast<uint64_t>(kBytes));

    // Read back with simple iostream path (not O_DIRECT)
    Tensor verify(cpuPlacement, TensorDescriptor(DataType::UINT8, {kBytes}));
    readEntireFileInto(verify.getMemPtr(), kBytes, filename);

    const uint8_t* v = verify.getMemPtr<uint8_t>();
    ASSERT_NE(v, nullptr);

    // Verify first 4KB is 0xAA and second 4KB is 0x55
    for (uint32_t i = 0; i < kAlign; ++i) {
        EXPECT_EQ(v[i], 0xAA) << "mismatch in block0 at i=" << i;
    }
    for (uint32_t i = 0; i < kAlign; ++i) {
        EXPECT_EQ(v[kAlign + i], 0x55) << "mismatch in block1 at i=" << i;
    }
}

class ScopedUringDirectCompatibilityTestHooks {
   public:
    ScopedUringDirectCompatibilityTestHooks(std::optional<int> queueInitResult, bool directOpenUnavailable) {
        UringDirect::testResetCompatibilityWarning();
        UringDirect::testSetIoUringQueueInitResult(queueInitResult);
        UringDirect::testSetDirectOpenUnavailable(directOpenUnavailable);
    }
    ~ScopedUringDirectCompatibilityTestHooks() {
        UringDirect::testSetIoUringQueueInitResult(std::nullopt);
        UringDirect::testSetDirectOpenUnavailable(false);
        UringDirect::testSetIoUringRegisterBuffersResult(std::nullopt);
        UringDirect::testSetNextIoUringSubmissionByteLimit(std::nullopt);
        UringDirect::testResetFallbackWorkerBlock();
        UringDirect::testResetCompatibilityWarning();
    }
};

static void runExplicitPreadBackendFixedReadWriteRoundTrip(UringDirect::IoBackend backend, const char* expectedBackendName, const std::string& stem) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    constexpr uint32_t kAlign = 4096;
    constexpr uint32_t kBytes = 2 * kAlign;

    Tensor writeBuffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {kBytes}), kAlign);
    Tensor readBuffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {kBytes}), kAlign);

    uint8_t* writePtr = writeBuffer.getMemPtr<uint8_t>();
    uint8_t* readPtr = readBuffer.getMemPtr<uint8_t>();
    ASSERT_NE(writePtr, nullptr);
    ASSERT_NE(readPtr, nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(writePtr) % kAlign, 0u);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(readPtr) % kAlign, 0u);

    for (uint32_t i = 0; i < kBytes; ++i) {
        writePtr[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xFFu);
        readPtr[i] = 0;
    }

    std::string filename = makeTmpPrefix(stem);
    ScopedUnlink cleanup(filename);

    {
        UringDirect writer(64, backend);
        ASSERT_STREQ(writer.requestedBackendName(), expectedBackendName);
        ASSERT_STREQ(writer.activeBackendName(), expectedBackendName);
        writer.registerReusableBuffers({writeBuffer.getMemPtr()}, {kBytes});
        writer.registerDumpFile(filename);
        ASSERT_STREQ(writer.activeBackendName(), expectedBackendName);

        ASSERT_TRUE(writer.submitWriteFixed(/*bufIndex=*/0,
                                           /*fileOffsetBytes=*/0,
                                           /*lenBytes=*/kBytes,
                                           /*bufOffsetBytes=*/0));
        EXPECT_EQ(writer.submit(), 1);
        auto writeCompletions = writer.waitCompletionsInOrder(1);
        ASSERT_EQ(writeCompletions.size(), 1u);
        ASSERT_EQ(writeCompletions[0].responseCode, static_cast<int>(kBytes));

        auto fsyncCompletion = writer.finishDumpedFile(false);
        EXPECT_EQ(fsyncCompletion.responseCode, 0);
    }

    ASSERT_EQ(fileSizeBytes(filename), static_cast<uint64_t>(kBytes));

    {
        UringDirect reader(64, backend);
        ASSERT_STREQ(reader.requestedBackendName(), expectedBackendName);
        ASSERT_STREQ(reader.activeBackendName(), expectedBackendName);
        reader.registerReusableBuffers({readBuffer.getMemPtr()}, {kBytes});
        reader.registerLoadFile(filename);
        ASSERT_STREQ(reader.activeBackendName(), expectedBackendName);

        ASSERT_TRUE(reader.submitReadFixed(/*bufIndex=*/0,
                                          /*fileOffsetBytes=*/0,
                                          /*lenBytes=*/kBytes,
                                          /*bufOffsetBytes=*/0));
        EXPECT_EQ(reader.submit(), 1);
        auto readCompletions = reader.waitCompletionsInOrder(1);
        ASSERT_EQ(readCompletions.size(), 1u);
        ASSERT_EQ(readCompletions[0].responseCode, static_cast<int>(kBytes));
    }

    EXPECT_EQ(std::memcmp(readBuffer.getMemPtr(), writeBuffer.getMemPtr(), kBytes), 0);
}

static void runExplicitPreadBackendAsyncFixedReadWrite(UringDirect::IoBackend backend, const char* expectedBackendName, const std::string& stem) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    constexpr uint32_t kAlign = 4096;
    constexpr uint32_t kBlocks = 4;
    constexpr uint32_t kBytes = kBlocks * kAlign;

    Tensor writeBuffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {kBytes}), kAlign);
    Tensor readBuffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {kBytes}), kAlign);

    uint8_t* writePtr = writeBuffer.getMemPtr<uint8_t>();
    uint8_t* readPtr = readBuffer.getMemPtr<uint8_t>();
    ASSERT_NE(writePtr, nullptr);
    ASSERT_NE(readPtr, nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(writePtr) % kAlign, 0u);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(readPtr) % kAlign, 0u);

    for (uint32_t i = 0; i < kBytes; ++i) {
        writePtr[i] = static_cast<uint8_t>((i * 19u + 23u) & 0xFFu);
        readPtr[i] = 0;
    }

    std::string filename = makeTmpPrefix(stem);
    ScopedUnlink cleanup(filename);

    {
        UringDirect::testResetFallbackWorkerBlock();
        UringDirect writer(kBlocks, backend);
        ASSERT_STREQ(writer.requestedBackendName(), expectedBackendName);
        ASSERT_STREQ(writer.activeBackendName(), expectedBackendName);
        writer.registerReusableBuffers({writeBuffer.getMemPtr()}, {kBytes});
        writer.registerDumpFile(filename);

        UringDirect::testSetFallbackWorkerBlockEnabled(true);
        for (uint32_t block = 0; block < kBlocks; ++block) {
            ASSERT_TRUE(writer.submitWriteFixed(/*bufIndex=*/0,
                                               /*fileOffsetBytes=*/block * kAlign,
                                               /*lenBytes=*/kAlign,
                                               /*bufOffsetBytes=*/block * kAlign));
        }
        EXPECT_EQ(writer.submit(), static_cast<int>(kBlocks));

        bool writeWorkersStarted = UringDirect::testWaitForFallbackWorkerStartedCount(kBlocks, std::chrono::seconds(5));
        EXPECT_TRUE(writeWorkersStarted) << "fallback pwrite requests were not picked up by multiple worker threads";
        EXPECT_TRUE(writer.pollCompletionsInOrder(kBlocks).empty())
            << "fallback write completed while test worker hook was still blocked";

        UringDirect::testSetFallbackWorkerBlockEnabled(false);
        ASSERT_TRUE(writeWorkersStarted);
        auto writeCompletions = writer.waitCompletionsInOrder(kBlocks);
        ASSERT_EQ(writeCompletions.size(), static_cast<std::size_t>(kBlocks));
        for (const auto& completion : writeCompletions) {
            ASSERT_EQ(completion.responseCode, static_cast<int>(kAlign));
        }

        auto fsyncCompletion = writer.finishDumpedFile(false);
        EXPECT_EQ(fsyncCompletion.responseCode, 0);
    }

    ASSERT_EQ(fileSizeBytes(filename), static_cast<uint64_t>(kBytes));

    {
        UringDirect::testResetFallbackWorkerBlock();
        UringDirect reader(kBlocks, backend);
        ASSERT_STREQ(reader.requestedBackendName(), expectedBackendName);
        ASSERT_STREQ(reader.activeBackendName(), expectedBackendName);
        reader.registerReusableBuffers({readBuffer.getMemPtr()}, {kBytes});
        reader.registerLoadFile(filename);

        UringDirect::testSetFallbackWorkerBlockEnabled(true);
        for (uint32_t block = 0; block < kBlocks; ++block) {
            ASSERT_TRUE(reader.submitReadFixed(/*bufIndex=*/0,
                                              /*fileOffsetBytes=*/block * kAlign,
                                              /*lenBytes=*/kAlign,
                                              /*bufOffsetBytes=*/block * kAlign));
        }
        EXPECT_EQ(reader.submit(), static_cast<int>(kBlocks));

        bool readWorkersStarted = UringDirect::testWaitForFallbackWorkerStartedCount(kBlocks, std::chrono::seconds(5));
        EXPECT_TRUE(readWorkersStarted) << "fallback pread requests were not picked up by multiple worker threads";
        EXPECT_TRUE(reader.pollCompletionsInOrder(kBlocks).empty())
            << "fallback read completed while test worker hook was still blocked";

        UringDirect::testSetFallbackWorkerBlockEnabled(false);
        ASSERT_TRUE(readWorkersStarted);
        auto readCompletions = reader.waitCompletionsInOrder(kBlocks);
        ASSERT_EQ(readCompletions.size(), static_cast<std::size_t>(kBlocks));
        for (const auto& completion : readCompletions) {
            ASSERT_EQ(completion.responseCode, static_cast<int>(kAlign));
        }
    }

    UringDirect::testResetFallbackWorkerBlock();
    EXPECT_EQ(std::memcmp(readBuffer.getMemPtr(), writeBuffer.getMemPtr(), kBytes), 0);
}


TEST(UringDirect, RegisterReusableBuffersRejectsReplacementWithInFlightIo) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    constexpr uint32_t kAlign = 4096;

    Tensor firstBuffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {kAlign}), kAlign);
    Tensor replacementBuffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {kAlign}), kAlign);
    uint8_t* p = firstBuffer.getMemPtr<uint8_t>();
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p) % kAlign, 0u);
    for (uint32_t i = 0; i < kAlign; ++i) {
        p[i] = static_cast<uint8_t>(i & 0xFFu);
    }

    std::string filename = makeTmpPrefix("replace_buffers_with_inflight_io");
    ScopedUnlink cleanup(filename);

    UringDirect::testResetFallbackWorkerBlock();
    UringDirect uring(/*queueDepth=*/1, UringDirect::IoBackend::PreadBuffered);
    uring.registerReusableBuffers({firstBuffer.getMemPtr()}, {kAlign});
    uring.registerDumpFile(filename);

    UringDirect::testSetFallbackWorkerBlockEnabled(true);
    ASSERT_TRUE(uring.submitWriteFixed(/*bufIndex=*/0,
                                       /*fileOffsetBytes=*/0,
                                       /*lenBytes=*/kAlign,
                                       /*bufOffsetBytes=*/0));
    EXPECT_EQ(uring.submit(), 1);
    ASSERT_TRUE(UringDirect::testWaitForFallbackWorkerStartedCount(1, std::chrono::seconds(5)));

    EXPECT_THROW(uring.registerReusableBuffers({replacementBuffer.getMemPtr()}, {kAlign}), std::runtime_error)
        << "Replacing registered buffers while exact-I/O is in flight can invalidate retry state.";

    UringDirect::testSetFallbackWorkerBlockEnabled(false);
    auto completions = uring.waitCompletionsInOrder(1);
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_EQ(completions[0].responseCode, static_cast<int>(kAlign));
    EXPECT_EQ(uring.finishDumpedFile(false).responseCode, 0);
    UringDirect::testResetFallbackWorkerBlock();
}

TEST(UringDirect, RejectsTransferLengthOutsideCompletionResponseRange) {
    std::string filename = makeTmpPrefix("reject_huge_cached_read_len");
    {
        std::ofstream out(filename, std::ios::binary);
        ASSERT_TRUE(out.good());
        out.put('\0');
    }
    ScopedUnlink cleanup(filename);

    UringDirect reader(/*queueDepth=*/1, UringDirect::IoBackend::PreadBuffered);
    reader.registerCachedLoadFile(filename);
    uint8_t byte = 0;
    const auto tooLargeForCompletion = static_cast<std::uint32_t>(std::numeric_limits<int>::max()) + 1u;
    EXPECT_THROW(reader.submitReadCached(&byte, /*fileOffsetBytes=*/0, tooLargeForCompletion), std::runtime_error)
        << "Completion responseCode is an int, so oversized logical transfers cannot be represented safely.";
}

TEST(UringDirect, SubmitReadvCachedScattersContiguousFileRangeWithBufferedFallback) {
    std::string filename = makeTmpPrefix("readv_cached_scatter_buffered");
    ScopedUnlink cleanup(filename);

    std::vector<uint8_t> fileBytes(96);
    for (std::size_t i = 0; i < fileBytes.size(); ++i) {
        fileBytes[i] = static_cast<uint8_t>((i * 17u + 3u) & 0xFFu);
    }
    {
        std::ofstream out(filename, std::ios::binary);
        ASSERT_TRUE(out.good());
        out.write(reinterpret_cast<const char*>(fileBytes.data()), static_cast<std::streamsize>(fileBytes.size()));
        ASSERT_TRUE(out.good());
    }

    constexpr std::uint32_t offset = 11;
    std::vector<uint8_t> first(7, 0);
    std::vector<uint8_t> second(19, 0);
    std::vector<uint8_t> third(13, 0);

    iovec iovecs[] = {
        iovec{first.data(), first.size()},
        iovec{second.data(), second.size()},
        iovec{third.data(), third.size()},
    };
    constexpr std::uint32_t totalBytes = 7 + 19 + 13;

    UringDirect reader(/*queueDepth=*/2, UringDirect::IoBackend::PreadBuffered);
    reader.registerCachedLoadFile(filename);
    ASSERT_TRUE(reader.submitReadvCached(iovecs, 3, /*fileOffsetBytes=*/offset, totalBytes));
    EXPECT_EQ(reader.submit(), 1);

    auto completions = reader.waitCompletionsInOrder(1);
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_EQ(completions[0].responseCode, static_cast<int>(totalBytes));

    EXPECT_TRUE(std::equal(first.begin(), first.end(), fileBytes.begin() + offset));
    EXPECT_TRUE(std::equal(second.begin(), second.end(), fileBytes.begin() + offset + first.size()));
    EXPECT_TRUE(std::equal(third.begin(), third.end(), fileBytes.begin() + offset + first.size() + second.size()));
}

TEST(UringDirect, SubmitReadvCachedBufferedFallbackIsAsyncAndAllowsMultipleInFlight) {
    std::string filename = makeTmpPrefix("readv_cached_async_buffered");
    ScopedUnlink cleanup(filename);

    std::vector<uint8_t> fileBytes(128);
    for (std::size_t i = 0; i < fileBytes.size(); ++i) {
        fileBytes[i] = static_cast<uint8_t>((i * 29u + 7u) & 0xFFu);
    }
    {
        std::ofstream out(filename, std::ios::binary);
        ASSERT_TRUE(out.good());
        out.write(reinterpret_cast<const char*>(fileBytes.data()), static_cast<std::streamsize>(fileBytes.size()));
        ASSERT_TRUE(out.good());
    }

    std::vector<uint8_t> a0(8, 0);
    std::vector<uint8_t> a1(8, 0);
    std::vector<uint8_t> b0(5, 0);
    std::vector<uint8_t> b1(11, 0);

    iovec firstRequest[] = {
        iovec{a0.data(), a0.size()},
        iovec{a1.data(), a1.size()},
    };
    iovec secondRequest[] = {
        iovec{b0.data(), b0.size()},
        iovec{b1.data(), b1.size()},
    };

    UringDirect::testResetFallbackWorkerBlock();
    UringDirect reader(/*queueDepth=*/2, UringDirect::IoBackend::PreadBuffered);
    reader.registerCachedLoadFile(filename);

    UringDirect::testSetFallbackWorkerBlockEnabled(true);
    ASSERT_TRUE(reader.submitReadvCached(firstRequest, 2, /*fileOffsetBytes=*/3, /*totalBytes=*/16));
    ASSERT_TRUE(reader.submitReadvCached(secondRequest, 2, /*fileOffsetBytes=*/37, /*totalBytes=*/16));
    EXPECT_EQ(reader.submit(), 2);

    ASSERT_TRUE(UringDirect::testWaitForFallbackWorkerStartedCount(2, std::chrono::seconds(5)));
    EXPECT_TRUE(reader.pollCompletionsInOrder(2).empty())
        << "fallback readv completed while test worker hook was still blocked";

    UringDirect::testSetFallbackWorkerBlockEnabled(false);
    auto completions = reader.waitCompletionsInOrder(2);
    ASSERT_EQ(completions.size(), 2u);
    EXPECT_EQ(completions[0].responseCode, 16);
    EXPECT_EQ(completions[1].responseCode, 16);

    EXPECT_TRUE(std::equal(a0.begin(), a0.end(), fileBytes.begin() + 3));
    EXPECT_TRUE(std::equal(a1.begin(), a1.end(), fileBytes.begin() + 3 + a0.size()));
    EXPECT_TRUE(std::equal(b0.begin(), b0.end(), fileBytes.begin() + 37));
    EXPECT_TRUE(std::equal(b1.begin(), b1.end(), fileBytes.begin() + 37 + b0.size()));
    UringDirect::testResetFallbackWorkerBlock();
}

TEST(UringDirect, SubmitReadvCachedRejectsInvalidIovecsAndTotalByteMismatch) {
    std::string filename = makeTmpPrefix("readv_cached_validation");
    {
        std::ofstream out(filename, std::ios::binary);
        ASSERT_TRUE(out.good());
        out.write("0123456789abcdef", 16);
        ASSERT_TRUE(out.good());
    }
    ScopedUnlink cleanup(filename);

    UringDirect reader(/*queueDepth=*/1, UringDirect::IoBackend::PreadBuffered);
    reader.registerCachedLoadFile(filename);

    std::vector<uint8_t> good(4, 0);
    iovec mismatch[] = {iovec{good.data(), good.size()}};
    EXPECT_THROW(reader.submitReadvCached(mismatch, 1, /*fileOffsetBytes=*/0, /*totalBytes=*/5), std::runtime_error);

    iovec nullBase[] = {iovec{nullptr, 4}};
    EXPECT_THROW(reader.submitReadvCached(nullBase, 1, /*fileOffsetBytes=*/0, /*totalBytes=*/4), std::runtime_error);

    iovec zeroLen[] = {iovec{good.data(), 0}};
    EXPECT_THROW(reader.submitReadvCached(zeroLen, 1, /*fileOffsetBytes=*/0, /*totalBytes=*/4), std::runtime_error);
}

TEST(UringDirect, SubmitReadvCachedHandlesShortIoUringCompletions) {
    ScopedUringDirectCompatibilityTestHooks hooks(std::nullopt, false);

    std::string filename = makeTmpPrefix("readv_cached_short_uring");
    ScopedUnlink cleanup(filename);

    std::vector<uint8_t> fileBytes(128);
    for (std::size_t i = 0; i < fileBytes.size(); ++i) {
        fileBytes[i] = static_cast<uint8_t>((i * 13u + 41u) & 0xFFu);
    }
    {
        std::ofstream out(filename, std::ios::binary);
        ASSERT_TRUE(out.good());
        out.write(reinterpret_cast<const char*>(fileBytes.data()), static_cast<std::streamsize>(fileBytes.size()));
        ASSERT_TRUE(out.good());
    }

    UringDirect reader(/*queueDepth=*/8, UringDirect::IoBackend::Auto);
    if (std::string(reader.activeBackendName()) != "uring_direct") {
        GTEST_SKIP() << "io_uring unavailable in this runtime; active backend is " << reader.activeBackendName();
    }
    reader.registerCachedLoadFile(filename);

    std::vector<uint8_t> first(9, 0);
    std::vector<uint8_t> second(17, 0);
    std::vector<uint8_t> third(6, 0);
    iovec iovecs[] = {
        iovec{first.data(), first.size()},
        iovec{second.data(), second.size()},
        iovec{third.data(), third.size()},
    };

    UringDirect::testSetNextIoUringSubmissionByteLimit(11);
    ASSERT_TRUE(reader.submitReadvCached(iovecs, 3, /*fileOffsetBytes=*/23, /*totalBytes=*/32));
    EXPECT_GE(reader.submit(), 1);

    auto completions = reader.waitCompletionsInOrder(1);
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_EQ(completions[0].responseCode, 32);

    EXPECT_TRUE(std::equal(first.begin(), first.end(), fileBytes.begin() + 23));
    EXPECT_TRUE(std::equal(second.begin(), second.end(), fileBytes.begin() + 23 + first.size()));
    EXPECT_TRUE(std::equal(third.begin(), third.end(), fileBytes.begin() + 23 + first.size() + second.size()));
}

TEST(UringDirect, FinishDumpedFileRejectsUndrainedWriteCompletions) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    constexpr uint32_t kAlign = 4096;

    Tensor buffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {kAlign}), kAlign);
    uint8_t* p = buffer.getMemPtr<uint8_t>();
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p) % kAlign, 0u);
    for (uint32_t i = 0; i < kAlign; ++i) {
        p[i] = static_cast<uint8_t>(i & 0xFFu);
    }

    std::string filename = makeTmpPrefix("finish_dumped_file_pending_completion");
    ScopedUnlink cleanup(filename);

    UringDirect uring(64, UringDirect::IoBackend::PreadBuffered);
    uring.registerReusableBuffers({buffer.getMemPtr()}, {kAlign});
    uring.registerDumpFile(filename);

    ASSERT_TRUE(uring.submitWriteFixed(/*bufIndex=*/0,
                                       /*fileOffsetBytes=*/0,
                                       /*lenBytes=*/kAlign,
                                       /*bufOffsetBytes=*/0));
    EXPECT_EQ(uring.submit(), 1);

    try {
        (void)uring.finishDumpedFile(false);
        FAIL() << "finishDumpedFile should reject undrained write completions";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("pending read/write completions"), std::string::npos);
    }

    auto completions = uring.waitCompletionsInOrder(1);
    ASSERT_EQ(completions.size(), 1u);
    ASSERT_EQ(completions[0].responseCode, static_cast<int>(kAlign));

    auto fsyncCompletion = uring.finishDumpedFile(false);
    EXPECT_EQ(fsyncCompletion.responseCode, 0);
}

TEST(UringDirectCompatibility, ExplicitUringDirectDoesNotFallback) {
    ScopedUringDirectCompatibilityTestHooks hooks(-EPERM, false);

    try {
        UringDirect uring(64, UringDirect::IoBackend::UringDirect);
        FAIL() << "explicit uring_direct should fail when io_uring_queue_init is unavailable";
    } catch (const std::runtime_error& e) {
        std::string message = e.what();
        EXPECT_NE(message.find("io_uring_queue_init failed"), std::string::npos);
        EXPECT_NE(message.find("Operation not permitted"), std::string::npos);
    }
}

TEST(UringDirectCompatibility, AutoFallsBackFromUnavailableIoUringToPreadDirectAndWarns) {
    ScopedUringDirectCompatibilityTestHooks hooks(-EPERM, false);

    testing::internal::CaptureStderr();
    UringDirect uring(64, UringDirect::IoBackend::Auto);
    std::string warning = testing::internal::GetCapturedStderr();

    EXPECT_STREQ(uring.requestedBackendName(), "auto");
    EXPECT_STREQ(uring.activeBackendName(), "pread_direct");
    EXPECT_NE(warning.find("io_uring_queue_init failed"), std::string::npos);
    EXPECT_NE(warning.find("Falling back to pread_direct"), std::string::npos);
    EXPECT_NE(warning.find("Docker/dev-container workaround"), std::string::npos);
    EXPECT_NE(warning.find("Managed cloud training environments"), std::string::npos);
}

TEST(UringDirectCompatibility, CannotAllocateMemoryIsAutoFallbackAvailabilityError) {
    EXPECT_TRUE(UringDirect::testIsBackendAvailabilityErrno(ENOMEM))
        << "io_uring fixed-buffer registration can fail with ENOMEM when memlock/pinned-memory limits are too small; "
           "auto mode should try the pread backends instead of aborting.";
}

TEST(UringDirectCompatibility, AutoFallsBackFromRegisterBuffersEnomemToPreadDirectAndWarns) {
    ScopedUringDirectCompatibilityTestHooks hooks(std::nullopt, false);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    constexpr uint32_t kAlign = 4096;
    Tensor buffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {kAlign}), kAlign);

    testing::internal::CaptureStderr();
    UringDirect uring(64, UringDirect::IoBackend::Auto);
    std::string initWarning = testing::internal::GetCapturedStderr();
    if (std::string(uring.activeBackendName()) != "uring_direct") {
        GTEST_SKIP() << "io_uring unavailable in this runtime; constructor warning was: " << initWarning;
    }

    UringDirect::testResetCompatibilityWarning();
    UringDirect::testSetIoUringRegisterBuffersResult(-ENOMEM);

    testing::internal::CaptureStderr();
    uring.registerReusableBuffers({buffer.getMemPtr()}, {kAlign});
    std::string warning = testing::internal::GetCapturedStderr();

    EXPECT_STREQ(uring.requestedBackendName(), "auto");
    EXPECT_STREQ(uring.activeBackendName(), "pread_direct");
    EXPECT_TRUE(uring.buffersRegistered());
    EXPECT_EQ(uring.numRegisteredBuffers(), 1u);
    EXPECT_NE(warning.find("io_uring_register_buffers failed"), std::string::npos);
    EXPECT_NE(warning.find("Cannot allocate memory"), std::string::npos);
    EXPECT_NE(warning.find("Falling back to pread_direct"), std::string::npos);
    EXPECT_NE(warning.find("memlock or pinned-memory budget"), std::string::npos);
}

TEST(UringDirectCompatibility, AutoFallsBackFromDirectOpenFailureToBufferedPreadAndStillWrites) {
    ScopedUringDirectCompatibilityTestHooks hooks(-EPERM, true);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    constexpr uint32_t kAlign = 4096;
    Tensor buffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {kAlign}), kAlign);
    uint8_t* p = buffer.getMemPtr<uint8_t>();
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p) % kAlign, 0u);
    for (uint32_t i = 0; i < kAlign; ++i) {
        p[i] = static_cast<uint8_t>((i * 17u) & 0xFFu);
    }

    std::string filename = makeTmpPrefix("uring_auto_buffered_fallback");
    ScopedUnlink cleanup(filename);

    testing::internal::CaptureStderr();
    UringDirect uring(64, UringDirect::IoBackend::Auto);
    std::string warning = testing::internal::GetCapturedStderr();
    ASSERT_STREQ(uring.activeBackendName(), "pread_direct");
    EXPECT_NE(warning.find("Falling back to pread_direct"), std::string::npos);

    uring.registerReusableBuffers({buffer.getMemPtr()}, {kAlign});
    uring.registerDumpFile(filename);
    ASSERT_STREQ(uring.activeBackendName(), "pread_buffered");

    ASSERT_TRUE(uring.submitWriteFixed(/*bufIndex=*/0,
                                       /*fileOffsetBytes=*/0,
                                       /*lenBytes=*/kAlign,
                                       /*bufOffsetBytes=*/0));
    EXPECT_EQ(uring.submit(), 1);
    auto comps = uring.waitCompletionsInOrder(1);
    ASSERT_EQ(comps.size(), 1u);
    ASSERT_EQ(comps[0].responseCode, static_cast<int>(kAlign));
    auto fsyncCompletion = uring.finishDumpedFile(false);
    EXPECT_EQ(fsyncCompletion.responseCode, 0);

    Tensor verify(cpuPlacement, TensorDescriptor(DataType::UINT8, {kAlign}));
    readEntireFileInto(verify.getMemPtr(), kAlign, filename);
    EXPECT_EQ(std::memcmp(verify.getMemPtr(), buffer.getMemPtr(), kAlign), 0);
}


TEST(UringDirectCompatibility, ExplicitPreadDirectFixedReadWriteRoundTrip) {
    ScopedUringDirectCompatibilityTestHooks hooks(std::nullopt, false);
    runExplicitPreadBackendFixedReadWriteRoundTrip(UringDirect::IoBackend::PreadDirect, "pread_direct", "explicit_pread_direct_roundtrip");
}

TEST(UringDirectCompatibility, ExplicitPreadBufferedFixedReadWriteRoundTrip) {
    ScopedUringDirectCompatibilityTestHooks hooks(std::nullopt, false);
    runExplicitPreadBackendFixedReadWriteRoundTrip(UringDirect::IoBackend::PreadBuffered, "pread_buffered", "explicit_pread_buffered_roundtrip");
}


TEST(UringDirectCompatibility, ExplicitPreadDirectFixedReadWriteIsAsyncAndAllowsMultipleInFlight) {
    ScopedUringDirectCompatibilityTestHooks hooks(std::nullopt, false);
    runExplicitPreadBackendAsyncFixedReadWrite(UringDirect::IoBackend::PreadDirect, "pread_direct", "explicit_pread_direct_async_roundtrip");
}

TEST(UringDirectCompatibility, ExplicitPreadBufferedFixedReadWriteIsAsyncAndAllowsMultipleInFlight) {
    ScopedUringDirectCompatibilityTestHooks hooks(std::nullopt, false);
    runExplicitPreadBackendAsyncFixedReadWrite(UringDirect::IoBackend::PreadBuffered, "pread_buffered", "explicit_pread_buffered_async_roundtrip");
}
