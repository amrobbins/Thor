#include <gtest/gtest.h>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TarFile/UringDirect.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    TensorDescriptor bufferDescriptor(TensorDescriptor::DataType::UINT8, {bufferSize});

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
    TensorDescriptor bufferDescriptor(TensorDescriptor::DataType::UINT8, {bufferSize});

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

    Tensor verifyBuffer(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {4 * sixteenMegs}));
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
    TensorDescriptor bufferDescriptor(TensorDescriptor::DataType::UINT8, {bufferSize});

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
    reader.registerReadFile(filename);

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
    Tensor verifyBuffer(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {4 * sixteenMegs}), 4096);

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
    TensorDescriptor bufDesc(TensorDescriptor::DataType::UINT8, {chunkBytes});

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
        ur.registerReadFile(filename);
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

    TensorDescriptor desc(TensorDescriptor::DataType::UINT8, {kBytes});
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
    Tensor verify(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {kBytes}));
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
