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

    uringDirect.submitWriteFixed(0, 0, sixteenMegs, 123);
    uringDirect.submitWriteFixed(1, sixteenMegs, sixteenMegs, 456);
    uringDirect.submitWriteFixed(2, 2 * sixteenMegs, sixteenMegs, 789);
    uringDirect.submitWriteFixed(3, 3 * sixteenMegs, sixteenMegs, 1011);
    uringDirect.submit();
    auto comps = uringDirect.waitCompletions(4);

    for (auto& c : comps) {
        ASSERT_GE(c.responseCode, 0) << "write failed: userData=" << c.userData << " res=" << c.responseCode
                                     << " errno=" << -c.responseCode;
        // For your case each should be exactly fiveHundredMB
        ASSERT_EQ(static_cast<uint64_t>(c.responseCode), sixteenMegs) << "short write: userData=" << c.userData;
    }
    ASSERT_EQ(fileSizeBytes(filename), 4 * sixteenMegs);

    uringDirect.finishDumpedFile(999, false);

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
}
