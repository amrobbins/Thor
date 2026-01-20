#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/TarFile/Crc32.h"

#include "Utilities/TarFile/ArchiveShardReaderWorker.h"

// ---------------------- helpers ----------------------

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
    const std::string& path() const { return path_; }

   private:
    std::string path_;
};

static bool env_bool(const char* name, bool def = false) {
    const char* v = std::getenv(name);
    if (!v || !*v)
        return def;
    return (std::strcmp(v, "1") == 0) || (std::strcmp(v, "true") == 0) || (std::strcmp(v, "TRUE") == 0);
}

// Deterministic payload generator
static void fillPattern(uint8_t* p, uint64_t n, uint32_t seed) {
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t x = uint32_t(i) ^ seed;
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        p[i] = uint8_t(x);
    }
}

static uint64_t alignDown4k(uint64_t x) { return x & ~uint64_t(0xFFF); }
static uint64_t alignUp4k(uint64_t x) { return (x + 0xFFF) & ~uint64_t(0xFFF); }

static void pwriteAll(int fd, const void* buf, uint64_t n, uint64_t off) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);
    uint64_t done = 0;
    while (done < n) {
        ssize_t w = ::pwrite(fd, p + done, (size_t)(n - done), (off_t)(off + done));
        if (w < 0) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error(std::string("pwrite failed: ") + std::strerror(errno));
        }
        if (w == 0) {
            throw std::runtime_error("pwrite returned 0");
        }
        done += (uint64_t)w;
    }
}

static std::string makeTempPath(const char* tag) { return std::string("/tmp/") + tag + "_" + std::to_string(::getpid()) + ".bin"; }

static uint64_t fileSizeBytes(const std::string& path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("stat failed: " + path + ": " + std::strerror(errno));
    }
    return (uint64_t)st.st_size;
}

// Create a binary file with payload regions written at exact offsets.
// Also write “poison” bytes around the payload (prefix/tail) so wrong-offset reads are obvious.
struct FilePayloadSpec {
    uint64_t payloadOffset;  // where bytes should start (can be non-4K)
    uint64_t payloadBytes;
    uint32_t seed;
};

static void createBinaryFixtureFile(const std::string& path, const std::vector<FilePayloadSpec>& specs) {
    int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
    if (fd < 0)
        throw std::runtime_error("open failed: " + path + ": " + std::strerror(errno));

    // Ensure file large enough for the reader's aligned reads
    uint64_t maxEnd = 0;
    for (const auto& s : specs) {
        uint64_t aligned = alignDown4k(s.payloadOffset);
        uint64_t prefix = s.payloadOffset - aligned;
        uint64_t totalRead = alignUp4k(prefix + s.payloadBytes);
        maxEnd = std::max(maxEnd, aligned + totalRead);
    }
    uint64_t finalSize = alignUp4k(maxEnd);
    if (::ftruncate(fd, (off_t)finalSize) != 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error("ftruncate failed: " + path + ": " + std::strerror(e));
    }

    // Write each payload and surrounding poison.
    // Poison helps detect if offsets are wrong (you'll see CRC/content mismatch loudly).
    std::vector<uint8_t> payload;
    std::array<uint8_t, 4096> poisonPrefix{};
    std::array<uint8_t, 4096> poisonTail{};

    for (const auto& s : specs) {
        payload.resize(s.payloadBytes);
        fillPattern(payload.data(), s.payloadBytes, s.seed);

        fillPattern(poisonPrefix.data(), poisonPrefix.size(), s.seed ^ 0xA5A5A5A5u);
        fillPattern(poisonTail.data(), poisonTail.size(), s.seed ^ 0x5A5A5A5Au);

        // write some poison before payload (if room)
        uint64_t preStart = (s.payloadOffset >= poisonPrefix.size()) ? (s.payloadOffset - poisonPrefix.size()) : 0;
        uint64_t preLen = s.payloadOffset - preStart;
        if (preLen)
            pwriteAll(fd, poisonPrefix.data() + (poisonPrefix.size() - preLen), preLen, preStart);

        // payload
        pwriteAll(fd, payload.data(), s.payloadBytes, s.payloadOffset);

        // poison after payload
        uint64_t postOff = s.payloadOffset + s.payloadBytes;
        uint64_t postLen = std::min<uint64_t>(poisonTail.size(), finalSize - postOff);
        if (postLen)
            pwriteAll(fd, poisonTail.data(), postLen, postOff);
    }

    ::close(fd);
}

static ThorImplementation::Tensor makeEmptyGpuTensor(uint64_t bytes, uint32_t device = 0) {
    using namespace ThorImplementation;
    TensorPlacement gpuPlace(TensorPlacement::MemDevices::GPU, device);
    TensorDescriptor desc(TensorDescriptor::DataType::UINT8, {bytes});
    return Tensor(gpuPlace, desc);
}

static void downloadGpuTensorToCpu(const ThorImplementation::Tensor& gpu, std::vector<uint8_t>& out) {
    using namespace ThorImplementation;
    const uint64_t bytes = gpu.getDescriptor().getTotalNumElements();  // UINT8 => elements == bytes
    out.resize(bytes);

    TensorPlacement cpuPlace(TensorPlacement::MemDevices::CPU, 0);
    TensorDescriptor desc(TensorDescriptor::DataType::UINT8, {bytes});
    Tensor cpu(cpuPlace, desc, 4096);

    Stream stream = Stream::getNextDownloadStream(gpu.getPlacement().getDeviceNum());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::memcpy(out.data(), cpu.getMemPtr<uint8_t>(), bytes);
}

// ---------------------- tests ----------------------

TEST(ArchiveShardReaderWorker, ReadsThreePayloadsAtArbitraryOffsets_ExactBytesAndCrcs) {
    const uint64_t bytes = (1u << 20) + 37;        // 1 MiB + 37
    const uint64_t baseStride = alignUp4k(bytes);  // >= bytes, 4K aligned
    const uint64_t gap = baseStride + 8207;        // extra guard so poison writes can't overlap either

    std::vector<FilePayloadSpec> specs = {
        {.payloadOffset = 0 * gap, .payloadBytes = bytes, .seed = 11},
        {.payloadOffset = 1 * gap, .payloadBytes = bytes, .seed = 22},
        {.payloadOffset = 2 * gap, .payloadBytes = bytes, .seed = 33},
    };

    std::string filePath = makeTempPath("reader_worker_fixture");
    ScopedUnlink cleanup(filePath);
    createBinaryFixtureFile(filePath, specs);

    // Expected CPU payloads
    std::vector<std::vector<uint8_t>> expected(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        expected[i].resize(specs[i].payloadBytes);
        fillPattern(expected[i].data(), specs[i].payloadBytes, specs[i].seed);
    }

    // Destination GPU tensors
    ThorImplementation::Tensor d0 = makeEmptyGpuTensor(bytes);
    ThorImplementation::Tensor d1 = makeEmptyGpuTensor(bytes);
    ThorImplementation::Tensor d2 = makeEmptyGpuTensor(bytes);

    std::vector<ArchiveFileReadParams> plan;
    plan.push_back({.deviceTensor = d0, .deviceOffsetBytes = 0, .numBytes = bytes, .archivePayloadOffsetBytes = specs[0].payloadOffset});
    plan.push_back({.deviceTensor = d1, .deviceOffsetBytes = 0, .numBytes = bytes, .archivePayloadOffsetBytes = specs[1].payloadOffset});
    plan.push_back({.deviceTensor = d2, .deviceOffsetBytes = 0, .numBytes = bytes, .archivePayloadOffsetBytes = specs[2].payloadOffset});

    ArchiveShardReaderWorker reader;
    std::vector<uint32_t> rcrcs;
    reader.process(plan, filePath, rcrcs);

    std::vector<uint8_t> got;
    downloadGpuTensorToCpu(d0, got);
    ASSERT_EQ(std::memcmp(got.data(), expected[0].data(), bytes), 0);

    downloadGpuTensorToCpu(d1, got);
    ASSERT_EQ(std::memcmp(got.data(), expected[1].data(), bytes), 0);

    downloadGpuTensorToCpu(d2, got);
    ASSERT_EQ(std::memcmp(got.data(), expected[2].data(), bytes), 0);

    ASSERT_EQ(rcrcs.size(), 3u);

    EXPECT_EQ(rcrcs[0], crc32_ieee(0xFFFFFFFF, expected[0].data(), (uint32_t)bytes));
    EXPECT_EQ(rcrcs[1], crc32_ieee(0xFFFFFFFF, expected[1].data(), (uint32_t)bytes));
    EXPECT_EQ(rcrcs[2], crc32_ieee(0xFFFFFFFF, expected[2].data(), (uint32_t)bytes));
}

TEST(ArchiveShardReaderWorker, ReadsIntoSingleGpuTensorWithDifferentDeviceOffsets) {
    const uint64_t bytes = (1u << 20) + 7;         // ~1 MiB
    const uint64_t gap = alignUp4k(bytes) + 8197;  // stride + extra guard

    std::vector<FilePayloadSpec> specs = {
        {.payloadOffset = 0 * gap, .payloadBytes = bytes, .seed = 101},
        {.payloadOffset = 1 * gap, .payloadBytes = bytes, .seed = 202},
        {.payloadOffset = 2 * gap, .payloadBytes = bytes, .seed = 303},
    };

    std::string filePath = makeTempPath("reader_worker_fixture_offsets");
    ScopedUnlink cleanup(filePath);
    createBinaryFixtureFile(filePath, specs);

    std::vector<std::vector<uint8_t>> expected(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        expected[i].resize(bytes);
        fillPattern(expected[i].data(), bytes, specs[i].seed);
    }

    // One big destination tensor
    ThorImplementation::Tensor big = makeEmptyGpuTensor(gap * specs.size());

    std::vector<ArchiveFileReadParams> plan;
    for (size_t i = 0; i < specs.size(); ++i) {
        plan.push_back(
            {.deviceTensor = big, .deviceOffsetBytes = gap * i, .numBytes = bytes, .archivePayloadOffsetBytes = specs[i].payloadOffset});
    }

    ArchiveShardReaderWorker reader;
    std::vector<uint32_t> rcrcs;
    reader.process(plan, filePath, rcrcs);

    // Download big tensor and verify each slice
    std::vector<uint8_t> bigCpu;
    downloadGpuTensorToCpu(big, bigCpu);

    for (size_t i = 0; i < specs.size(); ++i) {
        const uint8_t* got = bigCpu.data() + gap * i;
        ASSERT_EQ(std::memcmp(got, expected[i].data(), bytes), 0) << "Mismatch at slice " << i;
    }

    ASSERT_EQ(rcrcs.size(), specs.size());

    for (size_t i = 0; i < specs.size(); ++i) {
        EXPECT_EQ(rcrcs[i], crc32_ieee(0xFFFFFFFF, expected[i].data(), (uint32_t)bytes));
    }
}

TEST(ArchiveShardReaderWorker, LargePayloads) {
    const uint64_t bytes0 = 100'000'000ULL;
    const uint64_t bytes1 = (uint64_t(1) << 28) + 3;  // 256 MiB + 3

    const uint64_t guard = 8192;  // extra poison/guard room
    const uint64_t gap0 = alignUp4k(bytes0) + guard;
    const uint64_t off0 = 512;
    const uint64_t off1 = off0 + gap0 + 123;

    std::vector<FilePayloadSpec> specs = {
        {.payloadOffset = off0, .payloadBytes = bytes0, .seed = 777},
        {.payloadOffset = off1, .payloadBytes = bytes1, .seed = 888},
    };

    std::string filePath = makeTempPath("reader_worker_fixture_large");
    ScopedUnlink cleanup(filePath);
    createBinaryFixtureFile(filePath, specs);

    std::vector<uint8_t> exp0(bytes0), exp1(bytes1);
    fillPattern(exp0.data(), bytes0, specs[0].seed);
    fillPattern(exp1.data(), bytes1, specs[1].seed);

    ThorImplementation::Tensor d0 = makeEmptyGpuTensor(bytes0);
    ThorImplementation::Tensor d1 = makeEmptyGpuTensor(bytes1);

    std::vector<ArchiveFileReadParams> plan;
    plan.push_back({.deviceTensor = d0, .deviceOffsetBytes = 0, .numBytes = bytes0, .archivePayloadOffsetBytes = specs[0].payloadOffset});
    plan.push_back({.deviceTensor = d1, .deviceOffsetBytes = 0, .numBytes = bytes1, .archivePayloadOffsetBytes = specs[1].payloadOffset});

    ArchiveShardReaderWorker reader;
    std::vector<uint32_t> rcrcs;
    reader.process(plan, filePath, rcrcs);

    ASSERT_EQ(rcrcs.size(), 2u);
    EXPECT_EQ(rcrcs[0], crc32_ieee(0xFFFFFFFF, exp0.data(), (uint32_t)bytes0));
    EXPECT_EQ(rcrcs[1], crc32_ieee(0xFFFFFFFF, exp1.data(), (uint32_t)bytes1));

    // Spot-check start/end to avoid a full memcmp of hundreds of MB if you want:
    // But below is full verification; keep only if you like.
    std::vector<uint8_t> got;
    downloadGpuTensorToCpu(d0, got);
    ASSERT_EQ(std::memcmp(got.data(), exp0.data(), bytes0), 0);

    downloadGpuTensorToCpu(d1, got);
    ASSERT_EQ(std::memcmp(got.data(), exp1.data(), bytes1), 0);
}
