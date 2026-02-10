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

using namespace thor_file;

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

static std::string makeArchiveName(const char* tag) { return std::string(tag) + "_" + std::to_string(::getpid()) + ".bin"; }

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

static void createBinaryFixtureFile(const std::string& path, const std::vector<FilePayloadSpec>& specs, uint32_t poisonLength = 4096) {
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
    std::vector<uint8_t> poisonPrefix(poisonLength);
    std::vector<uint8_t> poisonTail(poisonLength);

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

    std::string archiveFileName = makeArchiveName("reader_worker_fixture");
    std::string archivePath = std::string("/tmp/") + archiveFileName;
    ScopedUnlink cleanup(archivePath);
    createBinaryFixtureFile(archivePath, specs);

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

    ArchivePlanEntry planEntry0;
    planEntry0.tensor = d0;
    planEntry0.tensorOffsetBytes = 0;
    planEntry0.numBytes = bytes;
    planEntry0.fileOffsetBytes = specs[0].payloadOffset;
    planEntry0.expectedCrc = crc32_ieee(0xFFFFFFFF, expected[0].data(), (uint32_t)bytes);

    ArchivePlanEntry planEntry1;
    planEntry1.tensor = d1;
    planEntry1.tensorOffsetBytes = 0;
    planEntry1.numBytes = bytes;
    planEntry1.fileOffsetBytes = specs[1].payloadOffset;
    planEntry1.expectedCrc = crc32_ieee(0xFFFFFFFF, expected[1].data(), (uint32_t)bytes);

    ArchivePlanEntry planEntry2;
    planEntry2.tensor = d2;
    planEntry2.tensorOffsetBytes = 0;
    planEntry2.numBytes = bytes;
    planEntry2.fileOffsetBytes = specs[2].payloadOffset;
    planEntry2.expectedCrc = crc32_ieee(0xFFFFFFFF, expected[2].data(), (uint32_t)bytes);

    std::vector<ArchivePlanEntry> planEntries = {planEntry0, planEntry1, planEntry2};

    std::string errorMessage;
    std::mutex mtx;
    ArchiveReaderContext context("/tmp/", errorMessage, mtx);
    ArchiveShardReaderWorker reader(context);
    ArchiveShardPlan shardPlan(archiveFileName);
    shardPlan.entries = planEntries;
    reader.process(shardPlan);

    std::vector<uint8_t> got;
    downloadGpuTensorToCpu(d0, got);
    ASSERT_EQ(std::memcmp(got.data(), expected[0].data(), bytes), 0);

    downloadGpuTensorToCpu(d1, got);
    ASSERT_EQ(std::memcmp(got.data(), expected[1].data(), bytes), 0);

    downloadGpuTensorToCpu(d2, got);
    ASSERT_EQ(std::memcmp(got.data(), expected[2].data(), bytes), 0);

    ASSERT_TRUE(errorMessage.empty());
}

TEST(ArchiveShardReaderWorker, ReadsIntoSingleGpuTensorWithDifferentDeviceOffsets) {
    const uint64_t bytes = (1u << 20) + 7;         // ~1 MiB
    const uint64_t gap = alignUp4k(bytes) + 8197;  // stride + extra guard

    std::vector<FilePayloadSpec> specs = {
        {.payloadOffset = 0 * gap, .payloadBytes = bytes, .seed = 101},
        {.payloadOffset = 1 * gap, .payloadBytes = bytes, .seed = 202},
        {.payloadOffset = 2 * gap, .payloadBytes = bytes, .seed = 303},
    };

    std::string archiveFileName = makeArchiveName("reader_worker_fixture_offsets");
    std::string archivePath = std::string("/tmp/") + archiveFileName;
    ScopedUnlink cleanup(archivePath);
    createBinaryFixtureFile(archivePath, specs);

    std::vector<std::vector<uint8_t>> expected(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        expected[i].resize(bytes);
        fillPattern(expected[i].data(), bytes, specs[i].seed);
    }

    // One big destination tensor
    ThorImplementation::Tensor big = makeEmptyGpuTensor(gap * specs.size());

    std::vector<ArchivePlanEntry> planEntries;
    for (size_t i = 0; i < specs.size(); ++i) {
        ArchivePlanEntry planEntry;
        planEntry.tensor = big;
        planEntry.tensorOffsetBytes = gap * i;
        planEntry.numBytes = bytes;
        planEntry.fileOffsetBytes = specs[i].payloadOffset;
        planEntry.expectedCrc = crc32_ieee(0xFFFFFFFF, expected[i].data(), (uint32_t)bytes);
        planEntries.push_back(planEntry);
    }

    std::string errorMessage;
    std::mutex mtx;
    ArchiveReaderContext context("/tmp/", errorMessage, mtx);
    ArchiveShardReaderWorker reader(context);
    ArchiveShardPlan shardPlan(archiveFileName);
    shardPlan.entries = planEntries;
    reader.process(shardPlan);

    // Download big tensor and verify each slice
    std::vector<uint8_t> bigCpu;
    downloadGpuTensorToCpu(big, bigCpu);

    for (size_t i = 0; i < specs.size(); ++i) {
        const uint8_t* got = bigCpu.data() + gap * i;
        ASSERT_EQ(std::memcmp(got, expected[i].data(), bytes), 0) << "Mismatch at slice " << i;
    }

    ASSERT_TRUE(errorMessage.empty());
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

    std::string archiveFileName = makeArchiveName("reader_worker_fixture_large");
    std::string archivePath = std::string("/tmp/") + archiveFileName;
    ScopedUnlink cleanup(archivePath);
    createBinaryFixtureFile(archivePath, specs);

    std::vector<uint8_t> exp0(bytes0), exp1(bytes1);
    fillPattern(exp0.data(), bytes0, specs[0].seed);
    fillPattern(exp1.data(), bytes1, specs[1].seed);

    ThorImplementation::Tensor d0 = makeEmptyGpuTensor(bytes0);
    ThorImplementation::Tensor d1 = makeEmptyGpuTensor(bytes1);

    std::vector<ArchivePlanEntry> planEntries;
    ArchivePlanEntry planEntry0;
    planEntry0.tensor = d0;
    planEntry0.tensorOffsetBytes = 0;
    planEntry0.numBytes = bytes0;
    planEntry0.fileOffsetBytes = specs[0].payloadOffset;
    planEntry0.expectedCrc = crc32_ieee(0xFFFFFFFF, exp0.data(), (uint32_t)bytes0);

    ArchivePlanEntry planEntry1;
    planEntry1.tensor = d1;
    planEntry1.tensorOffsetBytes = 0;
    planEntry1.numBytes = bytes1;
    planEntry1.fileOffsetBytes = specs[1].payloadOffset;
    planEntry1.expectedCrc = crc32_ieee(0xFFFFFFFF, exp1.data(), (uint32_t)bytes1);

    planEntries.push_back(planEntry0);
    planEntries.push_back(planEntry1);

    std::string errorMessage;
    std::mutex mtx;
    ArchiveReaderContext context("/tmp/", errorMessage, mtx);
    ArchiveShardReaderWorker reader(context);
    ArchiveShardPlan shardPlan(archiveFileName);
    shardPlan.entries = planEntries;
    reader.process(shardPlan);

    // Spot-check start/end to avoid a full memcmp of hundreds of MB if you want:
    // But below is full verification; keep only if you like.
    std::vector<uint8_t> got;
    downloadGpuTensorToCpu(d0, got);
    ASSERT_EQ(std::memcmp(got.data(), exp0.data(), bytes0), 0);

    downloadGpuTensorToCpu(d1, got);
    ASSERT_EQ(std::memcmp(got.data(), exp1.data(), bytes1), 0);

    ASSERT_TRUE(errorMessage.empty());
}

TEST(ArchiveShardReaderWorker, ReadsFiveSmallPayloadsAtArbitraryOffsets_ExactBytesAndCrcs) {
    // 5 payloads, each 10–20 bytes, placed far enough apart to avoid overlap/poisoning issues.
    const uint64_t baseStride = 0;          // keep offsets 4K aligned (nice for O_DIRECT-ish paths)
    const uint64_t gap = baseStride + 123;  // extra guard spacing

    std::vector<FilePayloadSpec> specs = {
        {.payloadOffset = 0 * gap, .payloadBytes = 10, .seed = 11},
        {.payloadOffset = 1 * gap, .payloadBytes = 12, .seed = 22},
        {.payloadOffset = 2 * gap, .payloadBytes = 15, .seed = 33},
        {.payloadOffset = 3 * gap, .payloadBytes = 18, .seed = 44},
        {.payloadOffset = 4 * gap, .payloadBytes = 20, .seed = 55},
    };

    std::string archiveFileName = makeArchiveName("reader_worker_fixture_small5");
    std::string archivePath = std::string("/tmp/") + archiveFileName;
    ScopedUnlink cleanup(archivePath);
    createBinaryFixtureFile(archivePath, specs, 32);

    // Expected CPU payloads
    std::vector<std::vector<uint8_t>> expected(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        expected[i].resize(specs[i].payloadBytes);
        fillPattern(expected[i].data(), specs[i].payloadBytes, specs[i].seed);
    }

    // Destination GPU tensors (exact-size tensors per payload)
    ThorImplementation::Tensor d0 = makeEmptyGpuTensor(specs[0].payloadBytes);
    ThorImplementation::Tensor d1 = makeEmptyGpuTensor(specs[1].payloadBytes);
    ThorImplementation::Tensor d2 = makeEmptyGpuTensor(specs[2].payloadBytes);
    ThorImplementation::Tensor d3 = makeEmptyGpuTensor(specs[3].payloadBytes);
    ThorImplementation::Tensor d4 = makeEmptyGpuTensor(specs[4].payloadBytes);

    ArchivePlanEntry planEntry0;
    planEntry0.tensor = d0;
    planEntry0.tensorOffsetBytes = 0;
    planEntry0.numBytes = specs[0].payloadBytes;
    planEntry0.fileOffsetBytes = specs[0].payloadOffset;
    planEntry0.expectedCrc = crc32_ieee(0xFFFFFFFF, expected[0].data(), (uint32_t)specs[0].payloadBytes);

    ArchivePlanEntry planEntry1;
    planEntry1.tensor = d1;
    planEntry1.tensorOffsetBytes = 0;
    planEntry1.numBytes = specs[1].payloadBytes;
    planEntry1.fileOffsetBytes = specs[1].payloadOffset;
    planEntry1.expectedCrc = crc32_ieee(0xFFFFFFFF, expected[1].data(), (uint32_t)specs[1].payloadBytes);

    ArchivePlanEntry planEntry2;
    planEntry2.tensor = d2;
    planEntry2.tensorOffsetBytes = 0;
    planEntry2.numBytes = specs[2].payloadBytes;
    planEntry2.fileOffsetBytes = specs[2].payloadOffset;
    planEntry2.expectedCrc = crc32_ieee(0xFFFFFFFF, expected[2].data(), (uint32_t)specs[2].payloadBytes);

    ArchivePlanEntry planEntry3;
    planEntry3.tensor = d3;
    planEntry3.tensorOffsetBytes = 0;
    planEntry3.numBytes = specs[3].payloadBytes;
    planEntry3.fileOffsetBytes = specs[3].payloadOffset;
    planEntry3.expectedCrc = crc32_ieee(0xFFFFFFFF, expected[3].data(), (uint32_t)specs[3].payloadBytes);

    ArchivePlanEntry planEntry4;
    planEntry4.tensor = d4;
    planEntry4.tensorOffsetBytes = 0;
    planEntry4.numBytes = specs[4].payloadBytes;
    planEntry4.fileOffsetBytes = specs[4].payloadOffset;
    planEntry4.expectedCrc = crc32_ieee(0xFFFFFFFF, expected[4].data(), (uint32_t)specs[4].payloadBytes);

    std::vector<ArchivePlanEntry> planEntries = {planEntry0, planEntry1, planEntry2, planEntry3, planEntry4};

    std::string errorMessage;
    std::mutex mtx;
    ArchiveReaderContext context("/tmp/", errorMessage, mtx);
    ArchiveShardReaderWorker reader(context);
    ArchiveShardPlan shardPlan(archiveFileName);
    shardPlan.entries = planEntries;
    reader.process(shardPlan);

    ASSERT_TRUE(errorMessage.empty());

    std::vector<uint8_t> got;

    downloadGpuTensorToCpu(d0, got);
    ASSERT_EQ(got.size(), specs[0].payloadBytes);
    ASSERT_EQ(std::memcmp(got.data(), expected[0].data(), specs[0].payloadBytes), 0);

    downloadGpuTensorToCpu(d1, got);
    ASSERT_EQ(got.size(), specs[1].payloadBytes);
    ASSERT_EQ(std::memcmp(got.data(), expected[1].data(), specs[1].payloadBytes), 0);

    downloadGpuTensorToCpu(d2, got);
    ASSERT_EQ(got.size(), specs[2].payloadBytes);
    ASSERT_EQ(std::memcmp(got.data(), expected[2].data(), specs[2].payloadBytes), 0);

    downloadGpuTensorToCpu(d3, got);
    ASSERT_EQ(got.size(), specs[3].payloadBytes);
    ASSERT_EQ(std::memcmp(got.data(), expected[3].data(), specs[3].payloadBytes), 0);

    downloadGpuTensorToCpu(d4, got);
    ASSERT_EQ(got.size(), specs[4].payloadBytes);
    ASSERT_EQ(std::memcmp(got.data(), expected[4].data(), specs[4].payloadBytes), 0);
}
