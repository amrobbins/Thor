// End-to-end tests for ArchiveShardWriterWorker::process(...):
//  - Builds GPU tensors with deterministic payloads
//  - Calls process() to write a TAR archive shard via UringDirect (O_DIRECT)
//  - Validates the resulting TAR with system `tar`:
//      * tar -tf lists entries
//      * tar -xOf extracts payload bytes exactly matching the source buffers
//  - Validates returned CRCs match computed CRC32c of each payload
//  - Validates output file ends on 4KB boundary (O_DIRECT-friendly)
//
// This test suite SKIPS unless RUN_TAR_INTEROP=1 is set.
// It also SKIPS if no CUDA device is available.
//
// Adjust include paths to match your repo layout.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <unistd.h>

#include <cuda_runtime.h>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/TarFile/Crc32c.h"
#include "Utilities/TarFile/UringDirect.h"

#include "Utilities/TarFile/ArchiveShardWriterWorker.h"

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

static std::string shellEscapeDoubleQuoted(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '\\' || c == '"')
            out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

static std::string runCommandCaptureStdout(const std::string& cmd, int* exitCodeOut = nullptr) {
    std::array<char, 4096> buf{};
    std::string out;

    FILE* pipe = ::popen(cmd.c_str(), "r");
    if (!pipe)
        throw std::runtime_error("popen failed for: " + cmd);

    while (!std::feof(pipe)) {
        size_t n = std::fread(buf.data(), 1, buf.size(), pipe);
        if (n)
            out.append(buf.data(), n);
    }
    int rc = ::pclose(pipe);
    if (exitCodeOut)
        *exitCodeOut = rc;
    return out;
}

static uint64_t fileSizeBytes(const std::string& path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("stat failed: " + path + ": " + std::strerror(errno));
    }
    return static_cast<uint64_t>(st.st_size);
}

// Deterministic payload generator: byte[i] = (i + seed) & 0xFF
static void fillPattern(uint8_t* p, uint64_t n, uint32_t seed) {
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t x = uint32_t(i) ^ seed;
        // simple mixing (not crypto, just non-trivial)
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        p[i] = uint8_t(x);
    }
}

// Create a GPU tensor containing payload bytes, and also return a CPU copy for verification.
static ThorImplementation::Tensor makeGpuTensorWithPayload(uint64_t bytes, uint8_t seed, std::vector<uint8_t>& cpuOut) {
    using namespace ThorImplementation;

    cpuOut.resize(bytes);

    // Fill CPU reference buffer
    fillPattern(cpuOut.data(), bytes, seed);

    // CPU tensor (source)
    TensorPlacement cpuPlace(TensorPlacement::MemDevices::CPU, 0);
    TensorDescriptor desc(TensorDescriptor::DataType::UINT8, {bytes});
    Tensor cpu(cpuPlace, desc, 4096);
    std::memcpy(cpu.getMemPtr<uint8_t>(), cpuOut.data(), bytes);

    // GPU tensor (deviceTensor)
    TensorPlacement gpuPlace(TensorPlacement::MemDevices::GPU, 0);
    Tensor gpu(gpuPlace, desc);

    // Upload whole payload. Adjust if your Tensor API differs.
    // Most of your code uses downloadSection(cpuBuffer.downloadSection(deviceTensor,...)),
    // so uploadSection should exist symmetrically.
    // If your method name differs, swap it here.
    Stream stream = Stream::getNextUploadStream(0);
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();

    return gpu;
}

static bool listingHasLine(const std::string& listing, const std::string& needle) {
    std::istringstream iss(listing);
    std::string line;
    while (std::getline(iss, line)) {
        // tar -t tends to print paths without leading "./" when created that way
        if (line == needle)
            return true;
    }
    return false;
}

TEST(ArchiveShardWriterWorker, SingleFile_WritesValidTarAndCorrectPayloadAndCrc) {
    // if (!env_bool("RUN_TAR_INTEROP", false)) {
    //     GTEST_SKIP() << "Set RUN_TAR_INTEROP=1 to run tar interop tests";
    // }

    // Require tar
    {
        int rc = 0;
        (void)runCommandCaptureStdout("tar --version >/dev/null 2>&1", &rc);
        if (rc != 0)
            GTEST_SKIP() << "`tar` not available on PATH";
    }

    // Require CUDA device
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount <= 0) {
        GTEST_SKIP() << "No CUDA device available";
    }

    using namespace ThorImplementation;

    const uint64_t bytes = uint64_t(1) << 24;  // 16 MiB (big enough to force 4K-aligned dumping)
    const std::string pathInTar = "dir/file_single.bin";

    std::vector<uint8_t> expected;
    Tensor deviceTensor = makeGpuTensorWithPayload(bytes, /*seed=*/7, expected);

    std::vector<ArchiveFileWriteParams> plan;
    plan.push_back(ArchiveFileWriteParams{
        .deviceTensor = deviceTensor,
        .offsetBytes = 0,
        .numBytes = bytes,  // this shard’s bytes
        .path_in_tar = pathInTar,
    });

    std::string archivePath = "/tmp/archive_shard_single_" + std::to_string(::getpid()) + ".tar";
    ScopedUnlink cleanup(archivePath);

    ArchiveShardWriterWorker worker;
    std::vector<uint32_t> crcs;
    worker.process(plan, archivePath, crcs);

    ASSERT_EQ(crcs.size(), plan.size());
    EXPECT_EQ(crcs[0], thor_file::Crc32c::compute(expected.data(), bytes));

    // Archive should end 4KB aligned (your appendTarEndOfArchive ensures this)
    ASSERT_EQ(fileSizeBytes(archivePath) & (4096u - 1u), 0u);

    // tar -tf lists entry
    {
        std::string cmd = "tar -tf " + shellEscapeDoubleQuoted(archivePath);
        int rc = 0;
        std::string listing = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -tf failed\ncmd: " << cmd << "\noutput:\n" << listing;
        EXPECT_TRUE(listingHasLine(listing, pathInTar)) << "missing " << pathInTar << "\nlisting:\n" << listing;
    }

    // tar -xOf extracts exact payload
    {
        std::string cmd = "tar -xOf " + shellEscapeDoubleQuoted(archivePath) + " " + shellEscapeDoubleQuoted(pathInTar);
        int rc = 0;
        std::string data = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -xOf failed\ncmd: " << cmd;

        ASSERT_EQ(data.size(), expected.size());
        ASSERT_EQ(std::memcmp(data.data(), expected.data(), expected.size()), 0);
    }
}

TEST(ArchiveShardWriterWorker, MultiFile_IncludingOverlongName_ValidTarAndCorrectPayloadsAndCrcs) {
    // if (!env_bool("RUN_TAR_INTEROP", false)) {
    //     GTEST_SKIP() << "Set RUN_TAR_INTEROP=1 to run tar interop tests";
    // }

    // Require tar
    {
        int rc = 0;
        (void)runCommandCaptureStdout("tar --version >/dev/null 2>&1", &rc);
        if (rc != 0)
            GTEST_SKIP() << "`tar` not available on PATH";
    }

    // Require CUDA device
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount <= 0) {
        GTEST_SKIP() << "No CUDA device available";
    }

    using namespace ThorImplementation;

    const uint64_t bytes = (uint64_t(1) << 24) + 123;  // 16 MiB + 123 (exercises TAR 512 padding)
    const std::string path0 = "a/b/file0.bin";
    const std::string path1 = std::string(300, 'x');  // forces PAX path (tar -xOf still works)
    const std::string path2 = "c/d/file2.bin";

    std::vector<uint8_t> expected0, expected1, expected2;

    Tensor t0 = makeGpuTensorWithPayload(bytes, /*seed=*/11, expected0);
    Tensor t1 = makeGpuTensorWithPayload(bytes, /*seed=*/22, expected1);
    Tensor t2 = makeGpuTensorWithPayload(bytes, /*seed=*/33, expected2);

    std::vector<ArchiveFileWriteParams> plan;
    plan.push_back(ArchiveFileWriteParams{
        .deviceTensor = t0,
        .offsetBytes = 0,
        .numBytes = bytes,
        .path_in_tar = path0,
    });
    plan.push_back(ArchiveFileWriteParams{
        .deviceTensor = t1,
        .offsetBytes = 0,
        .numBytes = bytes,
        .path_in_tar = path1,
    });
    plan.push_back(ArchiveFileWriteParams{
        .deviceTensor = t2,
        .offsetBytes = 0,
        .numBytes = bytes,
        .path_in_tar = path2,
    });

    std::string archivePath = "/tmp/archive_shard_multi_" + std::to_string(::getpid()) + ".tar";
    ScopedUnlink cleanup(archivePath);

    ArchiveShardWriterWorker worker;
    std::vector<uint32_t> crcs;
    worker.process(plan, archivePath, crcs);

    ASSERT_EQ(crcs.size(), plan.size());
    EXPECT_EQ(crcs[0], thor_file::Crc32c::compute(expected0.data(), bytes));
    EXPECT_EQ(crcs[1], thor_file::Crc32c::compute(expected1.data(), bytes));
    EXPECT_EQ(crcs[2], thor_file::Crc32c::compute(expected2.data(), bytes));

    // 4KB-aligned final size
    ASSERT_EQ(fileSizeBytes(archivePath) & (4096u - 1u), 0u);

    // tar -tf lists entries (including the PAX long path)
    {
        std::string cmd = "tar -tf " + shellEscapeDoubleQuoted(archivePath);
        int rc = 0;
        std::string listing = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -tf failed\ncmd: " << cmd << "\noutput:\n" << listing;

        EXPECT_TRUE(listingHasLine(listing, path0)) << "missing " << path0 << "\nlisting:\n" << listing;
        EXPECT_TRUE(listingHasLine(listing, path1)) << "missing long PAX path\nlisting:\n" << listing;
        EXPECT_TRUE(listingHasLine(listing, path2)) << "missing " << path2 << "\nlisting:\n" << listing;
    }

    auto extractAndCompare = [&](const std::string& path, const std::vector<uint8_t>& expected) {
        std::string cmd = "tar -xOf " + shellEscapeDoubleQuoted(archivePath) + " " + shellEscapeDoubleQuoted(path);
        int rc = 0;
        std::string data = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -xOf failed\ncmd: " << cmd;
        ASSERT_EQ(data.size(), expected.size());
        ASSERT_EQ(std::memcmp(data.data(), expected.data(), expected.size()), 0);
    };

    extractAndCompare(path0, expected0);
    extractAndCompare(path1, expected1);
    extractAndCompare(path2, expected2);
}

TEST(ArchiveShardWriterWorker, SingleFile_ExactlyFiveMillionBytes_WritesValidTarAndCorrectPayloadAndCrc) {
    // if (!env_bool("RUN_TAR_INTEROP", false)) {
    //     GTEST_SKIP() << "Set RUN_TAR_INTEROP=1 to run tar interop tests";
    // }

    // Require tar
    {
        int rc = 0;
        (void)runCommandCaptureStdout("tar --version >/dev/null 2>&1", &rc);
        if (rc != 0)
            GTEST_SKIP() << "`tar` not available on PATH";
    }

    // Require CUDA device
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount <= 0) {
        GTEST_SKIP() << "No CUDA device available";
    }

    using namespace ThorImplementation;

    const uint64_t bytes = 5'000'000ULL;  // exactly 5 million bytes (not 512- or 4K-aligned)
    const std::string pathInTar = "dir/file_5M.bin";

    std::vector<uint8_t> expected;
    Tensor deviceTensor = makeGpuTensorWithPayload(bytes, /*seed=*/99, expected);

    std::vector<ArchiveFileWriteParams> plan;
    plan.push_back(ArchiveFileWriteParams{
        .deviceTensor = deviceTensor,
        .offsetBytes = 0,
        .numBytes = bytes,
        .path_in_tar = pathInTar,
    });

    std::string archivePath = "/tmp/archive_shard_5m_" + std::to_string(::getpid()) + ".tar";
    ScopedUnlink cleanup(archivePath);

    ArchiveShardWriterWorker worker;
    std::vector<uint32_t> crcs;
    worker.process(plan, archivePath, crcs);

    ASSERT_EQ(crcs.size(), 1u);
    EXPECT_EQ(crcs[0], thor_file::Crc32c::compute(expected.data(), bytes));

    // Your writer should still end on a 4KB boundary due to appendTarEndOfArchive(...)
    ASSERT_EQ(fileSizeBytes(archivePath) & (4096u - 1u), 0u);

    // tar -tf lists entry
    {
        std::string cmd = "tar -tf " + shellEscapeDoubleQuoted(archivePath);
        int rc = 0;
        std::string listing = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -tf failed\ncmd: " << cmd << "\noutput:\n" << listing;
        EXPECT_TRUE(listingHasLine(listing, pathInTar)) << "missing " << pathInTar << "\nlisting:\n" << listing;
    }

    // tar -xOf extracts exact payload (verifies 512-padding correctness too)
    {
        std::string cmd = "tar -xOf " + shellEscapeDoubleQuoted(archivePath) + " " + shellEscapeDoubleQuoted(pathInTar);
        int rc = 0;
        std::string data = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -xOf failed\ncmd: " << cmd;

        ASSERT_EQ(data.size(), expected.size());
        ASSERT_EQ(std::memcmp(data.data(), expected.data(), expected.size()), 0);
    }
}

TEST(ArchiveShardWriterWorker, TwoFiles_100MB_And_500MB_WritesValidTarAndCorrectPayloadsAndCrcs) {
    // if (!env_bool("RUN_TAR_INTEROP", false)) {
    //     GTEST_SKIP() << "Set RUN_TAR_INTEROP=1 to run tar interop tests";
    // }

    // Require tar
    {
        int rc = 0;
        (void)runCommandCaptureStdout("tar --version >/dev/null 2>&1", &rc);
        if (rc != 0)
            GTEST_SKIP() << "`tar` not available on PATH";
    }

    // Require CUDA device
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount <= 0) {
        GTEST_SKIP() << "No CUDA device available";
    }

    using namespace ThorImplementation;

    const uint64_t oneHundredMB = 100'000'000ULL;
    const uint64_t fiveHundredMB = (uint64_t(1) << 29);  // 536,870,912

    const std::string path0 = "big/file_100m.bin";
    const std::string path1 = "big/file_500m.bin";

    std::vector<uint8_t> expected0;
    std::vector<uint8_t> expected1;

    Tensor t0 = makeGpuTensorWithPayload(oneHundredMB, /*seed=*/17, expected0);
    Tensor t1 = makeGpuTensorWithPayload(fiveHundredMB, /*seed=*/91, expected1);

    std::vector<ArchiveFileWriteParams> plan;
    plan.push_back(ArchiveFileWriteParams{
        .deviceTensor = t0,
        .offsetBytes = 0,
        .numBytes = oneHundredMB,
        .path_in_tar = path0,
    });
    plan.push_back(ArchiveFileWriteParams{
        .deviceTensor = t1,
        .offsetBytes = 0,
        .numBytes = fiveHundredMB,
        .path_in_tar = path1,
    });

    std::string archivePath = "/tmp/archive_shard_100m_500m_" + std::to_string(::getpid()) + ".tar";
    ScopedUnlink cleanup(archivePath);

    ArchiveShardWriterWorker worker;
    std::vector<uint32_t> crcs;
    worker.process(plan, archivePath, crcs);

    ASSERT_EQ(crcs.size(), 2u);
    EXPECT_EQ(crcs[0], thor_file::Crc32c::compute(expected0.data(), oneHundredMB));
    EXPECT_EQ(crcs[1], thor_file::Crc32c::compute(expected1.data(), fiveHundredMB));

    // Should end on a 4KB boundary due to appendTarEndOfArchive(...)
    ASSERT_EQ(fileSizeBytes(archivePath) & (4096u - 1u), 0u);

    // tar -tf lists both entries
    {
        std::string cmd = "tar -tf " + shellEscapeDoubleQuoted(archivePath);
        int rc = 0;
        std::string listing = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -tf failed\ncmd: " << cmd << "\noutput:\n" << listing;

        EXPECT_TRUE(listingHasLine(listing, path0)) << "missing " << path0 << "\nlisting:\n" << listing;
        EXPECT_TRUE(listingHasLine(listing, path1)) << "missing " << path1 << "\nlisting:\n" << listing;
    }

    // tar -xOf extracts exact payloads (this can take a bit for 500MB, so keep it interop-gated)
    auto extractAndCompare = [&](const std::string& path, const std::vector<uint8_t>& expected) {
        std::string cmd = "tar -xOf " + shellEscapeDoubleQuoted(archivePath) + " " + shellEscapeDoubleQuoted(path);
        int rc = 0;
        std::string data = runCommandCaptureStdout(cmd, &rc);
        ASSERT_EQ(rc, 0) << "tar -xOf failed\ncmd: " << cmd;
        ASSERT_EQ(data.size(), expected.size());
        ASSERT_EQ(std::memcmp(data.data(), expected.data(), expected.size()), 0);
    };

    extractAndCompare(path0, expected0);
    extractAndCompare(path1, expected1);
}
