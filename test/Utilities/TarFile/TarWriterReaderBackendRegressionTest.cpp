#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/TarWriter.h"
#include "Utilities/TarFile/UringDirect.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;
using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

class ScopedEnvVar {
   public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        const char* previous = std::getenv(name_.c_str());
        if (previous != nullptr) {
            hadPrevious_ = true;
            previous_ = previous;
        }
        ::setenv(name_.c_str(), value, /*overwrite=*/1);
    }

    ~ScopedEnvVar() {
        if (hadPrevious_) {
            ::setenv(name_.c_str(), previous_.c_str(), /*overwrite=*/1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

   private:
    std::string name_;
    bool hadPrevious_ = false;
    std::string previous_;
};

class ScopedDirectory {
   public:
    explicit ScopedDirectory(fs::path path) : path_(std::move(path)) {
        fs::remove_all(path_);
        fs::create_directories(path_);
    }

    ~ScopedDirectory() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    const fs::path& path() const { return path_; }

    ScopedDirectory(const ScopedDirectory&) = delete;
    ScopedDirectory& operator=(const ScopedDirectory&) = delete;

   private:
    fs::path path_;
};

class ScopedUringDirectShortIoHooks {
   public:
    ScopedUringDirectShortIoHooks() { UringDirect::testResetIoUringShortIoHooks(); }
    ~ScopedUringDirectShortIoHooks() { UringDirect::testResetIoUringShortIoHooks(); }

    ScopedUringDirectShortIoHooks(const ScopedUringDirectShortIoHooks&) = delete;
    ScopedUringDirectShortIoHooks& operator=(const ScopedUringDirectShortIoHooks&) = delete;
};

struct BackendCase {
    const char* envValue;
    const char* displayName;
};

struct TensorSpec {
    std::string pathInTar;
    uint64_t numBytes;
    uint32_t seed;
};

fs::path uniqueTempDir(const std::string& stem) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return fs::temp_directory_path() / (stem + "_" + std::to_string(::getpid()) + "_" + std::to_string(now));
}

void requireCudaDeviceOrSkip() {
    int deviceCount = 0;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount <= 0) {
        GTEST_SKIP() << "TarWriter/TarReader backend CRC regression requires a CUDA device.";
    }
}

uint8_t deterministicByte(uint64_t index, uint32_t seed) {
    uint32_t x = static_cast<uint32_t>(index) ^ (seed * 0x9E3779B9u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return static_cast<uint8_t>(x);
}

void fillDeterministic(uint8_t* data, uint64_t numBytes, uint32_t seed) {
    for (uint64_t i = 0; i < numBytes; ++i) {
        data[i] = deterministicByte(i, seed);
    }
}

void expectDeterministicBytes(const uint8_t* data, uint64_t numBytes, uint32_t seed, const std::string& pathInTar) {
    for (uint64_t i = 0; i < numBytes; ++i) {
        const uint8_t expected = deterministicByte(i, seed);
        if (data[i] != expected) {
            FAIL() << "payload mismatch for " << pathInTar << " at byte " << i << ": expected " << static_cast<uint32_t>(expected)
                   << " actual " << static_cast<uint32_t>(data[i]);
        }
    }
}

Tensor makeGpuUint8Tensor(uint64_t numBytes, uint32_t seed) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::UINT8, {numBytes});

    Tensor cpu(cpuPlacement, descriptor, 4096);
    fillDeterministic(cpu.getMemPtr<uint8_t>(), numBytes, seed);

    Tensor gpu(gpuPlacement, descriptor);
    Stream stream = Stream::getNextUploadStream(0);
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<uint8_t> copyGpuTensorToCpuBytes(const Tensor& gpu) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    Tensor cpu = gpu.clone(cpuPlacement);
    Stream stream = Stream::getNextDownloadStream(0);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<uint8_t> bytes(gpu.getArraySizeInBytes());
    std::memcpy(bytes.data(), cpu.getMemPtr<uint8_t>(), bytes.size());
    return bytes;
}

bool isUnavailableExplicitUringDirect(const BackendCase& backend, const std::string& message) {
    if (std::string(backend.envValue) != "uring_direct") {
        return false;
    }
    return message.find("io_uring_queue_init failed") != std::string::npos ||
           message.find("io_uring_register_buffers failed") != std::string::npos ||
           message.find("io_uring_register_files failed") != std::string::npos || message.find("open(O_DIRECT") != std::string::npos;
}

std::vector<TensorSpec> archiveRegressionTensorSpecs() {
    // This intentionally resembles saved model artifacts: many named tensor payloads,
    // mixed unaligned sizes, and several payloads larger than UringDirect's 16 MiB
    // per-op chunk size.  The io_uring CRC bug observed in SkuForecaster artifacts
    // manifested as a TarReader content CRC mismatch for one layer parameter file.
    return {
        {"layer4300_weights_parameter_weights.gds", 257'003ULL, 11},
        {"layer4300_biases_parameter_weights.gds", 5'000'000ULL, 12},
        {"layer4311_weights_parameter_weights.gds", 1'048'576ULL + 123ULL, 13},
        {"layer4311_biases_parameter_weights.gds", 333'337ULL, 14},
        {"layer4322_weights_parameter_weights.gds", 16'777'216ULL + 123ULL, 15},
        {"layer4322_biases_parameter_weights.gds", 65'539ULL, 16},
        {"layer4333_weights_parameter_weights.gds", 3'145'729ULL, 17},
        {"layer4333_biases_parameter_weights.gds", 786'461ULL, 18},
        {"layer4344_weights_parameter_weights.gds", 16'777'216ULL + 7'777ULL, 19},
        {"layer4344_biases_parameter_weights.gds", 98'309ULL, 20},
        {"layer4357_weights_parameter_weights.gds", 5'000'000ULL, 21},
        {"layer4357_biases_parameter_weights.gds", 524'309ULL, 22},
        {"layer4368_weights_parameter_weights.gds", 16'777'216ULL + 31ULL, 23},
        {"layer4368_biases_parameter_weights.gds", 131'101ULL, 24},
        {"layer4379_weights_parameter_weights.gds", 8'388'608ULL + 511ULL, 25},
        {"layer4379_biases_parameter_weights.gds", 262'153ULL, 26},
        {"layer4388_weights_parameter_weights.gds", 2'097'191ULL, 27},
        {"layer4388_biases_parameter_weights.gds", 17'411ULL, 28},
    };
}

std::vector<size_t> sequentialReadOrder(size_t count) {
    std::vector<size_t> order(count);
    for (size_t i = 0; i < count; ++i) {
        order[i] = i;
    }
    return order;
}

std::vector<size_t> networkLikeScrambledReadOrder(size_t count) {
    // Network placement does not promise to request tensors in archive-offset order.
    // Use a deterministic mixed order that alternates late, early, and middle
    // entries so the reader repeatedly changes file offsets while its two-buffer
    // pipeline is active.  The original SkuForecaster CRC failure occurred during
    // Network.place(), not during a simple same-order TarReader round trip.
    std::vector<size_t> order;
    order.reserve(count);
    size_t low = 0;
    size_t high = count;
    while (low < high) {
        --high;
        order.push_back(high);
        if (low < high) {
            order.push_back(low);
            ++low;
        }
    }
    if (count > 6) {
        std::rotate(order.begin() + 2, order.begin() + (order.size() / 2), order.end() - 1);
    }
    return order;
}

void runTarArchiveRoundTripForBackend(const BackendCase& backend,
                                      const std::vector<size_t>& readOrder,
                                      uint32_t repetitions,
                                      const std::string& caseName,
                                      uint64_t archiveShardSizeLimitBytes = 12'000'000ULL) {
    SCOPED_TRACE(std::string("backend=") + backend.displayName);
    SCOPED_TRACE("case=" + caseName);
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", backend.envValue);

    const std::vector<TensorSpec> specs = archiveRegressionTensorSpecs();
    ASSERT_EQ(readOrder.size(), specs.size());
    for (size_t index : readOrder) {
        ASSERT_LT(index, specs.size());
    }

    for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
        SCOPED_TRACE("repetition=" + std::to_string(repetition));
        ScopedDirectory archiveDir(
            uniqueTempDir(std::string("thor_tar_crc_regression_") + backend.envValue + "_" + caseName + "_" + std::to_string(repetition)));

        const std::string archiveName = "sku_forecaster_artifact_crc_regression";

        std::vector<Tensor> sourceTensors;
        sourceTensors.reserve(specs.size());
        thor_file::TarWriter writer(archiveName,
                                    archiveShardSizeLimitBytes,
                                    /*fileShardSizeLimitBytes=*/500'000'000ULL);
        for (const TensorSpec& spec : specs) {
            sourceTensors.push_back(makeGpuUint8Tensor(spec.numBytes, spec.seed + repetition * 101));
            writer.addArchiveFile(spec.pathInTar, sourceTensors.back());
        }

        try {
            writer.createArchive(archiveDir.path(), /*overwriteIfExists=*/true);
        } catch (const std::runtime_error& e) {
            if (isUnavailableExplicitUringDirect(backend, e.what())) {
                GTEST_SKIP() << "Explicit uring_direct backend is unavailable in this runtime: " << e.what();
            }
            throw;
        }

        thor_file::TarReader reader(archiveName, archiveDir.path());
        std::vector<Tensor> loadedTensors;
        loadedTensors.reserve(specs.size());
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
        for (uint64_t i = 0; i < specs.size(); ++i) {
            const TensorSpec& spec = specs[i];
            ASSERT_TRUE(reader.containsFile(spec.pathInTar));
            ASSERT_EQ(reader.getFileSize(spec.pathInTar), spec.numBytes);
            loadedTensors.emplace_back(gpuPlacement, TensorDescriptor(DataType::UINT8, {spec.numBytes}));
        }
        for (size_t specIndex : readOrder) {
            reader.registerReadRequest(specs[specIndex].pathInTar, loadedTensors[specIndex]);
        }

        try {
            reader.executeReadRequests();
        } catch (const std::runtime_error& e) {
            if (isUnavailableExplicitUringDirect(backend, e.what())) {
                GTEST_SKIP() << "Explicit uring_direct backend is unavailable in this runtime: " << e.what();
            }
            throw;
        }

        for (uint64_t i = 0; i < specs.size(); ++i) {
            std::vector<uint8_t> actual = copyGpuTensorToCpuBytes(loadedTensors[i]);
            ASSERT_EQ(actual.size(), specs[i].numBytes) << specs[i].pathInTar;
            expectDeterministicBytes(actual.data(), actual.size(), specs[i].seed + repetition * 101, specs[i].pathInTar);
        }
    }
}

enum class TarShortIoFault {
    ShortWriteSubmission,
    ShortReadSubmission,
};

void runFaultInjectedSingleShardRoundTrip(TarShortIoFault fault) {
    requireCudaDeviceOrSkip();
    ScopedUringDirectShortIoHooks hooks;

    const std::vector<TensorSpec> specs = archiveRegressionTensorSpecs();
    const std::vector<size_t> readOrder = networkLikeScrambledReadOrder(specs.size());
    const std::string caseName = fault == TarShortIoFault::ShortWriteSubmission ? "short_write_submission" : "short_read_submission";
    ScopedDirectory archiveDir(uniqueTempDir("thor_tar_crc_regression_" + caseName));
    const std::string archiveName = "sku_forecaster_artifact_crc_regression";

    std::vector<Tensor> sourceTensors;
    sourceTensors.reserve(specs.size());

    try {
        {
            ScopedEnvVar writeBackend("THOR_IO_BACKEND", fault == TarShortIoFault::ShortWriteSubmission ? "uring_direct" : "pread_direct");
            thor_file::TarWriter writer(archiveName,
                                        /*archiveShardSizeLimitBytes=*/512'000'000ULL,
                                        /*fileShardSizeLimitBytes=*/500'000'000ULL);
            for (const TensorSpec& spec : specs) {
                sourceTensors.push_back(makeGpuUint8Tensor(spec.numBytes, spec.seed));
                writer.addArchiveFile(spec.pathInTar, sourceTensors.back());
            }

            if (fault == TarShortIoFault::ShortWriteSubmission) {
                // Skip two matching write requests so this cannot pass merely by
                // handling a short first operation in an otherwise empty pipeline.
                UringDirect::testSetNextIoUringSubmissionByteLimitForOperation(
                    /*limitBytes=*/4096, UringDirect::TestIoOperation::Write, /*matchingOperationsToSkip=*/2);
            }

            writer.createArchive(archiveDir.path(), /*overwriteIfExists=*/true);
            if (fault == TarShortIoFault::ShortWriteSubmission) {
                ASSERT_EQ(UringDirect::testGetIoUringSubmissionByteLimitHitCount(), 1u)
                    << "The intended archive write fault was not exercised.";
            }
        }

        ASSERT_TRUE(fs::exists(archiveDir.path() / (archiveName + ".thor.tar")))
            << "The large shard limit should keep this production-shaped workload in one archive shard.";

        UringDirect::testResetIoUringShortIoHooks();
        ScopedEnvVar readBackend("THOR_IO_BACKEND", fault == TarShortIoFault::ShortReadSubmission ? "uring_direct" : "pread_direct");
        thor_file::TarReader reader(archiveName, archiveDir.path());
        std::vector<Tensor> loadedTensors;
        loadedTensors.reserve(specs.size());
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
        for (const TensorSpec& spec : specs) {
            ASSERT_TRUE(reader.containsFile(spec.pathInTar));
            ASSERT_EQ(reader.getFileSize(spec.pathInTar), spec.numBytes);
            loadedTensors.emplace_back(gpuPlacement, TensorDescriptor(DataType::UINT8, {spec.numBytes}));
        }
        for (size_t specIndex : readOrder) {
            reader.registerReadRequest(specs[specIndex].pathInTar, loadedTensors[specIndex]);
        }

        if (fault == TarShortIoFault::ShortReadSubmission) {
            UringDirect::testSetNextIoUringSubmissionByteLimitForOperation(
                /*limitBytes=*/4096, UringDirect::TestIoOperation::Read, /*matchingOperationsToSkip=*/3);
        }

        reader.executeReadRequests();
        if (fault == TarShortIoFault::ShortReadSubmission) {
            ASSERT_EQ(UringDirect::testGetIoUringSubmissionByteLimitHitCount(), 1u) << "The intended archive read fault was not exercised.";
        }

        for (uint64_t i = 0; i < specs.size(); ++i) {
            std::vector<uint8_t> actual = copyGpuTensorToCpuBytes(loadedTensors[i]);
            ASSERT_EQ(actual.size(), specs[i].numBytes) << specs[i].pathInTar;
            expectDeterministicBytes(actual.data(), actual.size(), specs[i].seed, specs[i].pathInTar);
        }
    } catch (const std::runtime_error& e) {
        if (isUnavailableExplicitUringDirect(BackendCase{"uring_direct", "uring_direct"}, e.what())) {
            GTEST_SKIP() << "Explicit uring_direct backend is unavailable in this runtime: " << e.what();
        }
        throw;
    }
}

}  // namespace

class TarWriterReaderBackendRegression : public ::testing::TestWithParam<BackendCase> {};

TEST_P(TarWriterReaderBackendRegression, PreservesModelArtifactPayloadCrcsInArchiveOrder) {
    requireCudaDeviceOrSkip();
    const std::vector<TensorSpec> specs = archiveRegressionTensorSpecs();
    runTarArchiveRoundTripForBackend(GetParam(), sequentialReadOrder(specs.size()), /*repetitions=*/1, "archive_order");
}

TEST_P(TarWriterReaderBackendRegression, DISABLED_PreservesModelArtifactPayloadCrcsInNetworkLikeReadOrder) {
    requireCudaDeviceOrSkip();
    const std::vector<TensorSpec> specs = archiveRegressionTensorSpecs();
    runTarArchiveRoundTripForBackend(GetParam(), networkLikeScrambledReadOrder(specs.size()), /*repetitions=*/8, "network_like_read_order");
}

TEST_P(TarWriterReaderBackendRegression, DISABLED_PreservesModelArtifactPayloadCrcsWhenLargeEntriesShareOneShard) {
    requireCudaDeviceOrSkip();
    const std::vector<TensorSpec> specs = archiveRegressionTensorSpecs();
    runTarArchiveRoundTripForBackend(GetParam(),
                                     networkLikeScrambledReadOrder(specs.size()),
                                     /*repetitions=*/2,
                                     "single_large_shard",
                                     /*archiveShardSizeLimitBytes=*/512'000'000ULL);
}

TEST(TarWriterReaderFileShardingRegression, PreservesOneLogicalFileAcrossPayloadFragments) {
    requireCudaDeviceOrSkip();
    ScopedEnvVar globalBackend("THOR_IO_BACKEND", "pread_buffered");
    ScopedEnvVar writerBackend("THOR_TAR_WRITE_IO_BACKEND", "pread_buffered");
    ScopedEnvVar readerBackend("THOR_TAR_READ_IO_BACKEND", "pread_buffered");
    ScopedDirectory archiveDir(uniqueTempDir("thor_tar_file_sharding_regression"));

    constexpr uint64_t kPayloadBytes = 2'500'123ULL;
    constexpr uint64_t kFileShardBytes = 1'000'000ULL;
    constexpr uint64_t kArchiveShardBytes = 1'100'000ULL;
    constexpr uint32_t kSeed = 2718;
    const std::string archiveName = "file_sharding_regression";
    const std::string logicalPath = "layer0_weights_parameter_weights.gds";

    Tensor source = makeGpuUint8Tensor(kPayloadBytes, kSeed);
    thor_file::TarWriter writer(archiveName, kArchiveShardBytes, kFileShardBytes);
    writer.addArchiveFile(logicalPath, source);
    writer.createArchive(archiveDir.path(), /*overwriteIfExists=*/true);

    thor_file::TarReader reader(archiveName, archiveDir.path());
    const auto entriesByPath = reader.getArchiveEntries();
    const auto entryIt = entriesByPath.find(logicalPath);
    ASSERT_NE(entryIt, entriesByPath.end());
    ASSERT_EQ(entryIt->second.size(), 3u);
    EXPECT_EQ(entriesByPath.find(logicalPath + "/shard0"), entriesByPath.end());
    EXPECT_EQ(reader.getFileSize(logicalPath), kPayloadBytes);

    const std::vector<thor_file::EntryInfo>& entries = entryIt->second;
    EXPECT_EQ(entries[0].tensorDataOffset, 0u);
    EXPECT_EQ(entries[0].size, kFileShardBytes);
    EXPECT_EQ(entries[1].tensorDataOffset, kFileShardBytes);
    EXPECT_EQ(entries[1].size, kFileShardBytes);
    EXPECT_EQ(entries[2].tensorDataOffset, 2 * kFileShardBytes);
    EXPECT_EQ(entries[2].size, kPayloadBytes - 2 * kFileShardBytes);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Tensor loaded(gpuPlacement, TensorDescriptor(DataType::UINT8, {kPayloadBytes}));
    reader.registerReadRequest(logicalPath, loaded);
    reader.executeReadRequests();

    const std::vector<uint8_t> actual = copyGpuTensorToCpuBytes(loaded);
    ASSERT_EQ(actual.size(), kPayloadBytes);
    expectDeterministicBytes(actual.data(), actual.size(), kSeed, logicalPath);

    Tensor undersized(gpuPlacement, TensorDescriptor(DataType::UINT8, {kPayloadBytes - 1}));
    EXPECT_THROW(reader.registerReadRequest(logicalPath, undersized), std::runtime_error);
}

TEST(TarWriterReaderDiagnostics, CrcMismatchReportsIndependentBufferedPreadClassification) {
    requireCudaDeviceOrSkip();
    ScopedEnvVar globalBackend("THOR_IO_BACKEND", "pread_buffered");
    ScopedEnvVar tarWriteBackend("THOR_TAR_WRITE_IO_BACKEND", "pread_direct");
    ScopedEnvVar tarReadBackend("THOR_TAR_READ_IO_BACKEND", "pread_direct");
    ScopedDirectory archiveDir(uniqueTempDir("thor_tar_crc_mismatch_diagnostic"));
    const std::string archiveName = "crc_mismatch_diagnostic";
    const TensorSpec spec{"layer9282_weights_parameter_weights.gds", 64'003ULL, 9282};

    {
        Tensor source = makeGpuUint8Tensor(spec.numBytes, spec.seed);
        thor_file::TarWriter writer(archiveName,
                                    /*archiveShardSizeLimitBytes=*/512'000'000ULL,
                                    /*fileShardSizeLimitBytes=*/500'000'000ULL);
        writer.addArchiveFile(spec.pathInTar, source);
        writer.createArchive(archiveDir.path(), /*overwriteIfExists=*/true);
    }

    thor_file::TarReader indexReader(archiveName, archiveDir.path());
    const auto entries = indexReader.getArchiveEntries();
    const auto entryIt = entries.find(spec.pathInTar);
    ASSERT_NE(entryIt, entries.end());
    ASSERT_EQ(entryIt->second.size(), 1u);
    const thor_file::EntryInfo& entry = entryIt->second.front();

    const fs::path shardPath = archiveDir.path() / (archiveName + ".thor.tar");
    int fd = ::open(shardPath.c_str(), O_RDWR | O_CLOEXEC);
    ASSERT_GE(fd, 0) << std::strerror(errno);
    uint8_t corruptedByte = 0;
    ASSERT_EQ(::pread(fd, &corruptedByte, 1, static_cast<off_t>(entry.fileDataOffset)), 1);
    corruptedByte ^= 0x5Au;
    ASSERT_EQ(::pwrite(fd, &corruptedByte, 1, static_cast<off_t>(entry.fileDataOffset)), 1);
    ASSERT_EQ(::fsync(fd), 0);
    ASSERT_EQ(::close(fd), 0);

    thor_file::TarReader reader(archiveName, archiveDir.path());
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Tensor loaded(gpuPlacement, TensorDescriptor(DataType::UINT8, {spec.numBytes}));
    reader.registerReadRequest(spec.pathInTar, loaded);

    try {
        reader.executeReadRequests();
        FAIL() << "Expected the deliberately corrupted payload to fail CRC validation.";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(spec.pathInTar), std::string::npos);
        EXPECT_NE(message.find("io_backend=pread_direct"), std::string::npos);
        EXPECT_NE(message.find("buffered_pread_crc="), std::string::npos);
        EXPECT_NE(message.find("diagnosis=archive_payload_or_index_corruption"), std::string::npos);
        EXPECT_NE(message.find("file_offset=" + std::to_string(entry.fileDataOffset)), std::string::npos);
        EXPECT_NE(message.find("size=" + std::to_string(entry.size)), std::string::npos);
    }
}

TEST(TarWriterReaderProductionGeometryRegression, DISABLED_UringWriterPreservesObservedLayer9282OffsetAndSize) {
    requireCudaDeviceOrSkip();
    ScopedUringDirectShortIoHooks hooks;

    // This reproduces the exact archive geometry from the BLENDERS failure:
    //   file_data_offset = 618,632,704
    //   size             =  98,697,216
    //
    // The earlier production-shaped regression archive was only about 74 MiB in
    // total, so it never exercised either this file offset or this payload size.
    constexpr uint64_t kObservedFileDataOffset = 618'632'704ULL;
    constexpr uint64_t kObservedPayloadBytes = 98'697'216ULL;
    constexpr uint64_t kFillerPayloadBytes = 26'445'312ULL;
    constexpr uint64_t kArchiveLimitBytes = 1'000'000'000ULL;
    constexpr uint32_t kRepeatedEntryCount = 6;
    const std::string targetPath = "layer9282_weights_parameter_weights.gds";

    ScopedEnvVar globalBackend("THOR_IO_BACKEND", "pread_direct");
    ScopedEnvVar writerBackend("THOR_TAR_WRITE_IO_BACKEND", "uring_direct");
    ScopedEnvVar readerBackend("THOR_TAR_READ_IO_BACKEND", "pread_direct");
    ScopedDirectory archiveDir(uniqueTempDir("thor_tar_observed_layer9282_geometry"));
    const std::string archiveName = "observed_layer9282_geometry";

    // Use distinct payloads around the target so a buffer-reuse or wrong-offset
    // write cannot pass merely because the neighboring entry contains identical
    // bytes. Reusing each large source for alternating entries keeps GPU memory
    // substantially below the size of the generated archive.
    Tensor largeA = makeGpuUint8Tensor(kObservedPayloadBytes, 9101);
    Tensor largeB = makeGpuUint8Tensor(kObservedPayloadBytes, 9102);
    Tensor target = makeGpuUint8Tensor(kObservedPayloadBytes, 9282);
    Tensor filler = makeGpuUint8Tensor(kFillerPayloadBytes, 9103);

    thor_file::TarWriter writer(archiveName,
                                /*archiveShardSizeLimitBytes=*/kArchiveLimitBytes,
                                /*fileShardSizeLimitBytes=*/500'000'000ULL);
    for (uint32_t i = 0; i < kRepeatedEntryCount; ++i) {
        writer.addArchiveFile("prefix_layer_" + std::to_string(i) + "_weights.gds", (i & 1u) == 0 ? largeA : largeB);
    }
    writer.addArchiveFile("prefix_alignment_filler.gds", filler);
    writer.addArchiveFile(targetPath, target);

    try {
        writer.createArchive(archiveDir.path(), /*overwriteIfExists=*/true);
    } catch (const std::runtime_error& e) {
        if (isUnavailableExplicitUringDirect(BackendCase{"uring_direct", "uring_direct"}, e.what())) {
            GTEST_SKIP() << "Explicit uring_direct backend is unavailable in this runtime: " << e.what();
        }
        throw;
    }

    ASSERT_TRUE(fs::exists(archiveDir.path() / (archiveName + ".thor.tar")))
        << "The configured archive limit should keep the observed geometry in one shard.";

    thor_file::TarReader reader(archiveName, archiveDir.path());
    const auto entries = reader.getArchiveEntries();
    const auto targetIt = entries.find(targetPath);
    ASSERT_NE(targetIt, entries.end());
    ASSERT_EQ(targetIt->second.size(), 1u);
    EXPECT_EQ(targetIt->second.front().archiveShard, 0u);
    EXPECT_EQ(targetIt->second.front().fileDataOffset, kObservedFileDataOffset);
    EXPECT_EQ(targetIt->second.front().size, kObservedPayloadBytes);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Tensor loaded(gpuPlacement, TensorDescriptor(DataType::UINT8, {kObservedPayloadBytes}));
    reader.registerReadRequest(targetPath, loaded);
    reader.executeReadRequests();

    const std::vector<uint8_t> actual = copyGpuTensorToCpuBytes(loaded);
    ASSERT_EQ(actual.size(), kObservedPayloadBytes);
    expectDeterministicBytes(actual.data(), actual.size(), 9282, targetPath);
}

TEST(TarWriterReaderShortIoRegression, ShortMiddleUringWriteSubmissionPublishesValidSingleShardArchive) {
#ifdef NDEBUG
    GTEST_SKIP() << "The archive-level io_uring short-I/O fault injection is only "
                    "supported in Debug builds.";
#endif
    runFaultInjectedSingleShardRoundTrip(TarShortIoFault::ShortWriteSubmission);
}

TEST(TarWriterReaderShortIoRegression, ShortMiddleUringReadSubmissionLoadsValidSingleShardArchive) {
#ifdef NDEBUG
    GTEST_SKIP() << "The archive-level io_uring short-I/O fault injection is only "
                    "supported in Debug builds.";
#endif
    runFaultInjectedSingleShardRoundTrip(TarShortIoFault::ShortReadSubmission);
}

INSTANTIATE_TEST_SUITE_P(ExplicitBackends,
                         TarWriterReaderBackendRegression,
                         ::testing::Values(BackendCase{"uring_direct", "uring_direct"},
                                           BackendCase{"pread_direct", "pread_direct"},
                                           BackendCase{"pread_buffered", "pread_buffered"}),
                         [](const ::testing::TestParamInfo<BackendCase>& info) { return std::string(info.param.displayName); });
