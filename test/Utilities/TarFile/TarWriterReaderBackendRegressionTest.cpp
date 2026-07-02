#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/TarWriter.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
           message.find("io_uring_register_files failed") != std::string::npos ||
           message.find("open(O_DIRECT") != std::string::npos;
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
                                      const std::string& caseName) {
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
                                    /*archiveShardSizeLimitBytes=*/12'000'000ULL,
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

}  // namespace

class TarWriterReaderBackendRegression : public ::testing::TestWithParam<BackendCase> {};

TEST_P(TarWriterReaderBackendRegression, PreservesModelArtifactPayloadCrcsInArchiveOrder) {
    requireCudaDeviceOrSkip();
    const std::vector<TensorSpec> specs = archiveRegressionTensorSpecs();
    runTarArchiveRoundTripForBackend(GetParam(), sequentialReadOrder(specs.size()), /*repetitions=*/1, "archive_order");
}

TEST_P(TarWriterReaderBackendRegression, PreservesModelArtifactPayloadCrcsInNetworkLikeReadOrder) {
    requireCudaDeviceOrSkip();
    const std::vector<TensorSpec> specs = archiveRegressionTensorSpecs();
    runTarArchiveRoundTripForBackend(GetParam(), networkLikeScrambledReadOrder(specs.size()), /*repetitions=*/8, "network_like_read_order");
}

INSTANTIATE_TEST_SUITE_P(ExplicitBackends,
                         TarWriterReaderBackendRegression,
                         ::testing::Values(BackendCase{"uring_direct", "uring_direct"},
                                           BackendCase{"pread_direct", "pread_direct"},
                                           BackendCase{"pread_buffered", "pread_buffered"}),
                         [](const ::testing::TestParamInfo<BackendCase>& info) { return std::string(info.param.displayName); });
