#include "test/Utilities/TensorOperations/Fp8ArgKeyDevice.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ThorImplementation::Fp8ArgKeyDevice {
namespace {

using CubReductionInternal::Fp8ArgFold;
using CubReductionInternal::Fp8ArgKeys4;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

// Thor streams are non-blocking, and default-stream memory APIs are banned by the source guards: run on our own.
struct TestStream {
    cudaStream_t stream = nullptr;
    TestStream() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"); }
    ~TestStream() { (void)cudaStreamDestroy(stream); }
    TestStream(const TestStream&) = delete;
    TestStream& operator=(const TestStream&) = delete;
    void copy(void* to, const void* from, size_t bytes, cudaMemcpyKind kind) const {
        check(cudaMemcpyAsync(to, from, bytes, kind, stream), "cudaMemcpyAsync");
    }
    void synchronize() const { check(cudaStreamSynchronize(stream), "cudaStreamSynchronize"); }
};

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { check(cudaMalloc(&data, count * sizeof(T) + 16), "cudaMalloc"); }
    ~DeviceBuffer() { (void)cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void keysKernel(const uint32_t* words, uint32_t count, Fp8ArgFold fold, uint32_t* keys,
                           uint32_t* negative_zero) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Fp8ArgKeys4 k = CubReductionInternal::fp8ArgKeys4(words[i], fold);
    keys[i] = k.keys;
    negative_zero[i] = k.negative_zero;
}

__global__ void columnsKernel(const uint8_t* input, uint32_t rows, uint32_t cols, uint32_t lanes, Fp8ArgFold fold,
                              uint32_t* index, uint32_t* value_byte) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    const uint32_t byte = col & 3u;  // exercise every byte position of the packed word
    const uint32_t chunk = (rows + lanes - 1u) / lanes;
    uint64_t best = CubReductionInternal::FP8_ARG_WIDE_IDENTITY;
    for (uint32_t base = 0; base < rows; base += chunk) {
        uint32_t lane = CubReductionInternal::FP8_ARG_LANE_IDENTITY;
        const uint32_t end = base + chunk < rows ? base + chunk : rows;
        for (uint32_t r = base; r < end; ++r) {
            const uint32_t word = static_cast<uint32_t>(input[static_cast<uint64_t>(r) * cols + col]) << (8u * byte);
            const Fp8ArgKeys4 k = CubReductionInternal::fp8ArgKeys4(word, fold);
            lane = CubReductionInternal::fp8ArgMin(lane, CubReductionInternal::fp8ArgLaneCandidate(k, byte, r - base));
        }
        best = CubReductionInternal::fp8ArgMin(best, CubReductionInternal::fp8ArgWiden(lane, base));
    }
    index[col] = static_cast<uint32_t>(CubReductionInternal::fp8ArgWideIndex(best));  // test columns < 2^32 rows
    value_byte[col] = CubReductionInternal::fp8ArgValueByte(CubReductionInternal::fp8ArgWideKey(best),
                                                            CubReductionInternal::fp8ArgWideNegativeZero(best), fold);
}

}  // namespace

void keys(const std::vector<uint32_t>& words, Fp8ArgFold fold, std::vector<uint32_t>& keys,
          std::vector<uint32_t>& negative_zero) {
    const uint32_t count = static_cast<uint32_t>(words.size());
    const TestStream stream;
    DeviceBuffer<uint32_t> in(count), out_keys(count), out_zero(count);
    stream.copy(in.data, words.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    keysKernel<<<(count + 255u) / 256u, 256, 0, stream.stream>>>(in.data, count, fold, out_keys.data, out_zero.data);
    check(cudaGetLastError(), "keysKernel launch");
    keys.resize(count);
    negative_zero.resize(count);
    stream.copy(keys.data(), out_keys.data, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    stream.copy(negative_zero.data(), out_zero.data, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    stream.synchronize();
}

void columns(const std::vector<uint8_t>& input, uint32_t rows, uint32_t cols, uint32_t lanes, Fp8ArgFold fold,
             std::vector<uint32_t>& index, std::vector<uint32_t>& value_byte) {
    if (input.size() != static_cast<size_t>(rows) * cols || lanes == 0) throw std::invalid_argument("bad columns()");
    const TestStream stream;
    DeviceBuffer<uint8_t> in(input.size());
    DeviceBuffer<uint32_t> out_index(cols), out_value(cols);
    stream.copy(in.data, input.data(), input.size(), cudaMemcpyHostToDevice);
    columnsKernel<<<(cols + 127u) / 128u, 128, 0, stream.stream>>>(in.data, rows, cols, lanes, fold, out_index.data,
                                                                    out_value.data);
    check(cudaGetLastError(), "columnsKernel launch");
    index.resize(cols);
    value_byte.resize(cols);
    stream.copy(index.data(), out_index.data, cols * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    stream.copy(value_byte.data(), out_value.data, cols * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    stream.synchronize();
}

}  // namespace ThorImplementation::Fp8ArgKeyDevice
