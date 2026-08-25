#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

#include "gtest/gtest.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                              \
    do {                                                                                                                   \
        int cudaDeviceCountForTest = 0;                                                                                    \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                                \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                             \
            GTEST_SKIP() << "CUDA device is required for mixed ragged cuDNN attention tests.";                            \
        }                                                                                                                  \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

std::optional<Tensor> allocateAttentionWorkspace(const CudnnAttentionDescriptor& descriptor, bool backward) {
    CudnnScaledDotProductAttention& attention = CudnnScaledDotProductAttention::instance();
    const uint64_t bytes = backward ? attention.backwardWorkspaceSizeInBytes(descriptor, 0)
                                    : attention.forwardWorkspaceSizeInBytes(descriptor, 0);
    if (bytes == 0) {
        return std::nullopt;
    }
    return Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {bytes}), 256);
}

std::vector<uint64_t> asUint64(const std::vector<int64_t>& values) {
    std::vector<uint64_t> converted;
    converted.reserve(values.size());
    for (int64_t value : values) {
        converted.push_back(static_cast<uint64_t>(value));
    }
    return converted;
}

uint64_t storageElements(const AttentionTensorSpec& spec) {
    uint64_t maxOffset = 0;
    for (size_t i = 0; i < spec.dimensions.size(); ++i) {
        maxOffset += static_cast<uint64_t>(spec.dimensions[i] - 1) * static_cast<uint64_t>(spec.strides[i]);
    }
    return maxOffset + 1;
}

Tensor makeHalfTensorWithSpec(const AttentionTensorSpec& spec, const std::vector<float>& values, Stream& stream) {
    const uint64_t storageSize = storageElements(spec);
    EXPECT_EQ(values.size(), storageSize);

    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP16, {storageSize}));
    __half* cpuValues = cpu.getMemPtr<__half>();
    for (uint64_t i = 0; i < storageSize; ++i) {
        cpuValues[i] = __float2half(values[i]);
    }

    Tensor gpuStorage(gpuPlacement, TensorDescriptor(DataType::FP16, {storageSize}));
    gpuStorage.copyFromAsync(cpu, stream);
    return gpuStorage.aliasView(asUint64(spec.dimensions), asUint64(spec.strides));
}

Tensor makeEmptyHalfTensorWithSpec(const AttentionTensorSpec& spec) {
    const uint64_t storageSize = storageElements(spec);
    Tensor gpuStorage(gpuPlacement, TensorDescriptor(DataType::FP16, {storageSize}));
    return gpuStorage.aliasView(asUint64(spec.dimensions), asUint64(spec.strides));
}

Tensor makePackedHalfTensor(uint64_t tokens, uint64_t heads, uint64_t dim, const std::vector<float>& values, Stream& stream) {
    const std::vector<uint64_t> dims{tokens, heads, dim};
    const uint64_t count = tokens * heads * dim;
    EXPECT_EQ(values.size(), count);

    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP16, dims));
    __half* cpuValues = cpu.getMemPtr<__half>();
    for (uint64_t i = 0; i < count; ++i) {
        cpuValues[i] = __float2half(values[i]);
    }
    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP16, dims));
    gpu.copyFromAsync(cpu, stream);
    return gpu;
}

Tensor makeEmptyPackedHalfTensor(uint64_t tokens, uint64_t heads, uint64_t dim) {
    return Tensor(gpuPlacement, TensorDescriptor(DataType::FP16, {tokens, heads, dim}));
}

Tensor makeInt32Tensor(const std::vector<int32_t>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::INT32, {values.size()}));
    int32_t* ptr = cpu.getMemPtr<int32_t>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }
    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::INT32, {values.size()}));
    gpu.copyFromAsync(cpu, stream);
    return gpu;
}

Tensor makeUint32Tensor(const std::vector<uint32_t>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::UINT32, {values.size()}));
    uint32_t* ptr = cpu.getMemPtr<uint32_t>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }
    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::UINT32, {values.size()}));
    gpu.copyFromAsync(cpu, stream);
    return gpu;
}

std::vector<float> copyHalfStorage(const Tensor& tensor, uint64_t storageSize, Stream& stream) {
    Tensor raw = tensor.aliasView({storageSize}, {1});
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP16, {storageSize}));
    cpu.copyFromAsync(raw, stream);
    stream.synchronize();

    std::vector<float> values(storageSize);
    const __half* ptr = cpu.getMemPtr<__half>();
    for (uint64_t i = 0; i < storageSize; ++i) {
        values[i] = __half2float(ptr[i]);
    }
    return values;
}

std::vector<float> copyFloatStorage(const Tensor& tensor, Stream& stream) {
    const uint64_t count = tensor.getTotalNumElements();
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, {count}));
    Tensor raw = tensor.aliasView({count}, {1});
    cpu.copyFromAsync(raw, stream);
    stream.synchronize();

    const float* ptr = cpu.getMemPtr<float>();
    return std::vector<float>(ptr, ptr + count);
}

std::vector<int32_t> copyInt32Storage(const Tensor& tensor, Stream& stream) {
    const uint64_t count = tensor.getTotalNumElements();
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::INT32, {count}));
    Tensor raw = tensor.aliasView({count}, {1});
    cpu.copyFromAsync(raw, stream);
    stream.synchronize();

    const int32_t* ptr = cpu.getMemPtr<int32_t>();
    return std::vector<int32_t>(ptr, ptr + count);
}

CudnnRaggedAttentionScratch makeScratch(uint64_t batch) {
    return CudnnRaggedAttentionScratch{
        .seqLenQ = Tensor(gpuPlacement, TensorDescriptor(DataType::INT32, {batch})),
        .seqLenKv = Tensor(gpuPlacement, TensorDescriptor(DataType::INT32, {batch})),
        .qElementOffsets = Tensor(gpuPlacement, TensorDescriptor(DataType::INT32, {batch + 1})),
        .kElementOffsets = Tensor(gpuPlacement, TensorDescriptor(DataType::INT32, {batch + 1})),
        .vElementOffsets = Tensor(gpuPlacement, TensorDescriptor(DataType::INT32, {batch + 1})),
        .oElementOffsets = Tensor(gpuPlacement, TensorDescriptor(DataType::INT32, {batch + 1})),
    };
}

uint64_t bshdIndex(uint64_t b, uint64_t s, uint64_t h, uint64_t d, uint64_t sequence, uint64_t heads, uint64_t dim) {
    return (((b * sequence + s) * heads + h) * dim + d);
}

uint64_t thdIndex(uint64_t t, uint64_t h, uint64_t d, uint64_t heads, uint64_t dim) {
    return ((t * heads + h) * dim + d);
}

void expectNearVectors(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance, const std::string& label) {
    ASSERT_EQ(actual.size(), expected.size()) << label;
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << label << " index=" << i;
    }
}

}  // namespace

TEST(CudnnMixedRaggedAttention, DenseQueryRaggedKvForwardBackwardMatchesPaddedDenseReference) {
    REQUIRE_CUDA_DEVICE();
    if (!CudnnScaledDotProductAttention::frontendAvailable()) {
        GTEST_SKIP() << "cuDNN Frontend attention is not available.";
    }

    constexpr uint64_t batch = 2;
    constexpr uint64_t heads = 2;
    constexpr uint64_t queryLength = 3;
    constexpr uint64_t kvCapacity = 7;
    constexpr uint64_t dim = 8;
    const std::vector<uint32_t> rowOffsets{0, 2, 5};
    const std::vector<int32_t> kvLengths{2, 3};

    Stream stream(0);
    CudnnScaledDotProductAttention::instance().clearSelectionCache();

    CudnnAttentionDescriptor mixed;
    mixed.q = AttentionTensorSpec::bshd(batch, heads, queryLength, dim, DataType::FP16);
    mixed.k = AttentionTensorSpec::bshd(batch, heads, kvCapacity, dim, DataType::FP16);
    mixed.v = AttentionTensorSpec::bshd(batch, heads, kvCapacity, dim, DataType::FP16);
    mixed.o = AttentionTensorSpec::bshd(batch, heads, queryLength, dim, DataType::FP16);
    mixed.k.ragged = true;
    mixed.v.ragged = true;
    mixed.computeDataType = DataType::FP32;
    mixed.intermediateDataType = DataType::FP32;
    mixed.usePaddingMask = true;
    mixed.generateStats = true;
    mixed.debugName = "mixed_dense_q_ragged_kv";
    ASSERT_NO_THROW(mixed.validateBackward());

    CudnnAttentionDescriptor dense = mixed;
    dense.k.ragged = false;
    dense.v.ragged = false;
    dense.debugName = "padded_dense_reference";
    ASSERT_NO_THROW(dense.validateBackward());

    std::vector<float> qValues(batch * queryLength * heads * dim);
    for (uint64_t b = 0; b < batch; ++b) {
        for (uint64_t s = 0; s < queryLength; ++s) {
            for (uint64_t h = 0; h < heads; ++h) {
                for (uint64_t d = 0; d < dim; ++d) {
                    qValues[bshdIndex(b, s, h, d, queryLength, heads, dim)] =
                        0.025f * static_cast<float>(1 + b) + 0.015f * static_cast<float>(s) +
                        0.01f * static_cast<float>(h) - 0.004f * static_cast<float>(d);
                }
            }
        }
    }

    std::vector<float> packedK(kvCapacity * heads * dim, 37.0f);
    std::vector<float> packedV(kvCapacity * heads * dim, -41.0f);
    for (uint64_t t = 0; t < rowOffsets.back(); ++t) {
        for (uint64_t h = 0; h < heads; ++h) {
            for (uint64_t d = 0; d < dim; ++d) {
                packedK[thdIndex(t, h, d, heads, dim)] =
                    0.03f * static_cast<float>(1 + t) + 0.012f * static_cast<float>(h) - 0.003f * static_cast<float>(d);
                packedV[thdIndex(t, h, d, heads, dim)] =
                    -0.02f * static_cast<float>(1 + t) + 0.017f * static_cast<float>(h) + 0.005f * static_cast<float>(d);
            }
        }
    }

    std::vector<float> denseK(batch * kvCapacity * heads * dim, 53.0f);
    std::vector<float> denseV(batch * kvCapacity * heads * dim, -59.0f);
    for (uint64_t b = 0; b < batch; ++b) {
        const uint64_t length = static_cast<uint64_t>(kvLengths[b]);
        for (uint64_t s = 0; s < length; ++s) {
            const uint64_t t = static_cast<uint64_t>(rowOffsets[b]) + s;
            for (uint64_t h = 0; h < heads; ++h) {
                for (uint64_t d = 0; d < dim; ++d) {
                    denseK[bshdIndex(b, s, h, d, kvCapacity, heads, dim)] = packedK[thdIndex(t, h, d, heads, dim)];
                    denseV[bshdIndex(b, s, h, d, kvCapacity, heads, dim)] = packedV[thdIndex(t, h, d, heads, dim)];
                }
            }
        }
    }

    Tensor qMixed = makeHalfTensorWithSpec(mixed.q, qValues, stream);
    Tensor qDense = makeHalfTensorWithSpec(dense.q, qValues, stream);
    Tensor kMixed = makePackedHalfTensor(kvCapacity, heads, dim, packedK, stream);
    Tensor vMixed = makePackedHalfTensor(kvCapacity, heads, dim, packedV, stream);
    Tensor kDense = makeHalfTensorWithSpec(dense.k, denseK, stream);
    Tensor vDense = makeHalfTensorWithSpec(dense.v, denseV, stream);
    Tensor oMixed = makeEmptyHalfTensorWithSpec(mixed.o);
    Tensor oDense = makeEmptyHalfTensorWithSpec(dense.o);
    Tensor statsMixed(gpuPlacement, TensorDescriptor(DataType::FP32, {batch, heads, queryLength, 1}));
    Tensor statsDense(gpuPlacement, TensorDescriptor(DataType::FP32, {batch, heads, queryLength, 1}));
    Tensor kvOffsets = makeUint32Tensor(rowOffsets, stream);
    Tensor denseSeqQ = makeInt32Tensor({static_cast<int32_t>(queryLength), static_cast<int32_t>(queryLength)}, stream);
    Tensor denseSeqKv = makeInt32Tensor(kvLengths, stream);
    CudnnRaggedAttentionScratch scratch = makeScratch(batch);

    CudnnAttentionForwardArgs mixedForward{
        .q = qMixed,
        .k = kMixed,
        .v = vMixed,
        .o = oMixed,
        .stats = statsMixed,
        .kvRowPartitionOffsets = kvOffsets,
        .raggedScratch = scratch,
    };
    CudnnAttentionForwardArgs denseForward{
        .q = qDense,
        .k = kDense,
        .v = vDense,
        .o = oDense,
        .stats = statsDense,
        .seqLenQ = denseSeqQ,
        .seqLenKv = denseSeqKv,
    };

    std::optional<Tensor> mixedForwardWorkspace = allocateAttentionWorkspace(mixed, false);
    std::optional<Tensor> denseForwardWorkspace = allocateAttentionWorkspace(dense, false);
    CudnnAttentionExecutablePlan mixedForwardPlan =
        CudnnScaledDotProductAttention::instance().prepareForward(mixed, mixedForward, stream);
    CudnnAttentionExecutablePlan denseForwardPlan =
        CudnnScaledDotProductAttention::instance().prepareForward(dense, denseForward, stream);
    ASSERT_NO_THROW(CudnnScaledDotProductAttention::instance().forward(mixedForwardPlan, mixedForward, mixedForwardWorkspace, stream));
    ASSERT_NO_THROW(CudnnScaledDotProductAttention::instance().forward(denseForwardPlan, denseForward, denseForwardWorkspace, stream));
    stream.synchronize();

    const uint64_t outputStorage = storageElements(mixed.o);
    expectNearVectors(copyHalfStorage(oMixed, outputStorage, stream), copyHalfStorage(oDense, outputStorage, stream), 0.02f, "forward");
    expectNearVectors(copyFloatStorage(statsMixed, stream), copyFloatStorage(statsDense, stream), 0.002f, "stats");
    EXPECT_EQ(copyInt32Storage(scratch.seqLenQ, stream),
              (std::vector<int32_t>{static_cast<int32_t>(queryLength), static_cast<int32_t>(queryLength)}));
    EXPECT_EQ(copyInt32Storage(scratch.seqLenKv, stream), kvLengths);
    // Dense BSHD Q/O can be described to cuDNN as uniform THD without moving
    // payloads. Backward consumes these synthetic offsets to use the stable
    // all-ragged cuDNN path.
    EXPECT_EQ(copyInt32Storage(scratch.qElementOffsets, stream), (std::vector<int32_t>{0, 48, 96}));
    EXPECT_EQ(copyInt32Storage(scratch.oElementOffsets, stream), (std::vector<int32_t>{0, 48, 96}));

    std::vector<float> dOValues(outputStorage);
    for (uint64_t i = 0; i < outputStorage; ++i) {
        dOValues[i] = 0.01f * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
    Tensor dOMixed = makeHalfTensorWithSpec(mixed.o, dOValues, stream);
    Tensor dODense = makeHalfTensorWithSpec(dense.o, dOValues, stream);
    Tensor dQMixed = makeEmptyHalfTensorWithSpec(mixed.q);
    Tensor dQDense = makeEmptyHalfTensorWithSpec(dense.q);
    Tensor dKMixed = makeEmptyPackedHalfTensor(kvCapacity, heads, dim);
    Tensor dVMixed = makeEmptyPackedHalfTensor(kvCapacity, heads, dim);
    Tensor dKDense = makeEmptyHalfTensorWithSpec(dense.k);
    Tensor dVDense = makeEmptyHalfTensorWithSpec(dense.v);

    // Make unwritten gradient regions deterministic.  In particular, every dense-Q
    // token is logically valid here, so a correct backward execution must overwrite
    // the sentinel in every dQ element rather than inheriting allocator history.
    constexpr float gradientSentinel = 7.0f;
    dQMixed.fill(gradientSentinel, stream);
    dQDense.fill(gradientSentinel, stream);
    dKMixed.fill(gradientSentinel, stream);
    dVMixed.fill(gradientSentinel, stream);
    dKDense.fill(gradientSentinel, stream);
    dVDense.fill(gradientSentinel, stream);

    CudnnAttentionBackwardArgs mixedBackward{
        .q = qMixed,
        .k = kMixed,
        .v = vMixed,
        .o = oMixed,
        .dO = dOMixed,
        .stats = statsMixed,
        .dQ = dQMixed,
        .dK = dKMixed,
        .dV = dVMixed,
        .kvRowPartitionOffsets = kvOffsets,
        .raggedScratch = scratch,
    };
    CudnnAttentionBackwardArgs denseBackward{
        .q = qDense,
        .k = kDense,
        .v = vDense,
        .o = oDense,
        .dO = dODense,
        .stats = statsDense,
        .dQ = dQDense,
        .dK = dKDense,
        .dV = dVDense,
        .seqLenQ = denseSeqQ,
        .seqLenKv = denseSeqKv,
    };

    std::optional<Tensor> mixedBackwardWorkspace = allocateAttentionWorkspace(mixed, true);
    std::optional<Tensor> denseBackwardWorkspace = allocateAttentionWorkspace(dense, true);
    CudnnAttentionExecutablePlan mixedBackwardPlan =
        CudnnScaledDotProductAttention::instance().prepareBackward(mixed, mixedBackward, stream);
    CudnnAttentionExecutablePlan denseBackwardPlan =
        CudnnScaledDotProductAttention::instance().prepareBackward(dense, denseBackward, stream);
    ASSERT_NO_THROW(CudnnScaledDotProductAttention::instance().backward(mixedBackwardPlan, mixedBackward, mixedBackwardWorkspace, stream));
    ASSERT_NO_THROW(CudnnScaledDotProductAttention::instance().backward(denseBackwardPlan, denseBackward, denseBackwardWorkspace, stream));
    stream.synchronize();

    const auto dQMixedValues = copyHalfStorage(dQMixed, storageElements(mixed.q), stream);
    const auto dQDenseValues = copyHalfStorage(dQDense, storageElements(dense.q), stream);
    for (size_t i = 0; i < dQMixedValues.size(); ++i) {
        EXPECT_NE(dQMixedValues[i], gradientSentinel) << "mixed dQ was not written at index=" << i;
        EXPECT_NE(dQDenseValues[i], gradientSentinel) << "dense dQ was not written at index=" << i;
    }
    expectNearVectors(dQMixedValues, dQDenseValues, 0.03f, "dQ");

    const auto dKMixedValues = copyHalfStorage(dKMixed, kvCapacity * heads * dim, stream);
    const auto dVMixedValues = copyHalfStorage(dVMixed, kvCapacity * heads * dim, stream);
    const auto dKDenseValues = copyHalfStorage(dKDense, storageElements(dense.k), stream);
    const auto dVDenseValues = copyHalfStorage(dVDense, storageElements(dense.v), stream);

    for (uint64_t b = 0; b < batch; ++b) {
        const uint64_t length = static_cast<uint64_t>(kvLengths[b]);
        for (uint64_t s = 0; s < length; ++s) {
            const uint64_t t = static_cast<uint64_t>(rowOffsets[b]) + s;
            for (uint64_t h = 0; h < heads; ++h) {
                for (uint64_t d = 0; d < dim; ++d) {
                    const uint64_t packedIndex = thdIndex(t, h, d, heads, dim);
                    const uint64_t denseIndex = bshdIndex(b, s, h, d, kvCapacity, heads, dim);
                    EXPECT_NEAR(dKMixedValues[packedIndex], dKDenseValues[denseIndex], 0.03f)
                        << "dK b=" << b << " s=" << s << " h=" << h << " d=" << d;
                    EXPECT_NEAR(dVMixedValues[packedIndex], dVDenseValues[denseIndex], 0.03f)
                        << "dV b=" << b << " s=" << s << " h=" << h << " d=" << d;
                }
            }
        }
    }
}

TEST(CudnnMixedRaggedAttention, RaggedQueryDenseKvForwardBackwardMatchesPaddedDenseReference) {
    REQUIRE_CUDA_DEVICE();
    if (!CudnnScaledDotProductAttention::frontendAvailable()) {
        GTEST_SKIP() << "cuDNN Frontend attention is not available.";
    }

    constexpr uint64_t batch = 2;
    constexpr uint64_t heads = 2;
    constexpr uint64_t queryCapacity = 7;
    constexpr uint64_t kvLength = 3;
    constexpr uint64_t dim = 8;
    const std::vector<uint32_t> queryRowOffsets{0, 2, 5};
    const std::vector<int32_t> queryLengths{2, 3};
    const std::vector<int32_t> kvLengths{static_cast<int32_t>(kvLength), static_cast<int32_t>(kvLength)};

    Stream stream(0);
    CudnnScaledDotProductAttention::instance().clearSelectionCache();

    CudnnAttentionDescriptor mixed;
    mixed.q = AttentionTensorSpec::bshd(batch, heads, queryCapacity, dim, DataType::FP16);
    mixed.k = AttentionTensorSpec::bshd(batch, heads, kvLength, dim, DataType::FP16);
    mixed.v = AttentionTensorSpec::bshd(batch, heads, kvLength, dim, DataType::FP16);
    mixed.o = AttentionTensorSpec::bshd(batch, heads, queryCapacity, dim, DataType::FP16);
    mixed.q.ragged = true;
    mixed.o.ragged = true;
    mixed.computeDataType = DataType::FP32;
    mixed.intermediateDataType = DataType::FP32;
    mixed.usePaddingMask = true;
    mixed.generateStats = true;
    mixed.debugName = "mixed_ragged_q_dense_kv";
    ASSERT_NO_THROW(mixed.validateBackward());

    CudnnAttentionDescriptor dense = mixed;
    dense.q.ragged = false;
    dense.o.ragged = false;
    dense.debugName = "padded_dense_query_reference";
    ASSERT_NO_THROW(dense.validateBackward());

    std::vector<float> packedQ(queryCapacity * heads * dim, 31.0f);
    std::vector<float> denseQ(batch * queryCapacity * heads * dim, 43.0f);
    for (uint64_t b = 0; b < batch; ++b) {
        const uint64_t length = static_cast<uint64_t>(queryLengths[b]);
        for (uint64_t s = 0; s < length; ++s) {
            const uint64_t t = static_cast<uint64_t>(queryRowOffsets[b]) + s;
            for (uint64_t h = 0; h < heads; ++h) {
                for (uint64_t d = 0; d < dim; ++d) {
                    const float value = 0.025f * static_cast<float>(1 + b) + 0.014f * static_cast<float>(s) +
                                        0.009f * static_cast<float>(h) - 0.003f * static_cast<float>(d);
                    packedQ[thdIndex(t, h, d, heads, dim)] = value;
                    denseQ[bshdIndex(b, s, h, d, queryCapacity, heads, dim)] = value;
                }
            }
        }
    }

    std::vector<float> denseK(batch * kvLength * heads * dim);
    std::vector<float> denseV(batch * kvLength * heads * dim);
    for (uint64_t b = 0; b < batch; ++b) {
        for (uint64_t s = 0; s < kvLength; ++s) {
            for (uint64_t h = 0; h < heads; ++h) {
                for (uint64_t d = 0; d < dim; ++d) {
                    const uint64_t index = bshdIndex(b, s, h, d, kvLength, heads, dim);
                    denseK[index] = 0.031f * static_cast<float>(1 + b) + 0.018f * static_cast<float>(s) +
                                    0.007f * static_cast<float>(h) - 0.002f * static_cast<float>(d);
                    denseV[index] = -0.021f * static_cast<float>(1 + b) + 0.013f * static_cast<float>(s) +
                                    0.006f * static_cast<float>(h) + 0.004f * static_cast<float>(d);
                }
            }
        }
    }

    Tensor qMixed = makePackedHalfTensor(queryCapacity, heads, dim, packedQ, stream);
    Tensor qDense = makeHalfTensorWithSpec(dense.q, denseQ, stream);
    Tensor kMixed = makeHalfTensorWithSpec(mixed.k, denseK, stream);
    Tensor vMixed = makeHalfTensorWithSpec(mixed.v, denseV, stream);
    Tensor kDense = makeHalfTensorWithSpec(dense.k, denseK, stream);
    Tensor vDense = makeHalfTensorWithSpec(dense.v, denseV, stream);
    Tensor oMixed = makeEmptyPackedHalfTensor(queryCapacity, heads, dim);
    Tensor oDense = makeEmptyHalfTensorWithSpec(dense.o);
    Tensor statsMixed(gpuPlacement, TensorDescriptor(DataType::FP32, {batch, heads, queryCapacity, 1}));
    Tensor statsDense(gpuPlacement, TensorDescriptor(DataType::FP32, {batch, heads, queryCapacity, 1}));
    // Poison invalid output/stat capacity independently in the mixed and padded
    // representations. Forward must overwrite every logical query position and
    // backward must ignore whatever remains outside each row's sequence length.
    oMixed.fill(29.0, stream);
    oDense.fill(-31.0, stream);
    statsMixed.fill(37.0, stream);
    statsDense.fill(-41.0, stream);
    Tensor queryOffsets = makeUint32Tensor(queryRowOffsets, stream);
    Tensor denseSeqQ = makeInt32Tensor(queryLengths, stream);
    Tensor denseSeqKv = makeInt32Tensor(kvLengths, stream);
    CudnnRaggedAttentionScratch scratch = makeScratch(batch);

    CudnnAttentionForwardArgs mixedForward{
        .q = qMixed,
        .k = kMixed,
        .v = vMixed,
        .o = oMixed,
        .stats = statsMixed,
        .qRowPartitionOffsets = queryOffsets,
        .raggedScratch = scratch,
    };
    CudnnAttentionForwardArgs denseForward{
        .q = qDense,
        .k = kDense,
        .v = vDense,
        .o = oDense,
        .stats = statsDense,
        .seqLenQ = denseSeqQ,
        .seqLenKv = denseSeqKv,
    };

    std::optional<Tensor> mixedForwardWorkspace = allocateAttentionWorkspace(mixed, false);
    std::optional<Tensor> denseForwardWorkspace = allocateAttentionWorkspace(dense, false);
    CudnnAttentionExecutablePlan mixedForwardPlan =
        CudnnScaledDotProductAttention::instance().prepareForward(mixed, mixedForward, stream);
    CudnnAttentionExecutablePlan denseForwardPlan =
        CudnnScaledDotProductAttention::instance().prepareForward(dense, denseForward, stream);
    ASSERT_NO_THROW(CudnnScaledDotProductAttention::instance().forward(mixedForwardPlan, mixedForward, mixedForwardWorkspace, stream));
    ASSERT_NO_THROW(CudnnScaledDotProductAttention::instance().forward(denseForwardPlan, denseForward, denseForwardWorkspace, stream));
    stream.synchronize();

    const auto oMixedValues = copyHalfStorage(oMixed, queryCapacity * heads * dim, stream);
    const auto oDenseValues = copyHalfStorage(oDense, storageElements(dense.o), stream);
    for (uint64_t b = 0; b < batch; ++b) {
        const uint64_t length = static_cast<uint64_t>(queryLengths[b]);
        for (uint64_t s = 0; s < length; ++s) {
            const uint64_t t = static_cast<uint64_t>(queryRowOffsets[b]) + s;
            for (uint64_t h = 0; h < heads; ++h) {
                for (uint64_t d = 0; d < dim; ++d) {
                    EXPECT_NEAR(oMixedValues[thdIndex(t, h, d, heads, dim)],
                                oDenseValues[bshdIndex(b, s, h, d, queryCapacity, heads, dim)],
                                0.02f)
                        << "forward b=" << b << " s=" << s << " h=" << h << " d=" << d;
                }
            }
        }
    }
    const auto statsMixedValues = copyFloatStorage(statsMixed, stream);
    const auto statsDenseValues = copyFloatStorage(statsDense, stream);
    for (uint64_t b = 0; b < batch; ++b) {
        const uint64_t length = static_cast<uint64_t>(queryLengths[b]);
        for (uint64_t h = 0; h < heads; ++h) {
            for (uint64_t s = 0; s < length; ++s) {
                const uint64_t index = (b * heads + h) * queryCapacity + s;
                EXPECT_NEAR(statsMixedValues[index], statsDenseValues[index], 0.002f)
                    << "stats b=" << b << " h=" << h << " s=" << s;
            }
        }
    }
    EXPECT_EQ(copyInt32Storage(scratch.seqLenQ, stream), queryLengths);
    EXPECT_EQ(copyInt32Storage(scratch.seqLenKv, stream), kvLengths);
    EXPECT_EQ(copyInt32Storage(scratch.qElementOffsets, stream), (std::vector<int32_t>{0, 32, 80}));
    EXPECT_EQ(copyInt32Storage(scratch.oElementOffsets, stream), (std::vector<int32_t>{0, 32, 80}));
    EXPECT_EQ(copyInt32Storage(scratch.kElementOffsets, stream), (std::vector<int32_t>{0, 48, 96}));
    EXPECT_EQ(copyInt32Storage(scratch.vElementOffsets, stream), (std::vector<int32_t>{0, 48, 96}));

    std::vector<float> packedDO(queryCapacity * heads * dim, -17.0f);
    std::vector<float> denseDO(batch * queryCapacity * heads * dim, 19.0f);
    for (uint64_t b = 0; b < batch; ++b) {
        const uint64_t length = static_cast<uint64_t>(queryLengths[b]);
        for (uint64_t s = 0; s < length; ++s) {
            const uint64_t t = static_cast<uint64_t>(queryRowOffsets[b]) + s;
            for (uint64_t h = 0; h < heads; ++h) {
                for (uint64_t d = 0; d < dim; ++d) {
                    const float value = 0.01f * static_cast<float>(static_cast<int>((t + h + d) % 11) - 5);
                    packedDO[thdIndex(t, h, d, heads, dim)] = value;
                    denseDO[bshdIndex(b, s, h, d, queryCapacity, heads, dim)] = value;
                }
            }
        }
    }

    Tensor dOMixed = makePackedHalfTensor(queryCapacity, heads, dim, packedDO, stream);
    Tensor dODense = makeHalfTensorWithSpec(dense.o, denseDO, stream);
    Tensor dQMixed = makeEmptyPackedHalfTensor(queryCapacity, heads, dim);
    Tensor dQDense = makeEmptyHalfTensorWithSpec(dense.q);
    Tensor dKMixed = makeEmptyHalfTensorWithSpec(mixed.k);
    Tensor dVMixed = makeEmptyHalfTensorWithSpec(mixed.v);
    Tensor dKDense = makeEmptyHalfTensorWithSpec(dense.k);
    Tensor dVDense = makeEmptyHalfTensorWithSpec(dense.v);

    constexpr float gradientSentinel = 7.0f;
    dQMixed.fill(gradientSentinel, stream);
    dQDense.fill(gradientSentinel, stream);
    dKMixed.fill(gradientSentinel, stream);
    dVMixed.fill(gradientSentinel, stream);
    dKDense.fill(gradientSentinel, stream);
    dVDense.fill(gradientSentinel, stream);

    CudnnAttentionBackwardArgs mixedBackward{
        .q = qMixed,
        .k = kMixed,
        .v = vMixed,
        .o = oMixed,
        .dO = dOMixed,
        .stats = statsMixed,
        .dQ = dQMixed,
        .dK = dKMixed,
        .dV = dVMixed,
        .qRowPartitionOffsets = queryOffsets,
        .raggedScratch = scratch,
    };
    CudnnAttentionBackwardArgs denseBackward{
        .q = qDense,
        .k = kDense,
        .v = vDense,
        .o = oDense,
        .dO = dODense,
        .stats = statsDense,
        .dQ = dQDense,
        .dK = dKDense,
        .dV = dVDense,
        .seqLenQ = denseSeqQ,
        .seqLenKv = denseSeqKv,
    };

    std::optional<Tensor> mixedBackwardWorkspace = allocateAttentionWorkspace(mixed, true);
    std::optional<Tensor> denseBackwardWorkspace = allocateAttentionWorkspace(dense, true);
    CudnnAttentionExecutablePlan mixedBackwardPlan =
        CudnnScaledDotProductAttention::instance().prepareBackward(mixed, mixedBackward, stream);
    CudnnAttentionExecutablePlan denseBackwardPlan =
        CudnnScaledDotProductAttention::instance().prepareBackward(dense, denseBackward, stream);
    ASSERT_NO_THROW(CudnnScaledDotProductAttention::instance().backward(mixedBackwardPlan, mixedBackward, mixedBackwardWorkspace, stream));
    ASSERT_NO_THROW(CudnnScaledDotProductAttention::instance().backward(denseBackwardPlan, denseBackward, denseBackwardWorkspace, stream));
    stream.synchronize();

    const auto dQMixedValues = copyHalfStorage(dQMixed, queryCapacity * heads * dim, stream);
    const auto dQDenseValues = copyHalfStorage(dQDense, storageElements(dense.q), stream);
    for (uint64_t b = 0; b < batch; ++b) {
        const uint64_t length = static_cast<uint64_t>(queryLengths[b]);
        for (uint64_t s = 0; s < length; ++s) {
            const uint64_t t = static_cast<uint64_t>(queryRowOffsets[b]) + s;
            for (uint64_t h = 0; h < heads; ++h) {
                for (uint64_t d = 0; d < dim; ++d) {
                    const uint64_t mixedIndex = thdIndex(t, h, d, heads, dim);
                    const uint64_t denseIndex = bshdIndex(b, s, h, d, queryCapacity, heads, dim);
                    EXPECT_NE(dQMixedValues[mixedIndex], gradientSentinel) << "mixed dQ was not written at index=" << mixedIndex;
                    EXPECT_NEAR(dQMixedValues[mixedIndex], dQDenseValues[denseIndex], 0.03f)
                        << "dQ b=" << b << " s=" << s << " h=" << h << " d=" << d;
                }
            }
        }
    }

    const auto dKMixedValues = copyHalfStorage(dKMixed, storageElements(mixed.k), stream);
    const auto dVMixedValues = copyHalfStorage(dVMixed, storageElements(mixed.v), stream);
    const auto dKDenseValues = copyHalfStorage(dKDense, storageElements(dense.k), stream);
    const auto dVDenseValues = copyHalfStorage(dVDense, storageElements(dense.v), stream);
    expectNearVectors(dKMixedValues, dKDenseValues, 0.03f, "dK");
    expectNearVectors(dVMixedValues, dVDenseValues, 0.03f, "dV");
}
