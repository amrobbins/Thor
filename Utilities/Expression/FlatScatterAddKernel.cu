#include "Utilities/Expression/FlatScatterAddKernel.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

template <typename T>
struct FlatScatterAddPlus {
    __host__ __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs + rhs;
    }
};

#define CUSPARSE_CHECK(call)                                                                                                             \
    do {                                                                                                                                 \
        cusparseStatus_t status__ = (call);                                                                                              \
        if (status__ != CUSPARSE_STATUS_SUCCESS) {                                                                                       \
            const char* name__ = cusparseGetErrorName(status__);                                                                         \
            const char* desc__ = cusparseGetErrorString(status__);                                                                       \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #call + " failed with " +           \
                                     (name__ ? name__ : "CUSPARSE_STATUS_UNKNOWN") + " (" + std::to_string(static_cast<int>(status__)) + \
                                     "): " + (desc__ ? desc__ : "<no description>"));                                                    \
        }                                                                                                                                \
    } while (0)

[[nodiscard]] cudaDataType cudaValueTypeFor(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return CUDA_R_32F;
        default:
            throw std::runtime_error("Flat scatter-add currently supports only FP32 update/output tensors.");
    }
}

template <typename T>
__global__ void materializeFlatScatterAddPairsKernel(
    const T* updates, const uint32_t* flat_indices, uint32_t* keys, T* values, uint64_t num_updates, uint64_t output_numel) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_updates) {
        return;
    }

    const uint32_t dst = flat_indices[idx];
    if (dst == UINT32_MAX || static_cast<uint64_t>(dst) >= output_numel) {
        // Invalid/sentinel updates become harmless zeros at key 0.  This avoids
        // a separate select/filter step and keeps the sparse descriptor size
        // fixed for graph-capturable hot-path execution.
        keys[idx] = 0;
        values[idx] = T(0.0f);
    } else {
        keys[idx] = dst;
        values[idx] = updates[idx];
    }
}

template <typename T>
void launchMaterializeFlatScatterAddPairs(const Tensor& updates,
                                          const Tensor& flat_indices,
                                          Tensor& keys,
                                          Tensor& values,
                                          uint64_t num_updates,
                                          uint64_t output_numel,
                                          Stream& stream) {
    if (num_updates == 0) {
        return;
    }
    constexpr uint32_t threads_per_block = 256;
    const uint32_t blocks = static_cast<uint32_t>((num_updates + threads_per_block - 1) / threads_per_block);
    materializeFlatScatterAddPairsKernel<T><<<blocks, threads_per_block, 0, stream.getStream()>>>(updates.getMemPtr<T>(),
                                                                                                  flat_indices.getMemPtr<uint32_t>(),
                                                                                                  keys.getMemPtr<uint32_t>(),
                                                                                                  values.getMemPtr<T>(),
                                                                                                  num_updates,
                                                                                                  output_numel);
}

void launchMaterializeFlatScatterAddPairs(const Tensor& updates,
                                          const Tensor& flat_indices,
                                          Tensor& keys,
                                          Tensor& values,
                                          uint64_t num_updates,
                                          uint64_t output_numel,
                                          Stream& stream) {
    switch (updates.getDataType()) {
        case DataType::FP32:
            launchMaterializeFlatScatterAddPairs<float>(updates, flat_indices, keys, values, num_updates, output_numel, stream);
            break;
        default:
            throw std::runtime_error("Flat scatter-add currently supports only FP32 update tensors.");
    }
}

size_t querySortPairsTempBytes(
    const Tensor& keys_in, const Tensor& keys_out, const Tensor& values_in, const Tensor& values_out, uint64_t num_updates) {
    if (num_updates == 0) {
        return 1;
    }
    if (keys_in.getDataType() != DataType::UINT32 || keys_out.getDataType() != DataType::UINT32 ||
        values_in.getDataType() != DataType::FP32 || values_out.getDataType() != DataType::FP32) {
        throw std::runtime_error("Flat scatter-add sort expects UINT32 keys and FP32 values.");
    }
    const int cub_items = checkedCubNumItems(num_updates);
    size_t bytes = 1;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,
                                               bytes,
                                               keys_in.getMemPtr<uint32_t>(),
                                               const_cast<uint32_t*>(keys_out.getMemPtr<uint32_t>()),
                                               values_in.getMemPtr<float>(),
                                               const_cast<float*>(values_out.getMemPtr<float>()),
                                               cub_items,
                                               0,
                                               static_cast<int>(8 * sizeof(uint32_t))));
    return std::max<size_t>(bytes, 1);
}

void runSortPairs(BuiltFlatScatterAdd& built, Stream& stream) {
    if (built.num_updates == 0) {
        return;
    }
    const int cub_items = checkedCubNumItems(built.num_updates);
    void* temp_ptr = mutableCubTempStoragePtr(built.sort_temp_storage);
    size_t temp_bytes = built.sort_temp_storage_bytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp_ptr,
                                               temp_bytes,
                                               built.materialized_keys.getMemPtr<uint32_t>(),
                                               built.sorted_keys.getMemPtr<uint32_t>(),
                                               built.materialized_values.getMemPtr<float>(),
                                               built.sorted_values.getMemPtr<float>(),
                                               cub_items,
                                               0,
                                               static_cast<int>(8 * sizeof(uint32_t)),
                                               stream.getStream()));
}

template <typename T>
size_t queryReduceByKeyTempBytes(const Tensor& sorted_keys,
                                 const Tensor& unique_keys,
                                 const Tensor& sorted_values,
                                 const Tensor& reduced_values,
                                 const Tensor& num_unique,
                                 uint64_t num_updates) {
    if (num_updates == 0) {
        return 1;
    }
    const int cub_items = checkedCubNumItems(num_updates);
    size_t bytes = 1;
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(nullptr,
                                              bytes,
                                              sorted_keys.getMemPtr<uint32_t>(),
                                              const_cast<uint32_t*>(unique_keys.getMemPtr<uint32_t>()),
                                              sorted_values.getMemPtr<T>(),
                                              const_cast<T*>(reduced_values.getMemPtr<T>()),
                                              const_cast<uint32_t*>(num_unique.getMemPtr<uint32_t>()),
                                              FlatScatterAddPlus<T>(),
                                              cub_items));
    return std::max<size_t>(bytes, 1);
}

size_t queryReduceByKeyTempBytes(const Tensor& sorted_keys,
                                 const Tensor& unique_keys,
                                 const Tensor& sorted_values,
                                 const Tensor& reduced_values,
                                 const Tensor& num_unique,
                                 uint64_t num_updates) {
    switch (sorted_values.getDataType()) {
        case DataType::FP32:
            return queryReduceByKeyTempBytes<float>(sorted_keys, unique_keys, sorted_values, reduced_values, num_unique, num_updates);
        default:
            throw std::runtime_error("Flat scatter-add currently supports only FP32 reduced values.");
    }
}

template <typename T>
void runReduceByKey(BuiltFlatScatterAdd& built, Stream& stream) {
    if (built.num_updates == 0) {
        return;
    }
    const int cub_items = checkedCubNumItems(built.num_updates);
    void* temp_ptr = (void*)built.reduce_temp_storage.getMemPtr<void>();
    size_t temp_bytes = built.reduce_temp_storage_bytes;
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(temp_ptr,
                                              temp_bytes,
                                              built.sorted_keys.getMemPtr<uint32_t>(),
                                              built.unique_keys.getMemPtr<uint32_t>(),
                                              built.sorted_values.getMemPtr<T>(),
                                              built.reduced_values.getMemPtr<T>(),
                                              built.num_unique.getMemPtr<uint32_t>(),
                                              FlatScatterAddPlus<T>(),
                                              cub_items,
                                              stream.getStream()));
}

void runReduceByKey(BuiltFlatScatterAdd& built, Stream& stream) {
    switch (built.value_dtype) {
        case DataType::FP32:
            runReduceByKey<float>(built, stream);
            break;
        default:
            throw std::runtime_error("Flat scatter-add currently supports only FP32 reduced values.");
    }
}

}  // namespace

BuiltFlatScatterAdd::~BuiltFlatScatterAdd() {
    if (sparse_vec != nullptr) {
        (void)cusparseDestroySpVec(sparse_vec);
        sparse_vec = nullptr;
    }
    if (dense_vec != nullptr) {
        (void)cusparseDestroyDnVec(dense_vec);
        dense_vec = nullptr;
    }
    if (cusparse_handle != nullptr) {
        (void)cusparseDestroy(cusparse_handle);
        cusparse_handle = nullptr;
    }
}

std::shared_ptr<BuiltFlatScatterAdd> prepareFlatScatterAdd(const Tensor& updates,
                                                           const Tensor& flat_indices,
                                                           Tensor& output,
                                                           FlatScatterAddIndexPolicy index_policy) {
    if (updates.getPlacement() != flat_indices.getPlacement() || updates.getPlacement() != output.getPlacement()) {
        throw std::runtime_error("Flat scatter-add updates, indices, and output must be on the same placement.");
    }
    if (flat_indices.getDataType() != DataType::UINT32) {
        throw std::runtime_error("Flat scatter-add indices must be UINT32.");
    }
    if (updates.getDataType() != output.getDataType()) {
        throw std::runtime_error("Flat scatter-add updates and output must have the same dtype.");
    }
    if (updates.getDataType() != DataType::FP32) {
        throw std::runtime_error("Flat scatter-add currently supports only FP32 update/output tensors.");
    }
    if (updates.getDimensions() != flat_indices.getDimensions()) {
        throw std::runtime_error("Flat scatter-add updates and indices must have the same shape.");
    }

    auto built = std::make_shared<BuiltFlatScatterAdd>();
    built->num_updates = updates.getTotalNumElements();
    built->output_numel = output.getTotalNumElements();
    built->value_dtype = updates.getDataType();
    built->index_policy = index_policy;

    const TensorPlacement placement = output.getPlacement();
    const TensorDescriptor key_desc(DataType::UINT32, {std::max<uint64_t>(built->num_updates, 1)});
    const TensorDescriptor value_desc(built->value_dtype, {std::max<uint64_t>(built->num_updates, 1)});

    const cudaDataType value_type = cudaValueTypeFor(built->value_dtype);
    CUSPARSE_CHECK(cusparseCreate(&built->cusparse_handle));
    CUSPARSE_CHECK(cusparseSetPointerMode(built->cusparse_handle, CUSPARSE_POINTER_MODE_HOST));

    if (index_policy == FlatScatterAddIndexPolicy::CanonicalizeDuplicatesAndSkipInvalid) {
        built->materialized_keys = Tensor(placement, key_desc);
        built->materialized_values = Tensor(placement, value_desc);
        built->sorted_keys = Tensor(placement, key_desc);
        built->sorted_values = Tensor(placement, value_desc);
        built->unique_keys = Tensor(placement, key_desc);
        built->reduced_values = Tensor(placement, value_desc);
        built->num_unique = Tensor(placement, TensorDescriptor(DataType::UINT32, {1}));

        built->sort_temp_storage_bytes = querySortPairsTempBytes(
            built->materialized_keys, built->sorted_keys, built->materialized_values, built->sorted_values, built->num_updates);
        built->sort_temp_storage = allocateCubTemporaryStorage(placement, built->sort_temp_storage_bytes);

        built->reduce_temp_storage_bytes = queryReduceByKeyTempBytes(
            built->sorted_keys, built->unique_keys, built->sorted_values, built->reduced_values, built->num_unique, built->num_updates);
        built->reduce_temp_storage = allocateCubTemporaryStorage(placement, built->reduce_temp_storage_bytes);

        CUSPARSE_CHECK(cusparseCreateSpVec(&built->sparse_vec,
                                           static_cast<int64_t>(built->output_numel),
                                           static_cast<int64_t>(std::max<uint64_t>(built->num_updates, 1)),
                                           built->unique_keys.getMemPtr<uint32_t>(),
                                           built->reduced_values.getMemPtr<float>(),
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           value_type));
    } else if (index_policy == FlatScatterAddIndexPolicy::IndicesAreUniqueAndValid) {
        if (built->num_updates == 0) {
            // The run path returns before calling cuSPARSE for empty updates,
            // but create a valid descriptor anyway so plan construction remains
            // uniform and destructor-safe.
            built->materialized_keys = Tensor(placement, key_desc);
            built->materialized_values = Tensor(placement, value_desc);
            CUSPARSE_CHECK(cusparseCreateSpVec(&built->sparse_vec,
                                               static_cast<int64_t>(built->output_numel),
                                               1,
                                               built->materialized_keys.getMemPtr<uint32_t>(),
                                               built->materialized_values.getMemPtr<float>(),
                                               CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO,
                                               value_type));
        } else {
            CUSPARSE_CHECK(cusparseCreateSpVec(&built->sparse_vec,
                                               static_cast<int64_t>(built->output_numel),
                                               static_cast<int64_t>(built->num_updates),
                                               const_cast<uint32_t*>(flat_indices.getMemPtr<uint32_t>()),
                                               const_cast<float*>(updates.getMemPtr<float>()),
                                               CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO,
                                               value_type));
        }
    } else {
        throw std::runtime_error("Unknown flat scatter-add index policy.");
    }

    CUSPARSE_CHECK(
        cusparseCreateDnVec(&built->dense_vec, static_cast<int64_t>(built->output_numel), output.getMemPtr<float>(), value_type));
    return built;
}

void runFlatScatterAdd(
    const std::shared_ptr<BuiltFlatScatterAdd>& built, const Tensor& updates, const Tensor& flat_indices, Tensor& output, Stream& stream) {
    if (!built) {
        throw std::runtime_error("runFlatScatterAdd requires a prepared plan.");
    }
    if (updates.getTotalNumElements() != built->num_updates || flat_indices.getTotalNumElements() != built->num_updates ||
        output.getTotalNumElements() != built->output_numel) {
        throw std::runtime_error("Flat scatter-add runtime tensors do not match the prepared plan shape.");
    }
    if (updates.getDataType() != built->value_dtype || output.getDataType() != built->value_dtype ||
        flat_indices.getDataType() != DataType::UINT32) {
        throw std::runtime_error("Flat scatter-add runtime tensor dtypes do not match the prepared plan.");
    }

    if (built->num_updates == 0) {
        output.memsetAsync(stream, 0);
        return;
    }

    if (built->index_policy == FlatScatterAddIndexPolicy::CanonicalizeDuplicatesAndSkipInvalid) {
        built->unique_keys.memsetAsync(stream, 0);
        built->reduced_values.memsetAsync(stream, 0);
        built->num_unique.memsetAsync(stream, 0);

        launchMaterializeFlatScatterAddPairs(
            updates, flat_indices, built->materialized_keys, built->materialized_values, built->num_updates, built->output_numel, stream);

        runSortPairs(*built, stream);

        runReduceByKey(*built, stream);
    } else if (built->index_policy != FlatScatterAddIndexPolicy::IndicesAreUniqueAndValid) {
        throw std::runtime_error("Unknown flat scatter-add index policy.");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUSPARSE_CHECK(cusparseSetStream(built->cusparse_handle, stream.getStream()));

    // CUDA 13.3 marks cusparseAxpby as deprecated, but it remains the
    // vendor primitive that exactly matches Thor's canonicalized sparse
    // flat-update -> dense-vector accumulation hot path. Keep the warning
    // suppressed locally so Thor's project-wide -Werror policy does not make
    // this translation unit fail to compile while this backend remains in use.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    CUSPARSE_CHECK(cusparseAxpby(built->cusparse_handle, &alpha, built->sparse_vec, &beta, built->dense_vec));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

}  // namespace ThorImplementation
