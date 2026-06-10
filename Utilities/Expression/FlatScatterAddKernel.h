#pragma once

#include <cusparse.h>

#include <cstdint>
#include <memory>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitives.h"

namespace ThorImplementation {

// Internal no-atomic flat scatter-add plan used by scan min/max backward
// and future indexing/scatter-style expression stages.
//
// CanonicalizeDuplicatesAndSkipInvalid semantics:
//   output[:] = 0
//   for i in [0, updates.numel):
//     dst = flat_indices[i]
//     if dst != UINT32_MAX and dst < output.numel:
//       output[dst] += updates[i]
// Duplicate destinations are canonicalized by CUB sort + reduce-by-key, then
// the unique sparse updates are applied to the dense output with cuSPARSE Axpby.
//
// IndicesAreUniqueAndValid semantics are the same, but the caller guarantees
// every flat index is in-bounds, non-sentinel, and appears at most once.  That
// lets the hot path skip materialize/sort/reduce and feed the caller's update
// and index tensors directly to cuSPARSE Axpby.
enum class FlatScatterAddIndexPolicy { CanonicalizeDuplicatesAndSkipInvalid, IndicesAreUniqueAndValid };

struct BuiltFlatScatterAdd {
    Tensor materialized_keys;
    Tensor materialized_values;
    Tensor sorted_keys;
    Tensor sorted_values;
    Tensor unique_keys;
    Tensor reduced_values;
    Tensor num_unique;
    Tensor sort_temp_storage;
    Tensor reduce_temp_storage;

    size_t sort_temp_storage_bytes = 0;
    size_t reduce_temp_storage_bytes = 0;
    uint64_t num_updates = 0;
    uint64_t output_numel = 0;
    DataType value_dtype = DataType::FP32;
    FlatScatterAddIndexPolicy index_policy = FlatScatterAddIndexPolicy::CanonicalizeDuplicatesAndSkipInvalid;

    cusparseHandle_t cusparse_handle = nullptr;
    cusparseSpVecDescr_t sparse_vec = nullptr;
    cusparseDnVecDescr_t dense_vec = nullptr;

    BuiltFlatScatterAdd() = default;
    BuiltFlatScatterAdd(const BuiltFlatScatterAdd&) = delete;
    BuiltFlatScatterAdd& operator=(const BuiltFlatScatterAdd&) = delete;
    BuiltFlatScatterAdd(BuiltFlatScatterAdd&&) = delete;
    BuiltFlatScatterAdd& operator=(BuiltFlatScatterAdd&&) = delete;
    ~BuiltFlatScatterAdd();
};

[[nodiscard]] std::shared_ptr<BuiltFlatScatterAdd> prepareFlatScatterAdd(
    const Tensor& updates,
    const Tensor& flat_indices,
    Tensor& output,
    FlatScatterAddIndexPolicy index_policy = FlatScatterAddIndexPolicy::CanonicalizeDuplicatesAndSkipInvalid);

void runFlatScatterAdd(const std::shared_ptr<BuiltFlatScatterAdd>& built,
                       const Tensor& updates,
                       const Tensor& flat_indices,
                       Tensor& output,
                       Stream& stream);

}  // namespace ThorImplementation
