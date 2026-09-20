#include "Utilities/TensorOperations/Cub/CubArgReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

#include <limits>

namespace ThorImplementation::CubReductionInternal {

size_t queryArgMinReductionBytes(const Tensor& input,
                                 Tensor* value_output,
                                 Tensor* index_output,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream) {
    return queryOperationArgReductionBytes(input,
                                           value_output,
                                           index_output,
                                           geometry,
                                           ArgMinimumCandidateFp32{},
                                           ArgReductionCandidateFp32{std::numeric_limits<uint64_t>::max(),
                                                                     std::numeric_limits<float>::infinity()},
                                           stream);
}

size_t queryArgMinReductionBytes(DataType input_dtype,
                                 std::optional<DataType> value_output_dtype,
                                 std::optional<DataType> index_output_dtype,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream) {
    return queryOperationArgReductionBytes(input_dtype,
                                           value_output_dtype,
                                           index_output_dtype,
                                           geometry,
                                           ArgMinimumCandidateFp32{},
                                           ArgReductionCandidateFp32{std::numeric_limits<uint64_t>::max(),
                                                                     std::numeric_limits<float>::infinity()},
                                           stream);
}

void launchArgMinReduction(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor* value_output,
                           Tensor* index_output,
                           const CubReductionGeometry& geometry,
                           Stream& stream) {
    launchOperationArgReduction(temp_storage,
                                temp_storage_bytes,
                                input,
                                value_output,
                                index_output,
                                geometry,
                                ArgMinimumCandidateFp32{},
                                ArgReductionCandidateFp32{std::numeric_limits<uint64_t>::max(),
                                                          std::numeric_limits<float>::infinity()},
                                stream);
}


size_t queryComposedArgMinReductionStageBytes(const Tensor& value_input,
                                               const Tensor* carried_index_input,
                                               Tensor* value_output,
                                               Tensor* index_output,
                                               const CubReductionGeometry& geometry,
                                               uint64_t domain_stride,
                                               DataType carried_index_dtype,
                                               const Stream& stream) {
    return queryComposedOperationArgReductionStageBytes(
        value_input,
        carried_index_input,
        value_output,
        index_output,
        geometry,
        domain_stride,
        carried_index_dtype,
        ArgMinimumCandidateFp32{},
        ArgReductionCandidateFp32{std::numeric_limits<uint64_t>::max(), std::numeric_limits<float>::infinity()},
        stream);
}

size_t queryComposedArgMinReductionStageBytes(DataType value_input_dtype,
                                               bool has_carried_index_input,
                                               std::optional<DataType> value_output_dtype,
                                               std::optional<DataType> index_output_dtype,
                                               const CubReductionGeometry& geometry,
                                               uint64_t domain_stride,
                                               DataType carried_index_dtype,
                                               const Stream& stream) {
    return queryComposedOperationArgReductionStageBytes(
        value_input_dtype,
        has_carried_index_input,
        value_output_dtype,
        index_output_dtype,
        geometry,
        domain_stride,
        carried_index_dtype,
        ArgMinimumCandidateFp32{},
        ArgReductionCandidateFp32{std::numeric_limits<uint64_t>::max(), std::numeric_limits<float>::infinity()},
        stream);
}

void launchComposedArgMinReductionStage(const Tensor& temp_storage,
                                         size_t temp_storage_bytes,
                                         const Tensor& value_input,
                                         const Tensor* carried_index_input,
                                         Tensor* value_output,
                                         Tensor* index_output,
                                         const CubReductionGeometry& geometry,
                                         uint64_t domain_stride,
                                         DataType carried_index_dtype,
                                         Stream& stream) {
    launchComposedOperationArgReductionStage(
        temp_storage,
        temp_storage_bytes,
        value_input,
        carried_index_input,
        value_output,
        index_output,
        geometry,
        domain_stride,
        carried_index_dtype,
        ArgMinimumCandidateFp32{},
        ArgReductionCandidateFp32{std::numeric_limits<uint64_t>::max(), std::numeric_limits<float>::infinity()},
        stream);
}

}  // namespace ThorImplementation::CubReductionInternal
