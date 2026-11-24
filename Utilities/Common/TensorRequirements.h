#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <stdexcept>
#include <string>

// Argument checks shared by the reducers, the ragged kernels and RowPartition.
namespace ThorImplementation::TensorRequirements {

[[nodiscard]] std::string dtypeName(DataType dtype);

void requireDenseContiguousGpuTensor(const Tensor& tensor, const char* name);
void requireSameGpuPlacement(const Tensor& a, const Tensor& b, const char* a_name, const char* b_name);
void requireStorageForNumItems(const Tensor& tensor, const char* name, uint64_t num_items);
void requireTempStorage(const Tensor& temp_storage, const TensorPlacement& placement, size_t temp_storage_bytes);

}  // namespace ThorImplementation::TensorRequirements
