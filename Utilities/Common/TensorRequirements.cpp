#include "Utilities/Common/TensorRequirements.h"

namespace ThorImplementation::TensorRequirements {

std::string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

void requireDenseContiguousGpuTensor(const Tensor& tensor, const char* name) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string(name) + " must be initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string(name) + " must be a GPU tensor.");
    }
    if (!tensor.isDenseContiguous()) {
        throw std::invalid_argument(std::string(name) + " must be dense contiguous.");
    }
}

void requireSameGpuPlacement(const Tensor& a, const Tensor& b, const char* a_name, const char* b_name) {
    if (a.getPlacement() != b.getPlacement()) {
        throw std::invalid_argument(std::string(a_name) + " and " + b_name + " must live on the same GPU placement.");
    }
}

void requireStorageForNumItems(const Tensor& tensor, const char* name, uint64_t num_items) {
    if (tensor.getTotalNumElements() < num_items) {
        throw std::invalid_argument(std::string(name) + " has fewer elements than num_items.");
    }
}

void requireTempStorage(const Tensor& temp_storage, const TensorPlacement& placement, size_t temp_storage_bytes) {
    requireDenseContiguousGpuTensor(temp_storage, "temp_storage");
    if (temp_storage.getPlacement() != placement) {
        throw std::invalid_argument("temp_storage must live on the same GPU placement as the operation tensors.");
    }
    if (temp_storage.getDataType() != DataType::UINT8) {
        throw std::invalid_argument("temp_storage must have dtype uint8.");
    }
    if (temp_storage.getArraySizeInBytes() < temp_storage_bytes) {
        throw std::invalid_argument("temp_storage has fewer bytes than temp_storage_bytes.");
    }
}

}  // namespace ThorImplementation::TensorRequirements
