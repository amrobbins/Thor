#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;
}

CubDeviceExclusiveSumPlan prepareCubDeviceExclusiveSum(const Tensor& input,
                                                       const Tensor& output,
                                                       uint64_t num_items) {
    validateExclusiveSum(input, output, num_items);
    const int cub_items = checkedCubNumItems(num_items);

    size_t bytes = 1;
    if (num_items != 0) {
        auto query = [&]<typename T>() -> size_t {
            size_t queried_bytes = 0;
            CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,
                                                     queried_bytes,
                                                     input.getMemPtr<T>(),
                                                     const_cast<T*>(output.getMemPtr<T>()),
                                                     cub_items));
            return queried_bytes;
        };
        bytes = dispatchScanDType(input.getDataType(), query);
    }

    CubDeviceExclusiveSumPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.num_items = num_items;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceExclusiveSumTempBytes(const Tensor& input, const Tensor& output, uint64_t num_items) {
    return prepareCubDeviceExclusiveSum(input, output, num_items).temp_storage_bytes;
}

void cubDeviceExclusiveSum(const CubDeviceExclusiveSumPlan& plan,
                           const Tensor& temp_storage,
                           const Tensor& input,
                           Tensor& output,
                           Stream& stream) {
    validateExclusiveSum(input, output, plan.num_items);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB exclusive-sum plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int cub_items = checkedCubNumItems(plan.num_items);
    if (plan.num_items == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch = [&]<typename T>() -> void {
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp_storage_ptr,
                                                 temp_storage_bytes,
                                                 input.getMemPtr<T>(),
                                                 output.getMemPtr<T>(),
                                                 cub_items,
                                                 stream.getStream()));
    };

    dispatchScanDType(plan.dtype, launch);
}

void cubDeviceExclusiveSum(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor& output,
                           uint64_t num_items,
                           Stream& stream) {
    CubDeviceExclusiveSumPlan plan = prepareCubDeviceExclusiveSum(input, output, num_items);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB exclusive-sum requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceExclusiveSum(plan, temp_storage, input, output, stream);
}

}  // namespace ThorImplementation
