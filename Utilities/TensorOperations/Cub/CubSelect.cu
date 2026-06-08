#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

template <typename T, typename FlagT>
size_t querySelectFlaggedBytes(const Tensor& input,
                               const Tensor& flags,
                               const Tensor& output,
                               const Tensor& num_selected_out,
                               int cub_items) {
    size_t queried_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr,
                                          queried_bytes,
                                          input.getMemPtr<T>(),
                                          flags.getMemPtr<FlagT>(),
                                          const_cast<T*>(output.getMemPtr<T>()),
                                          const_cast<uint32_t*>(num_selected_out.getMemPtr<uint32_t>()),
                                          cub_items));
    return queried_bytes;
}

template <typename T, typename FlagT>
void launchSelectFlagged(void* temp_storage_ptr,
                         size_t temp_storage_bytes,
                         const Tensor& input,
                         const Tensor& flags,
                         Tensor& output,
                         Tensor& num_selected_out,
                         int cub_items,
                         Stream& stream) {
    CUDA_CHECK(cub::DeviceSelect::Flagged(temp_storage_ptr,
                                          temp_storage_bytes,
                                          input.getMemPtr<T>(),
                                          flags.getMemPtr<FlagT>(),
                                          output.getMemPtr<T>(),
                                          num_selected_out.getMemPtr<uint32_t>(),
                                          cub_items,
                                          stream.getStream()));
}

}  // namespace

CubDeviceSelectFlaggedPlan prepareCubDeviceSelectFlagged(const Tensor& input,
                                                         const Tensor& flags,
                                                         const Tensor& output,
                                                         const Tensor& num_selected_out,
                                                         uint64_t num_items) {
    validateSelectFlagged(input, flags, output, num_selected_out, num_items);
    const int cub_items = checkedCubNumItems(num_items);

    size_t bytes = 1;
    if (num_items != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            auto query_flag = [&]<typename FlagT>() -> size_t {
                return querySelectFlaggedBytes<T, FlagT>(input, flags, output, num_selected_out, cub_items);
            };
            return dispatchSelectFlagDType(flags.getDataType(), query_flag);
        };
        bytes = dispatchSelectValueDType(input.getDataType(), query_value);
    }

    CubDeviceSelectFlaggedPlan plan;
    plan.placement = input.getPlacement();
    plan.input_dtype = input.getDataType();
    plan.flag_dtype = flags.getDataType();
    plan.num_items = num_items;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSelectFlaggedTempBytes(const Tensor& input,
                                       const Tensor& flags,
                                       const Tensor& output,
                                       const Tensor& num_selected_out,
                                       uint64_t num_items) {
    return prepareCubDeviceSelectFlagged(input, flags, output, num_selected_out, num_items).temp_storage_bytes;
}

void cubDeviceSelectFlagged(const CubDeviceSelectFlaggedPlan& plan,
                            const Tensor& temp_storage,
                            const Tensor& input,
                            const Tensor& flags,
                            Tensor& output,
                            Tensor& num_selected_out,
                            Stream& stream) {
    validateSelectFlagged(input, flags, output, num_selected_out, plan.num_items);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.input_dtype || flags.getDataType() != plan.flag_dtype) {
        throw std::invalid_argument("CUB select-flagged plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int cub_items = checkedCubNumItems(plan.num_items);
    if (plan.num_items == 0) {
        CUDA_CHECK(cudaMemsetAsync(num_selected_out.getMemPtr<uint32_t>(), 0, sizeof(uint32_t), stream.getStream()));
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch_value = [&]<typename T>() -> void {
        auto launch_flag = [&]<typename FlagT>() -> void {
            launchSelectFlagged<T, FlagT>(temp_storage_ptr, temp_storage_bytes, input, flags, output, num_selected_out, cub_items, stream);
        };
        dispatchSelectFlagDType(plan.flag_dtype, launch_flag);
    };

    dispatchSelectValueDType(plan.input_dtype, launch_value);
}

void cubDeviceSelectFlagged(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& input,
                            const Tensor& flags,
                            Tensor& output,
                            Tensor& num_selected_out,
                            uint64_t num_items,
                            Stream& stream) {
    CubDeviceSelectFlaggedPlan plan = prepareCubDeviceSelectFlagged(input, flags, output, num_selected_out, num_items);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB select-flagged requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceSelectFlagged(plan, temp_storage, input, flags, output, num_selected_out, stream);
}

}  // namespace ThorImplementation
