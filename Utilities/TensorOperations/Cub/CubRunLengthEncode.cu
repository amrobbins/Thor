#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

[[nodiscard]] bool isFp8RleDType(DataType dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
    return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2;
#else
    static_cast<void>(dtype);
    return false;
#endif
}

}  // namespace

CubDeviceRunLengthEncodePlan prepareCubDeviceRunLengthEncode(const Tensor& input,
                                                             const Tensor& unique_out,
                                                             const Tensor& counts_out,
                                                             const Tensor& num_runs_out,
                                                             uint64_t num_items) {
    validateRle(input, unique_out, counts_out, num_runs_out, num_items);
    const int cub_items = checkedCubNumItems(num_items);

    size_t bytes = 1;
    if (num_items != 0) {
        auto query = [&]<typename T>() -> size_t {
            size_t queried_bytes = 0;
            CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(nullptr,
                                                          queried_bytes,
                                                          input.getMemPtr<T>(),
                                                          const_cast<T*>(unique_out.getMemPtr<T>()),
                                                          const_cast<uint32_t*>(counts_out.getMemPtr<uint32_t>()),
                                                          const_cast<uint32_t*>(num_runs_out.getMemPtr<uint32_t>()),
                                                          cub_items));
            return queried_bytes;
        };
        if (isFp8RleDType(input.getDataType())) {
            // CUB radix sort accepts CUDA FP8 key types on the Thor-supported CUDA/CCCL surface,
            // but DeviceRunLengthEncode's default equality path does not currently instantiate
            // cuda::std::equal_to for __nv_fp8_* directly. FP8 values are one byte, so encode
            // them through an equivalent uint8_t bitwise path and write the unique byte patterns
            // back into the FP8 output buffer.  Use getMemPtr<void>() for this reinterpretation
            // so Tensor's descriptor/type guard still rejects accidental typed access elsewhere.
            const auto* input_bytes = static_cast<const uint8_t*>(input.getMemPtr<void>());
            auto* unique_bytes = const_cast<uint8_t*>(static_cast<const uint8_t*>(unique_out.getMemPtr<void>()));
            size_t queried_bytes = 0;
            CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(nullptr,
                                                          queried_bytes,
                                                          input_bytes,
                                                          unique_bytes,
                                                          const_cast<uint32_t*>(counts_out.getMemPtr<uint32_t>()),
                                                          const_cast<uint32_t*>(num_runs_out.getMemPtr<uint32_t>()),
                                                          cub_items));
            bytes = queried_bytes;
        } else {
            bytes = dispatchRleDType(input.getDataType(), query);
        }
    }

    CubDeviceRunLengthEncodePlan plan;
    plan.placement = input.getPlacement();
    plan.input_dtype = input.getDataType();
    plan.num_items = num_items;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceRunLengthEncodeTempBytes(const Tensor& input,
                                         const Tensor& unique_out,
                                         const Tensor& counts_out,
                                         const Tensor& num_runs_out,
                                         uint64_t num_items) {
    return prepareCubDeviceRunLengthEncode(input, unique_out, counts_out, num_runs_out, num_items).temp_storage_bytes;
}

void cubDeviceRunLengthEncode(const CubDeviceRunLengthEncodePlan& plan,
                              const Tensor& temp_storage,
                              const Tensor& input,
                              Tensor& unique_out,
                              Tensor& counts_out,
                              Tensor& num_runs_out,
                              Stream& stream) {
    validateRle(input, unique_out, counts_out, num_runs_out, plan.num_items);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.input_dtype) {
        throw std::invalid_argument("CUB run-length-encode plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int cub_items = checkedCubNumItems(plan.num_items);
    if (plan.num_items == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch = [&]<typename T>() -> void {
        CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(temp_storage_ptr,
                                                      temp_storage_bytes,
                                                      input.getMemPtr<T>(),
                                                      unique_out.getMemPtr<T>(),
                                                      counts_out.getMemPtr<uint32_t>(),
                                                      num_runs_out.getMemPtr<uint32_t>(),
                                                      cub_items,
                                                      stream.getStream()));
    };

    if (isFp8RleDType(plan.input_dtype)) {
        // See the preparation path above: run FP8 RLE through byte-wise equality.
        const auto* input_bytes = static_cast<const uint8_t*>(input.getMemPtr<void>());
        auto* unique_bytes = static_cast<uint8_t*>(unique_out.getMemPtr<void>());
        CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(temp_storage_ptr,
                                                      temp_storage_bytes,
                                                      input_bytes,
                                                      unique_bytes,
                                                      counts_out.getMemPtr<uint32_t>(),
                                                      num_runs_out.getMemPtr<uint32_t>(),
                                                      cub_items,
                                                      stream.getStream()));
    } else {
        dispatchRleDType(plan.input_dtype, launch);
    }
}

void cubDeviceRunLengthEncode(const Tensor& temp_storage,
                              size_t temp_storage_bytes,
                              const Tensor& input,
                              Tensor& unique_out,
                              Tensor& counts_out,
                              Tensor& num_runs_out,
                              uint64_t num_items,
                              Stream& stream) {
    CubDeviceRunLengthEncodePlan plan = prepareCubDeviceRunLengthEncode(input, unique_out, counts_out, num_runs_out, num_items);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB RLE requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceRunLengthEncode(plan, temp_storage, input, unique_out, counts_out, num_runs_out, stream);
}

}  // namespace ThorImplementation
