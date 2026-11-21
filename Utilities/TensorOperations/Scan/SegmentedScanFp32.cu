// The segmented scans writing FP32 or UINT32 values (SegmentedScanKernel.cuh; the arg scans: SegmentedScanArg.cu).

#include "Utilities/TensorOperations/Scan/SegmentedScanKernel.cuh"

namespace ThorImplementation::SegmentedScanKernels {

THOR_SEGMENTED_SCAN_INSTANTIATE(float, float, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(float, float, Max);
THOR_SEGMENTED_SCAN_INSTANTIATE(float, float, Min);
THOR_SEGMENTED_SCAN_INSTANTIATE(float, float, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(uint32_t, uint32_t, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(uint32_t, uint32_t, Max);
THOR_SEGMENTED_SCAN_INSTANTIATE(uint32_t, uint32_t, Min);
THOR_SEGMENTED_SCAN_INSTANTIATE(uint32_t, uint32_t, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__half, float, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_bfloat16, float, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e4m3, float, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e5m2, float, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__half, float, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_bfloat16, float, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e4m3, float, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e5m2, float, Product);

}  // namespace ThorImplementation::SegmentedScanKernels
