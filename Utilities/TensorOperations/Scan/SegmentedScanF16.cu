// The segmented scans writing FP16 or BF16 (SegmentedScanKernel.cuh): min and max in their own type, and the float sums
// and products, whose kernels write SharedFloat16, either format (the launch picks it).

#include "Utilities/TensorOperations/Scan/SegmentedScanKernel.cuh"

namespace ThorImplementation::SegmentedScanKernels {

THOR_SEGMENTED_SCAN_INSTANTIATE(__half, __half, Max);
THOR_SEGMENTED_SCAN_INSTANTIATE(__half, __half, Min);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_bfloat16, __nv_bfloat16, Max);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_bfloat16, __nv_bfloat16, Min);
THOR_SEGMENTED_SCAN_INSTANTIATE(float, SharedFloat16, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_bfloat16, SharedFloat16, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__half, SharedFloat16, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e5m2, SharedFloat16, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e4m3, SharedFloat16, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(float, SharedFloat16, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_bfloat16, SharedFloat16, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__half, SharedFloat16, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e5m2, SharedFloat16, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e4m3, SharedFloat16, Product);

}  // namespace ThorImplementation::SegmentedScanKernels
