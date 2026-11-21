// The segmented scans writing FP8 (SegmentedScanKernel.cuh): min and max in their own type, and the float sums and
// products, whose kernels write SharedFloat8, either format (the launch picks it).

#include "Utilities/TensorOperations/Scan/SegmentedScanKernel.cuh"

namespace ThorImplementation::SegmentedScanKernels {

THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e4m3, __nv_fp8_e4m3, Max);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e4m3, __nv_fp8_e4m3, Min);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e5m2, __nv_fp8_e5m2, Max);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e5m2, __nv_fp8_e5m2, Min);
THOR_SEGMENTED_SCAN_INSTANTIATE(float, SharedFloat8, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_bfloat16, SharedFloat8, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__half, SharedFloat8, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e5m2, SharedFloat8, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e4m3, SharedFloat8, Sum);
THOR_SEGMENTED_SCAN_INSTANTIATE(float, SharedFloat8, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_bfloat16, SharedFloat8, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__half, SharedFloat8, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e5m2, SharedFloat8, Product);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e4m3, SharedFloat8, Product);

}  // namespace ThorImplementation::SegmentedScanKernels
