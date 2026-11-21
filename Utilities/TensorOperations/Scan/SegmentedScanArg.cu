// The segmented argmax and argmin scans: every input dtype, writing UINT32 indices (SegmentedScanKernel.cuh). One
// kernel serves both (OP ArgMax; argmin is Launch::arg_min).

#include "Utilities/TensorOperations/Scan/SegmentedScanKernel.cuh"

namespace ThorImplementation::SegmentedScanKernels {

THOR_SEGMENTED_SCAN_INSTANTIATE(float, uint32_t, ArgMax);
THOR_SEGMENTED_SCAN_INSTANTIATE(uint32_t, uint32_t, ArgMax);
THOR_SEGMENTED_SCAN_INSTANTIATE(__half, uint32_t, ArgMax);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_bfloat16, uint32_t, ArgMax);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e4m3, uint32_t, ArgMax);
THOR_SEGMENTED_SCAN_INSTANTIATE(__nv_fp8_e5m2, uint32_t, ArgMax);

}  // namespace ThorImplementation::SegmentedScanKernels
