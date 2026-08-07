#pragma once

#include <cstdint>

namespace ThorImplementation {

// Read-only stamp-time diagnostics for a cuBLASLt-backed Expression Matmul stage.
// These values describe the selected kernel artifact; collecting them does not
// benchmark or synchronize the runtime execution path.
struct StampedMatmulKernelDiagnostic {
    int32_t m = 0;
    int32_t n = 0;
    int32_t k = 0;
    int32_t batch_count = 1;
    uint64_t flop_count = 0;
    bool has_measured_kernel = false;
    float waves_count = 0.0f;
    double picker_runtime_ms = 0.0;
    uint64_t workspace_bytes = 0;
    int algorithm_id = -1;
};

struct StampedMatmulStageDiagnostic {
    uint32_t stage_index = 0;
    uint32_t lane_index = 0;
    uint32_t dependency_count = 0;
    StampedMatmulKernelDiagnostic kernel;
};

}  // namespace ThorImplementation
