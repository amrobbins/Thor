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
    uintptr_t execution_state_id = 0;
    uintptr_t workspace_state_id = 0;
};

struct StampedMatmulStageDiagnostic {
    uint32_t stage_index = 0;
    uint32_t lane_index = 0;
    uint32_t dependency_count = 0;
    StampedMatmulKernelDiagnostic kernel;
};

// Read-only runtime-extent diagnostics for physical consumers that deliberately
// execute beyond a ragged tensor's logical active prefix. These diagnostics do
// not synchronize or launch work; they describe the exact pre-read sanitation
// implied by the current host-published row-partition extent and stamped bucket policy.
enum class PackedRowConsumerKind : uint8_t {
    Matmul = 0,
    RmsNorm = 1,
};

struct PackedRowConsumerDiagnostic {
    uint32_t stage_index = 0;
    PackedRowConsumerKind kind = PackedRowConsumerKind::Matmul;
    uint64_t active_rows = 0;
    uint64_t selected_rows = 0;
    uint64_t full_capacity_rows = 0;
    uint64_t sanitized_rows = 0;
    uint32_t sanitized_operand_count = 0;
    // Bytes the physical consumer will actually sanitize before its next read.
    uint64_t sanitized_bytes = 0;
    // Comparison baseline: bytes that would be written if the same row-bound
    // operands were canonicalized through full packed capacity instead.
    uint64_t full_tail_bytes = 0;
};

}  // namespace ThorImplementation
