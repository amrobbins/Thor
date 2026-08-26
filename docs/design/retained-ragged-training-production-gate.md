# Retained ragged training production gate (T10)

T10 is a qualification gate for the retained padded ragged Conv1D training path. It does not introduce another backend or representation. T9A-T9G establish the individual correctness contracts; T10 gathers them into one release-style target and adds the remaining cross-product and performance qualification.

Run the complete gate with:

```bash
cmake --build <build-dir> --target check-retained-ragged-training-production-gate
```

The target requires a CUDA device. It deliberately runs a disabled CUDA preflight so a GPU-less machine cannot pass merely because the ordinary CUDA tests skipped. It also enables the timing qualification through `THOR_T10_RETAINED_RAGGED_TRAINING_GATE=1`; normal unit-test runs skip that timing-sensitive comparison.

## Completion criteria

| Criterion | Gate evidence |
| --- | --- |
| Forward topology correct | T9E realistic three-convolution topology; T10 mixed-dtype/grouped/depthwise two-convolution topology |
| Backward topology correct | T9C retained active-local backward; T9D reduction exit; T9E complete three-convolution dgrad/wgrad spine |
| Poison-tail correctness | T9A dY sanitation, T9B X+dY sanitation, T9G NaN/+Inf/-Inf shared-producer adversarial test |
| Fanout correctness | T9G compatible forward + incompatible unpack + independently sanitizing dgrad/wgrad consumers |
| No runtime plan/kernel growth | T9F 2,048 width-transition executions with fixed plan/kernel families, workspace identities, preparation counters and cleared selection cache |
| All-empty correctness | T9A exact width-zero dX behavior; T9B exact-zero dW; T9D exact-zero parameter reductions |
| FP16/BF16/FP32 | T10 multilayer forward+dgrad+wgrad numerical qualification |
| Grouped/depthwise | T9A/T9B direct grouped/depthwise references; T10 BF16 grouped and FP32 depthwise multilayer qualification |
| Multilayer training equivalence | T9E three-convolution full forward/backward reference; T10 mixed-dtype two-convolution `dX/dW1/dW2` references |
| Benchmark does not materially regress | T10 CUDA-event median comparison against an equivalent explicit packed-boundary forward/backward baseline |
| No im2col/unfold-like temporary | `explicit_unfold_workspace_bytes == 0` in T9A/T9B/T9E/T9F/T10 retained convolution diagnostics |
| No unexpected representation boundaries | T9C/T9D/T9E topology/ancestry assertions plus the T10 reduction-free single-public-unpack assertion |

## Performance threshold

The T10 performance comparison builds the same `conv -> ReLU -> conv` training mathematics in two forms:

1. one retained forward equation plus one retained backward equation; and
2. two forward equations and two backward equations separated by explicit packed ragged values.

The second form intentionally reproduces the materialized packed boundary without adding extra mathematical operations. Timing uses CUDA events, warmup iterations, alternating measurement order, and the median of seven samples. The retained path must be no slower than 1.15x the packed-boundary median plus 0.02 ms absolute slack. The slack prevents very small kernels from turning timer noise into a false release failure while still detecting a material regression.

The timing comparison is a regression gate, not the safety argument for retained representation. Correctness and ownership are established structurally by T9A-T9G and the T10 functional tests.
