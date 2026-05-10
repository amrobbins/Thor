# Thor cuDNN Scaled Dot Product Attention

This directory adds a low-level Thor tensor operation for cuDNN Frontend SDPA.  It is intended to become Thor's first-class Attention execution stage rather than an expression-level matmul-softmax-matmul decomposition.

## Current scope

Implemented in `CudnnAttention.h/.cpp`:

- FP16/BF16 SDPA forward and backward through the cuDNN Frontend C++ Graph API.
- FP8 SDPA forward with explicit scale/descale and amax tensors.
- Graph-plan caching keyed by GPU, pass, tensor dimensions/strides/dtypes, mask/dropout/padding/ragged/paged options, and scale policy.
- BHSD and BSHD convenience layout builders while preserving semantic `[B,H,S,D]` indexing.
- MHA, MQA, and GQA validation.
- Causal top-left, causal bottom-right, sliding-window masks, padding masks, ALiBi, additive bias, ragged offsets, paged KV table hooks, and Philox/custom dropout hooks for FP16/BF16.
- Warm-up APIs for stamping/compilation paths that want to build cuDNN plans before first execution.

Reserved but intentionally not enabled yet:

- FP8 backward.  The descriptor and args already carry the required tensors.  Enable it after validating FP8 forward numerics/performance on the target Blackwell/Hopper machine and after deciding whether Thor gradients should remain FP8 or accumulate in a higher-precision side buffer.
- Full API `Attention` / `MultiHeadAttention` layer integration.  The right integration point is a new `CompiledExecutionStage::Kind::Attention` stage so Thor does not lose cuDNN's fused FlashAttention path by lowering to generic expression ops.

## Build requirements

- CUDA toolkit headers/libraries available to Thor.
- cuDNN backend/runtime available to Thor.
- cuDNN Frontend C++ headers available as `<cudnn_frontend.h>`.  NVIDIA recommends cudnn-frontend v1.23.0 for cuDNN 9.21.0 and later.

If the frontend header is absent, the wrapper still compiles but throws an informative runtime error when used.

## Suggested validation order

1. Descriptor-only unit tests for layout/shape/mask validation.
2. Forward inference FP16/BF16 vs CPU/PyTorch reference: no mask, causal top-left, causal bottom-right, sliding window, MQA/GQA.
3. Training FP16/BF16: forward stats + backward gradients vs reference with tolerances by dtype.
4. Variable length: padding-mask path, then ragged THD offsets.
5. Decode/paged KV cache path with a real KV-cache allocator.
6. FP8 forward: explicit E4M3/E5M2 scale/descale, amax propagation, and stress tests over small/large sequence lengths.
7. Enable FP8 backward only after steps 1-6 are stable on the actual CUDA/cuDNN/GPU stack.
8. Wire into expression scheduling as `Attention` stage and benchmark against Thor decomposition, PyTorch SDPA, FlashAttention, and Transformer Engine.
