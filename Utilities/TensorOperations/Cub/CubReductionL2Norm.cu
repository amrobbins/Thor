#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

// L2 shares the additive CUDA/CUB kernel instantiations emitted by CubReductionSum.cu and
// CubReductionSumSquares.cu. Keeping this translation unit intentionally template-free prevents optional square-root
// finalization from duplicating the complete reduction geometry/kernel matrix.
