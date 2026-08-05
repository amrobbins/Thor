#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

// Mean shares the additive CUB kernel instantiations emitted by CubReductionSum.cu. Keeping this translation unit
// intentionally template-free prevents Sum x Mean from duplicating the complete reduction geometry/kernel matrix.
