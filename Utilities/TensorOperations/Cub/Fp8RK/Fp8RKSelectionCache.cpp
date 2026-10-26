#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectionCache.h"

namespace ThorImplementation::CubReductionInternal::Fp8RK {

Fp8RKSelectionCache& fp8RKSelectionCache() {
    static Fp8RKSelectionCache repository;
    return repository;
}

} // namespace ThorImplementation::CubReductionInternal::Fp8RK
