#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"

namespace ThorImplementation::CubReductionInternal {

DenseReductionSelectionCache& denseReductionSelectionCache() {
    static DenseReductionSelectionCache repository;
    return repository;
}

}  // namespace ThorImplementation::CubReductionInternal
