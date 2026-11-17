#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"

namespace ThorImplementation::ReductionInternal {

DenseReductionSelectionCache& denseReductionSelectionCache() {
    static DenseReductionSelectionCache repository;
    return repository;
}

}  // namespace ThorImplementation::ReductionInternal
