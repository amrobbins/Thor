#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <vector>

namespace ThorImplementation::Ensemble {

void launchWeightedMean(Tensor destination,
                        const std::vector<Tensor>& sources,
                        const std::vector<double>& weights,
                        Stream stream);

}  // namespace ThorImplementation::Ensemble
