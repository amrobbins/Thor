#include "DeepLearning/Api/Data/BatchPolicy.h"

#include <stdexcept>

namespace Thor {

BatchPolicy::BatchPolicy(uint64_t batchSize, bool randomizeTrain, std::optional<uint64_t> randomSeed)
    : batchSize(batchSize), randomizeTrain(randomizeTrain), randomSeed(randomSeed) {
    if (batchSize == 0) {
        throw std::runtime_error("BatchPolicy batch_size must be >= 1.");
    }
    if (!randomizeTrain && randomSeed.has_value()) {
        throw std::runtime_error("BatchPolicy random_seed requires randomize_train=true.");
    }
}

}  // namespace Thor
