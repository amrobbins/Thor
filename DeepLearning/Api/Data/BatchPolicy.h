#pragma once

#include <cstdint>
#include <optional>

namespace Thor {

/** Immutable batching policy attached to a reusable dataset/split definition. */
class BatchPolicy {
   public:
    BatchPolicy(uint64_t batchSize, bool randomizeTrain = true, std::optional<uint64_t> randomSeed = std::nullopt);

    [[nodiscard]] uint64_t getBatchSize() const { return batchSize; }
    [[nodiscard]] bool getRandomizeTrain() const { return randomizeTrain; }
    [[nodiscard]] std::optional<uint64_t> getRandomSeed() const { return randomSeed; }

    bool operator==(const BatchPolicy &rhs) const = default;
    bool operator!=(const BatchPolicy &rhs) const = default;

   private:
    uint64_t batchSize;
    bool randomizeTrain;
    std::optional<uint64_t> randomSeed;
};

}  // namespace Thor
