#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace Thor {
class ParameterSpecification;
class PlacedNetwork;

class BoundParameter {
   public:
    BoundParameter(std::shared_ptr<ParameterSpecification> parameter, PlacedNetwork* placedNetwork, uint64_t apiLayerId);

    [[nodiscard]] const std::string& getName() const;
    [[nodiscard]] bool isTrainable() const;
    [[nodiscard]] bool isTrainingEnabled() const;
    void setTrainingEnabled(bool enabled);
    [[nodiscard]] bool hasOptimizer() const;

   private:
    std::shared_ptr<ParameterSpecification> parameter;
    PlacedNetwork* placedNetwork = nullptr;
    uint64_t apiLayerId = 0;
};

}  // namespace Thor
