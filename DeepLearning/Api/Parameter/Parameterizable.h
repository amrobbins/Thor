#pragma once

#include "DeepLearning/Api/Parameter/BoundParameter.h"

#include <cstdint>
#include <debug/unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace Thor {

class ParameterSpecification;
class PlacedNetwork;

class Parameterizable {
   public:
    virtual ~Parameterizable() = default;

    Parameterizable(const uint64_t owningLayerId) : owningLayerId(owningLayerId) {};

    void addParameter(const std::shared_ptr<ParameterSpecification>& parameter);
    std::vector<std::string> listParameters() const;
    virtual bool hasParameter(const std::string& name) const;
    uint64_t getOwningLayerId() const { return owningLayerId; }

    std::shared_ptr<ParameterSpecification> getParameterSpecification(const std::string& name) const;

    BoundParameter getBoundParameter(PlacedNetwork& placedNetwork, const std::string& name) const;
    std::vector<BoundParameter> getBoundParameter(PlacedNetwork& placedNetwork, const std::string name) const;

    std::string listParametersString() const;

   protected:
    std::unordered_map<std::string, std::shared_ptr<ParameterSpecification>> parameters{};

   private:
    const uint64_t owningLayerId;
};

}  // namespace Thor
