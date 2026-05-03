#pragma once

#include "DeepLearning/Api/Parameter/BoundParameter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Thor {

class ParameterSpecification;
class PlacedNetwork;

class Parameterizable {
   public:
    virtual ~Parameterizable() = default;

    Parameterizable() : owningLayerId(0) {}
    explicit Parameterizable(const uint64_t owningLayerId) : owningLayerId(owningLayerId) {}

    void addParameter(const std::shared_ptr<ParameterSpecification>& parameter);
    std::vector<std::string> listParameters() const;
    virtual bool hasParameter(const std::string& name) const;
    virtual uint64_t getParameterizableId() const { return owningLayerId; }
    uint64_t getOwningLayerId() const { return getParameterizableId(); }

    std::shared_ptr<ParameterSpecification> getParameterSpecification(const std::string& name) const;
    virtual std::vector<std::shared_ptr<ParameterSpecification>> getParameters() const;

    BoundParameter getBoundParameter(PlacedNetwork& placedNetwork, const std::string& name) const;
    std::vector<BoundParameter> getBoundParameters(PlacedNetwork& placedNetwork) const;

    std::string listParametersString() const;

   protected:
    std::unordered_map<std::string, std::shared_ptr<ParameterSpecification>> parameters{};

   private:
    const uint64_t owningLayerId;
};

}  // namespace Thor
