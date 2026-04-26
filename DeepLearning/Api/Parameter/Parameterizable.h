#pragma once

#include "DeepLearning/Api/Parameter/BoundParameter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Thor {

class Parameter;
class PlacedNetwork;

class Parameterizable {
   public:
    virtual ~Parameterizable() = default;

    Parameterizable() = default;

    virtual uint64_t getParameterizableId() const = 0;
    virtual const std::vector<std::shared_ptr<Parameter>>& getParameters() const = 0;

    std::shared_ptr<Parameter> getParameter(const std::string& name) const;
    BoundParameter getBoundParameter(PlacedNetwork& placedNetwork, const std::string& name) const;
    std::vector<BoundParameter> getBoundParameters(PlacedNetwork& placedNetwork) const;
};

}  // namespace Thor
