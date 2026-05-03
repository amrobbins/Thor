#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Utilities/TarFile/TarWriter.h"

namespace ThorImplementation {
class StampedNetwork;
}

namespace Thor {

class ParameterSpecification;
class PlacedNetwork;
class BoundParameter;

class Parameterizable {
   public:
    virtual ~Parameterizable() = default;

    explicit Parameterizable(const uint64_t owningLayerId) : owningLayerId(owningLayerId) {}

    void addParameter(const std::shared_ptr<ParameterSpecification>& parameter);
    std::vector<std::string> listParameters() const;
    virtual bool hasParameter(const std::string& name) const;
    virtual uint64_t getParameterizableId() const { return owningLayerId; }
    uint64_t getOwningLayerId() const { return getParameterizableId(); }

    std::shared_ptr<ParameterSpecification> getParameterSpecification(const std::string& name) const;
    virtual std::vector<std::shared_ptr<ParameterSpecification>> getParameters() const;

    BoundParameter getBoundParameter(PlacedNetwork* placedNetwork, const std::string& name) const;
    std::vector<BoundParameter> getBoundParameters(PlacedNetwork* placedNetwork) const;

    std::string listParametersString() const;

    nlohmann::json getParametersArchitectureJson() const;
    void serializeParameters(nlohmann::json& parameterJson,
                             thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork,
                             const std::string& filenamePrefix) const;

   protected:
    std::vector<std::shared_ptr<ParameterSpecification>> parameters{};
    std::unordered_map<std::string, size_t> parameterIndexByName{};
    // std::unordered_map<std::string, std::shared_ptr<ParameterSpecification>> parameters{};

   private:
    const uint64_t owningLayerId;
};

}  // namespace Thor
