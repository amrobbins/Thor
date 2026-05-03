#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "Utilities/Common/Stream.h"
#include "Utilities/TarFile/TarWriter.h"

namespace ThorImplementation {
class StampedNetwork;
}

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

    static nlohmann::json serialize(nlohmann::json parameterJson,
                                    std::shared_ptr<ParameterSpecification> parameterSpecification,
                                    thor_file::TarWriter& archiveWriter,
                                    Stream stream,
                                    bool saveOptimizerState,
                                    ThorImplementation::StampedNetwork& stampedNetwork,
                                    const std::string& filenamePrefix,
                                    const uint64_t apiLayerId);

   private:
    std::shared_ptr<ParameterSpecification> parameter;
    PlacedNetwork* placedNetwork = nullptr;
    uint64_t apiLayerId = 0;
};

}  // namespace Thor
