#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <optional>
#include <string>

namespace Thor {

class Network;

// Exposes a ragged tensor without extending its logical extent. The returned
// row partition is authoritative and values beyond offsets[B] remain undefined
// capacity; external consumers inherit the same consumer-responsibility rule as
// internal consumers.
class RaggedNetworkOutput {
   public:
    class Builder;

    RaggedNetworkOutput() = default;

    const std::string& getName() const { return name_; }
    RaggedTensor getInput() const { return input_; }
    RaggedTensor getFeatureOutput() const { return output_; }

   private:
    std::string name_;
    RaggedTensor input_;
    RaggedTensor output_;

    friend class Builder;
};

class RaggedNetworkOutput::Builder {
   public:
    RaggedNetworkOutput build();

    Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!network_.has_value());
        network_ = &network;
        return *this;
    }

    Builder& name(const std::string& name) {
        THOR_THROW_IF_FALSE(!name.empty());
        THOR_THROW_IF_FALSE(!name_.has_value());
        name_ = name;
        return *this;
    }

    Builder& inputTensor(const RaggedTensor& input) {
        THOR_THROW_IF_FALSE(input.isInitialized());
        THOR_THROW_IF_FALSE(!input_.has_value());
        input_ = input;
        return *this;
    }

   private:
    std::optional<Network*> network_;
    std::optional<std::string> name_;
    std::optional<RaggedTensor> input_;
};

}  // namespace Thor
