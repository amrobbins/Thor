#pragma once

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Thor {

class Network;

class RaggedNetworkInput {
   public:
    class Builder;
};

class RaggedNetworkInput::Builder {
   public:
    virtual RaggedTensor build();

    virtual RaggedNetworkInput::Builder& network(Network& network);
    virtual RaggedNetworkInput::Builder& name(const std::string& name);
    virtual RaggedNetworkInput::Builder& valuesDataType(DataType dataType);
    virtual RaggedNetworkInput::Builder& offsetsDataType(DataType dataType);
    virtual RaggedNetworkInput::Builder& trailingDimensions(const std::vector<uint64_t>& dimensions);
    virtual RaggedNetworkInput::Builder& maxTotalValues(uint64_t maxTotalValues);
    virtual RaggedNetworkInput::Builder& batchSize(uint64_t batchSize);

   private:
    std::optional<Network*> network_;
    std::optional<std::string> name_;
    std::optional<DataType> valuesDataType_;
    DataType offsetsDataType_ = DataType::UINT32;
    std::vector<uint64_t> trailingDimensions_;
    std::optional<uint64_t> maxTotalValues_;
    std::optional<uint64_t> batchSize_;
};

}  // namespace Thor
