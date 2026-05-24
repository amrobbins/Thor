#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

class Embedding : public TrainableLayer {
   public:
    class Builder;

    Embedding() = default;
    ~Embedding() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Embedding>(*this); }

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const nlohmann::json& j, Network* network);
    nlohmann::json architectureJson() const override;

    uint64_t getVocabularySize() const { return vocabularySize; }
    uint64_t getEmbeddingDim() const { return embeddingDim; }
    Tensor::DataType getWeightsDataType() const { return weightsDataType; }
    std::optional<uint64_t> getPaddingIndex() const { return paddingIndex; }
    bool usesSparseGradients() const { return sparseGradients; }

    int getConnectionType(Tensor connectingTensor) const override {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (connectingTensor == featureInputs[i])
                return static_cast<int>(i);
        }
        for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
            if (connectingTensor == featureOutputs[i])
                return static_cast<int>(i);
        }
        throw std::runtime_error("Tensor is not connected to this Embedding layer.");
    }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    std::string getLayerType() const override { return "Embedding"; }

   private:
    static bool isSupportedIndexDataType(Tensor::DataType dataType);
    static bool isSupportedWeightsDataType(Tensor::DataType dataType);
    static std::string dataTypeName(Tensor::DataType dataType);
    static void validateIndexTensor(const Tensor& tensor, const std::string& what);

    uint64_t vocabularySize = 0;
    uint64_t embeddingDim = 0;
    Tensor::DataType weightsDataType = Tensor::DataType::FP32;
    std::optional<uint64_t> paddingIndex = std::nullopt;
    bool sparseGradients = true;

    friend class Network;
    friend class Builder;
};

class Embedding::Builder {
   public:
    virtual ~Builder() = default;

    virtual Embedding build();

    virtual Embedding::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual Embedding::Builder& featureInput(Tensor featureInput) {
        THOR_THROW_IF_FALSE(featureInput.isInitialized());
        validateIndexTensor(featureInput, "featureInput");
        this->_featureInputs.push_back(featureInput);
        if (_featureInputs.size() > 1) {
            THOR_THROW_IF_FALSE(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            THOR_THROW_IF_FALSE(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual Embedding::Builder& vocabularySize(uint64_t vocabularySize) {
        THOR_THROW_IF_FALSE(!this->_vocabularySize.has_value());
        this->_vocabularySize = vocabularySize;
        return *this;
    }

    virtual Embedding::Builder& embeddingDim(uint64_t embeddingDim) {
        THOR_THROW_IF_FALSE(!this->_embeddingDim.has_value());
        this->_embeddingDim = embeddingDim;
        return *this;
    }

    virtual Embedding::Builder& weightsDataType(Tensor::DataType weightsDataType) {
        THOR_THROW_IF_FALSE(!this->_weightsDataType.has_value());
        this->_weightsDataType = weightsDataType;
        return *this;
    }

    virtual Embedding::Builder& paddingIndex(uint64_t paddingIndex) {
        THOR_THROW_IF_FALSE(!this->_paddingIndex.has_value());
        this->_paddingIndex = paddingIndex;
        return *this;
    }

    virtual Embedding::Builder& sparseGradients(bool sparseGradients) {
        THOR_THROW_IF_FALSE(!this->_sparseGradients.has_value());
        this->_sparseGradients = sparseGradients;
        return *this;
    }

    virtual Embedding::Builder& weightsInitializer(std::shared_ptr<Initializer> weightsInitializer) {
        THOR_THROW_IF_FALSE(this->_weightsInitializer == nullptr);
        if (weightsInitializer != nullptr)
            this->_weightsInitializer = weightsInitializer->clone();
        return *this;
    }

    virtual Embedding::Builder& weightsOptimizer(std::shared_ptr<Optimizer> weightsOptimizer) {
        THOR_THROW_IF_FALSE(this->_weightsOptimizer == nullptr);
        this->_weightsOptimizer = weightsOptimizer;
        return *this;
    }

   private:
    void verifyConfig() const;

    std::optional<Network*> _network = std::nullopt;
    std::vector<Tensor> _featureInputs;
    std::optional<uint64_t> _vocabularySize = std::nullopt;
    std::optional<uint64_t> _embeddingDim = std::nullopt;
    std::optional<Tensor::DataType> _weightsDataType = std::nullopt;
    std::optional<uint64_t> _paddingIndex = std::nullopt;
    std::optional<bool> _sparseGradients = std::nullopt;
    std::shared_ptr<Initializer> _weightsInitializer = nullptr;
    std::shared_ptr<Optimizer> _weightsOptimizer = nullptr;
};

}  // namespace Thor
