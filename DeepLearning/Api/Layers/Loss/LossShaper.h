#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/LossShaper.h"
#include "DeepLearning/Implementation/Layers/Utility/Reshape.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include <optional>

namespace Thor {

class LossShaper : public Layer {
   public:
    class Builder;
    LossShaper() {}

    ~LossShaper() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<LossShaper>(*this); }

    std::string getLayerType() const override { return "LossShaper"; }

    virtual Tensor getLossInput() const { return lossInput; }
    virtual Tensor getLossOutput() const { return lossOutput; }

    // getLossInput() and getLossOutput() are synonyms for getFeatureInput().value() and getFeatureOutput().value() in losses:
    std::optional<Tensor> getFeatureInput() const override { return getLossInput(); }
    std::optional<Tensor> getFeatureOutput() const override { return getLossOutput(); }

    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override {
        if (lossInput.isInitialized() && inputTensor == lossInput) {
            return "raw_loss";
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::string> getOutputPortName(const Tensor& outputTensor) const override {
        if (!lossOutput.isInitialized() || outputTensor != lossOutput) {
            return std::nullopt;
        }
        switch (outputLossType) {
            case ThorImplementation::LossShaper::OutputLossType::BATCH:
                return "batch_loss";
            case ThorImplementation::LossShaper::OutputLossType::PER_EXAMPLE:
                return "per_example_loss";
            case ThorImplementation::LossShaper::OutputLossType::PER_OUTPUT:
                return "per_output_loss";
        }
        THOR_UNREACHABLE();
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == lossInput || connectingApiTensor == lossOutput);

        std::vector<uint64_t> implementationInputLossDimensions = createRepresentativeImplementationDimensions(lossInput.getDimensions());
        std::vector<uint64_t> implementationOutputLossDimensions =
            getImplementationOutputDimensions(implementationInputLossDimensions, outputLossType);

        if (implementationInputLossDimensions == implementationOutputLossDimensions) {
            // In this case we need a nop, so just place a reshape with the same shape, this carries no compute cost or memory cost.
            // Tell reshape to match the batch size:
            implementationOutputLossDimensions[0] = 0;
            std::shared_ptr<ThorImplementation::Reshape> nopReshape =
                std::make_shared<ThorImplementation::Reshape>(implementationOutputLossDimensions);
            return nopReshape;
        } else {
            std::shared_ptr<ThorImplementation::LossShaper> lossShaper = std::make_shared<ThorImplementation::LossShaper>(outputLossType);
            return lossShaper;
        }
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        THOR_THROW_IF_FALSE(batchSize > 0);
        std::vector<uint64_t> implementationInputLossDimensions = createRepresentativeImplementationDimensions(lossInput.getDimensions());
        implementationInputLossDimensions[0] = batchSize;
        std::vector<uint64_t> implementationOutputLossDimensions =
            getImplementationOutputDimensions(implementationInputLossDimensions, outputLossType);

        if (implementationInputLossDimensions == implementationOutputLossDimensions)
            return 0;

        const std::vector<uint32_t> axes =
            ThorImplementation::LossShaper::getReductionAxes(implementationInputLossDimensions, outputLossType);
        const ThorImplementation::TensorDescriptor inputDescriptor(lossInput.getDataType(), implementationInputLossDimensions);
        const ThorImplementation::TensorDescriptor outputDescriptor(lossOutput.getDataType(), implementationOutputLossDimensions);
        ThorImplementation::CubReduction reduction(
            ThorImplementation::CubReductionOp::Sum, axes, lossOutput.getDataType(), 1.0f);
        Stream queryStream = Stream::getNextUploadStream(tensorPlacement.getDeviceNum());
        return outputDescriptor.getArraySizeInBytes() + reduction.queryWorkspaceSizeInBytes(inputDescriptor, queryStream);
    }

    static std::vector<uint64_t> getImplementationOutputDimensions(std::vector<uint64_t> implementationInputLossDimensions,
                                                                   ThorImplementation::LossShaper::OutputLossType outputLossType) {
        std::vector<uint64_t> implementationOutputLossDimensions =
            ThorImplementation::LossShaper::getOutputDimensions(implementationInputLossDimensions, outputLossType);
        return implementationOutputLossDimensions;
    }

    static std::vector<uint64_t> createRepresentativeImplementationDimensions(std::vector<uint64_t> apiInputLossDimensions) {
        // The API layer does not have a batch dimension, so a stand in batch dimension is added
        uint64_t arbitraryNonSingularBatchDimension = 10;
        std::vector<uint64_t> implementationInputLossDimensions;
        implementationInputLossDimensions.push_back(arbitraryNonSingularBatchDimension);
        for (uint32_t i = 0; i < apiInputLossDimensions.size(); ++i)
            implementationInputLossDimensions.push_back(apiInputLossDimensions[i]);
        return implementationInputLossDimensions;
    }

    // PER_OUTPUT reduces only the implementation batch axis and preserves every API-visible
    // loss dimension. BATCH and PER_EXAMPLE each report a scalar API feature shape [1].
    static std::vector<uint64_t> getApiOutputDimensions(std::vector<uint64_t> apiInputLossDimensions,
                                                        ThorImplementation::LossShaper::OutputLossType outputLossType) {
        // The API layer does not have a batch dimension, so a stand in batch dimension is added
        // and then the implementation layer is asked what the resulting dimensions will be
        // The batch dimension is popped off of that response and returned as the api layer dimensions
        std::vector<uint64_t> implementationInputLossDimensions = createRepresentativeImplementationDimensions(apiInputLossDimensions);
        std::vector<uint64_t> implementationOutputLossDimensions =
            getImplementationOutputDimensions(implementationInputLossDimensions, outputLossType);

        THOR_THROW_IF_FALSE(!implementationOutputLossDimensions.empty());
        implementationOutputLossDimensions.erase(implementationOutputLossDimensions.begin());
        if (implementationOutputLossDimensions.empty())
            implementationOutputLossDimensions.push_back(1);
        return implementationOutputLossDimensions;
    }

    Tensor lossInput;
    Tensor lossOutput;
    ThorImplementation::LossShaper::OutputLossType outputLossType;
};

class LossShaper::Builder {
   public:
    virtual LossShaper construct() const {
        THOR_THROW_IF_FALSE(_lossInput.has_value());
        THOR_THROW_IF_FALSE(!_lossInput.value().getDimensions().empty());
        THOR_THROW_IF_FALSE(_outputLossType.has_value());
        THOR_THROW_IF_FALSE(_outputLossType.value() == ThorImplementation::LossShaper::OutputLossType::BATCH ||
               _outputLossType.value() == ThorImplementation::LossShaper::OutputLossType::PER_OUTPUT ||
               _outputLossType.value() == ThorImplementation::LossShaper::OutputLossType::PER_EXAMPLE);

        LossShaper lossShaper;
        lossShaper.lossInput = _lossInput.value();
        lossShaper.outputLossType = _outputLossType.value();

        std::vector<uint64_t> apiOutputLossDimensions = getApiOutputDimensions(_lossInput.value().getDimensions(), _outputLossType.value());
        lossShaper.lossOutput = Tensor(_lossInput.value().getDataType(), apiOutputLossDimensions);

        lossShaper.initialized = true;
        return lossShaper;
    }

    virtual LossShaper build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        LossShaper lossShaper;
        lossShaper = construct();
        lossShaper.addToNetwork(_network.value());
        return lossShaper;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        return construct().getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }

    virtual LossShaper::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual LossShaper::Builder &lossInput(Tensor _lossInput) {
        THOR_THROW_IF_FALSE(!this->_lossInput.has_value());
        this->_lossInput = _lossInput;
        // Remember that API layer does not have the batch dimension
        // Batch size is set when stamping a network input
        if (_lossInput.getDimensions().size() == 0)
            this->_lossInput.value().reshape({1});
        return *this;
    }

    virtual LossShaper::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!_outputLossType.has_value());
        _outputLossType = ThorImplementation::LossShaper::OutputLossType::BATCH;
        return *this;
    }

    virtual LossShaper::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!_outputLossType.has_value());
        _outputLossType = ThorImplementation::LossShaper::OutputLossType::PER_OUTPUT;
        return *this;
    }

    virtual LossShaper::Builder &reportsPerExampleLoss() {
        THOR_THROW_IF_FALSE(!_outputLossType.has_value());
        _outputLossType = ThorImplementation::LossShaper::OutputLossType::PER_EXAMPLE;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _lossInput;
    std::optional<ThorImplementation::LossShaper::OutputLossType> _outputLossType;
};

}  // namespace Thor
