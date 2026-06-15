#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/LossShaper.h"
#include "DeepLearning/Implementation/Layers/Utility/Reshape.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"
#include <optional>
#include <limits>

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

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    static uint64_t flattenedNonBatchDim(const std::vector<uint64_t>& implementationInputLossDimensions) {
        THOR_THROW_IF_FALSE(implementationInputLossDimensions.size() >= 2);
        uint64_t result = 1;
        for (uint32_t i = 1; i < implementationInputLossDimensions.size(); ++i) {
            THOR_THROW_IF_FALSE(implementationInputLossDimensions[i] > 0);
            result *= implementationInputLossDimensions[i];
        }
        return result;
    }

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
            std::vector<uint64_t> implementationDimensions;
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
        std::vector<uint64_t> implementationInputLossDimensions = createRepresentativeImplementationDimensions(lossInput.getDimensions());
        std::vector<uint64_t> implementationOutputLossDimensions =
            getImplementationOutputDimensions(implementationInputLossDimensions, outputLossType);
        bool reduceBatchDim = implementationInputLossDimensions[0] != 1 && implementationOutputLossDimensions[0] == 1;
        const uint64_t inputLossDim = flattenedNonBatchDim(implementationInputLossDimensions);
        const uint64_t outputLossDim = implementationOutputLossDimensions.size() > 1 ? implementationOutputLossDimensions[1] : 1;
        bool reduceLossDim = inputLossDim != 1 && outputLossDim == 1;

        if (implementationInputLossDimensions == implementationOutputLossDimensions)
            return 0;

        THOR_THROW_IF_FALSE(implementationInputLossDimensions[0] <= std::numeric_limits<uint32_t>::max());
        THOR_THROW_IF_FALSE(inputLossDim <= std::numeric_limits<uint32_t>::max());
        const uint32_t representativeBatch = static_cast<uint32_t>(implementationInputLossDimensions[0]);
        const uint32_t flattenedLossDim32 = static_cast<uint32_t>(inputLossDim);

        int deviceNum = tensorPlacement.getDeviceNum();
        uint64_t workspaceSizeInBytes;
        workspaceSizeInBytes = ThorImplementation::BatchReduce(representativeBatch,
                                                               representativeBatch,
                                                               flattenedLossDim32,
                                                               reduceBatchDim,
                                                               reduceLossDim,
                                                               lossInput.getDataType(),
                                                               lossInput.getDataType(),
                                                               // I just need any stream, so get one that is exists anyway.
                                                               Stream::getNextUploadStream(deviceNum))
                                   .getWorkspaceSizeInBytes();

        return lossOutput.getTotalSizeInBytes() + workspaceSizeInBytes;
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

    static std::vector<uint64_t> getApiOutputDimensions(std::vector<uint64_t> apiInputLossDimensions,
                                                        ThorImplementation::LossShaper::OutputLossType outputLossType) {
        // The API layer does not have a batch dimension, so a stand in batch dimension is added
        // and then the implementation layer is asked what the resulting dimensions will be
        // The batch dimension is popped off of that response and returned as the api layer dimensions
        std::vector<uint64_t> implementationInputLossDimensions = createRepresentativeImplementationDimensions(apiInputLossDimensions);
        std::vector<uint64_t> implementationOutputLossDimensions =
            getImplementationOutputDimensions(implementationInputLossDimensions, outputLossType);

        THOR_THROW_IF_FALSE(implementationOutputLossDimensions.size() == 2);
        std::vector<uint64_t> apiOutputLossDimensions(1, implementationOutputLossDimensions[1]);
        return apiOutputLossDimensions;
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
               _outputLossType.value() == ThorImplementation::LossShaper::OutputLossType::CLASSWISE ||
               _outputLossType.value() == ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE);

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

    virtual LossShaper::Builder &reportsClasswiseLoss() {
        THOR_THROW_IF_FALSE(!_outputLossType.has_value());
        _outputLossType = ThorImplementation::LossShaper::OutputLossType::CLASSWISE;
        return *this;
    }

    virtual LossShaper::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!_outputLossType.has_value());
        _outputLossType = ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _lossInput;
    std::optional<ThorImplementation::LossShaper::OutputLossType> _outputLossType;
};

}  // namespace Thor
