#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/LossShaper.h"
#include "DeepLearning/Implementation/Layers/Utility/Reshape.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

namespace Thor {

class LossShaper : public Layer {
   public:
    class Builder;
    LossShaper() {}

    virtual ~LossShaper() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<LossShaper>(*this); }

    virtual std::string getLayerType() const { return "LossShaper"; }

    virtual Tensor getLossInput() const { return lossInput; }
    virtual Tensor getLossOutput() const { return lossOutput; }

    // getLossInput() and getLossOutput() are synonyms for getFeatureInput() and getFeatureOutput() in losses:
    virtual Optional<Tensor> getFeatureInput() const { return getLossInput(); }
    virtual Optional<Tensor> getFeatureOutput() const { return getLossOutput(); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             std::vector<std::shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == lossInput || connectingApiTensor == lossOutput);

        std::vector<uint64_t> implementationInputLossDimensions = createRepresentativeImplementationDimensions(lossInput.getDimensions());
        std::vector<uint64_t> implementationOutputLossDimensions =
            getImplementationOutputDimensions(implementationInputLossDimensions, outputLossType);

        if (implementationInputLossDimensions == implementationOutputLossDimensions) {
            // In this case we need a nop, so just place a reshape with the same shape, this carries no compute cost or memory cost.
            std::vector<uint64_t> implementationDimensions;
            // Tell reshape to match the batch size:
            implementationOutputLossDimensions[0] = 0;
            ThorImplementation::Reshape *nopReshape = new ThorImplementation::Reshape(implementationOutputLossDimensions);
            Thor::Layer::connectTwoLayers(drivingLayer, nopReshape, drivingApiLayer, this, connectingApiTensor);
            return nopReshape;
        } else {
            ThorImplementation::LossShaper *lossShaper = new ThorImplementation::LossShaper(outputLossType);
            Thor::Layer::connectTwoLayers(drivingLayer, lossShaper, drivingApiLayer, this, connectingApiTensor);
            return lossShaper;
        }
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        std::vector<uint64_t> implementationInputLossDimensions = createRepresentativeImplementationDimensions(lossInput.getDimensions());
        std::vector<uint64_t> implementationOutputLossDimensions =
            getImplementationOutputDimensions(implementationInputLossDimensions, outputLossType);
        bool reduceBatchDim = implementationInputLossDimensions[0] != 1 && implementationOutputLossDimensions[0] == 1;
        bool reduceLossDim = implementationInputLossDimensions[1] != 1 && implementationOutputLossDimensions[1] == 1;

        if (implementationInputLossDimensions == implementationOutputLossDimensions)
            return 0;

        int deviceNum = tensorPlacement.getDeviceNum();
        uint64_t workspaceSizeInBytes;
        workspaceSizeInBytes = ThorImplementation::BatchReduce(implementationInputLossDimensions[0],
                                                               implementationInputLossDimensions[0],
                                                               implementationInputLossDimensions[1],
                                                               reduceBatchDim,
                                                               reduceLossDim,
                                                               Tensor::convertToImplementationDataType(lossInput.getDataType()),
                                                               Tensor::convertToImplementationDataType(lossInput.getDataType()),
                                                               Stream::getStaticStream(deviceNum))
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

        assert(implementationOutputLossDimensions.size() == 2);
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
        assert(_lossInput.isPresent());
        assert(_lossInput.get().getDimensions().size() == 1);
        assert(_outputLossType.isPresent());
        assert(_outputLossType.get() == ThorImplementation::LossShaper::OutputLossType::BATCH ||
               _outputLossType.get() == ThorImplementation::LossShaper::OutputLossType::CLASSWISE ||
               _outputLossType.get() == ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE);

        LossShaper lossShaper;
        lossShaper.lossInput = _lossInput.get();
        lossShaper.outputLossType = _outputLossType.get();

        std::vector<uint64_t> apiOutputLossDimensions = getApiOutputDimensions(_lossInput.get().getDimensions(), _outputLossType.get());
        lossShaper.lossOutput = Tensor(_lossInput.get().getDataType(), apiOutputLossDimensions);

        lossShaper.initialized = true;
        return lossShaper;
    }

    virtual LossShaper build() {
        assert(_network.isPresent());
        LossShaper lossShaper;
        lossShaper = construct();
        lossShaper.addToNetwork(_network.get());
        return lossShaper;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        return construct().getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }

    virtual LossShaper::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual LossShaper::Builder &lossInput(Tensor _lossInput) {
        assert(!this->_lossInput.isPresent());
        this->_lossInput = _lossInput;
        // Remember that API layer does not have the batch dimension
        // Batch size is set when stamping a network input
        if (_lossInput.getDimensions().size() == 0)
            this->_lossInput.get().reshape({1});
        return *this;
    }

    virtual LossShaper::Builder &reportsBatchLoss() {
        assert(_outputLossType.isEmpty());
        _outputLossType = ThorImplementation::LossShaper::OutputLossType::BATCH;
        return *this;
    }

    virtual LossShaper::Builder &reportsClasswiseLoss() {
        assert(_outputLossType.isEmpty());
        _outputLossType = ThorImplementation::LossShaper::OutputLossType::CLASSWISE;
        return *this;
    }

    virtual LossShaper::Builder &reportsElementwiseLoss() {
        assert(_outputLossType.isEmpty());
        _outputLossType = ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _lossInput;
    Optional<ThorImplementation::LossShaper::OutputLossType> _outputLossType;
};

}  // namespace Thor
