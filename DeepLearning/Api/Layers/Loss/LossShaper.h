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

    virtual shared_ptr<Layer> clone() const { return make_shared<LossShaper>(*this); }

    virtual string getLayerType() const { return "LossShaper"; }

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
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == lossInput || connectingApiTensor == lossOutput);

        vector<uint64_t> inputDimensions = lossInput.getDimensions();
        vector<uint64_t> outputDimensions = ThorImplementation::LossShaper::getOutputDimensions(lossInput.getDimensions(), outputLossType);

        if (inputDimensions == outputDimensions) {
            // In this case we need a nop, so just place a reshape with the same shape.
            vector<uint64_t> implementationDimensions;
            // Tell reshape to match the batch size:
            implementationDimensions.push_back(0);
            for (uint32_t i = 0; i < outputDimensions.size(); ++i)
                implementationDimensions.push_back(outputDimensions[i]);
            ThorImplementation::Reshape *nopReshape = new ThorImplementation::Reshape(outputDimensions);
            Thor::Layer::connectTwoLayers(drivingLayer, nopReshape, drivingApiLayer, this, connectingApiTensor);
            return nopReshape;
        } else {
            ThorImplementation::LossShaper *lossShaper = new ThorImplementation::LossShaper(outputLossType);
            Thor::Layer::connectTwoLayers(drivingLayer, lossShaper, drivingApiLayer, this, connectingApiTensor);
            return lossShaper;
        }
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        vector<uint64_t> inputDimensions = lossInput.getDimensions();
        vector<uint64_t> outputDimensions = lossOutput.getDimensions();
        bool reduceBatchDim = inputDimensions[0] != 1 && outputDimensions[0] == 1;
        bool reduceLossDim = inputDimensions[1] != 1 && outputDimensions[1] == 1;

        if (inputDimensions == outputDimensions)
            return 0;

        // Only compute this once per device to avoid creating and destroying many streams
        static std::map<int, uint64_t> workspaceSizePerDevice;
        int deviceNum = tensorPlacement.getDeviceNum();
        uint64_t workspaceSizeInBytes;
        if (workspaceSizePerDevice.count(deviceNum) == 1) {
            workspaceSizeInBytes = workspaceSizePerDevice[deviceNum];
        } else {
            workspaceSizeInBytes = ThorImplementation::BatchReduce(lossInput.getDimensions()[0],
                                                                   lossInput.getDimensions()[0],
                                                                   lossInput.getDimensions()[1],
                                                                   reduceBatchDim,
                                                                   reduceLossDim,
                                                                   Tensor::convertToImplementationDataType(lossInput.getDataType()),
                                                                   Tensor::convertToImplementationDataType(lossInput.getDataType()),
                                                                   Stream(deviceNum))
                                       .getWorkspaceSizeInBytes();
            workspaceSizePerDevice[deviceNum] = workspaceSizeInBytes;
        }

        return lossOutput.getTotalSizeInBytes() + workspaceSizeInBytes;
    }

    Tensor lossInput;
    Tensor lossOutput;
    ThorImplementation::LossShaper::OutputLossType outputLossType;
};

class LossShaper::Builder {
   public:
    virtual LossShaper construct() const {
        assert(_lossInput.isPresent());
        assert(_outputLossType.isPresent());
        assert(_outputLossType.get() == ThorImplementation::LossShaper::OutputLossType::BATCH ||
               _outputLossType.get() == ThorImplementation::LossShaper::OutputLossType::CLASSWISE ||
               _outputLossType.get() == ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE);

        LossShaper lossShaper;
        lossShaper.lossInput = _lossInput.get();
        lossShaper.outputLossType = _outputLossType.get();

        vector<uint64_t> outputDimensions =
            ThorImplementation::LossShaper::getOutputDimensions(_lossInput.get().getDimensions(), _outputLossType.get());
        lossShaper.lossOutput = Tensor(_lossInput.get().getDataType(), outputDimensions);

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

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        return construct().getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }

    virtual LossShaper::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual LossShaper::Builder &lossInput(Tensor _lossInput) {
        assert(!this->_lossInput.isPresent());
        assert(!_lossInput.getDimensions().empty());
        this->_lossInput = _lossInput;
        if (_lossInput.getDimensions().size() == 1)
            this->_lossInput.get().reshape({_lossInput.getDimensions()[0], 1});
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
