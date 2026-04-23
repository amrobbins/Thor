#pragma once

#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace ThorImplementation {
class CustomLayer : public TrainableLayer {
   public:
    virtual ~CustomLayer() = default;

    // Backward-compatible single-input single-output form.
    CustomLayer(DynamicExpression expr,
                const TensorPlacement& placement,
                const std::vector<std::shared_ptr<Parameter>>& parameters,
                bool inferenceOnly,
                int64_t stampedId = -1,
                bool useFastMath = false);

    // Named-port form. Port indices are the graph connection types.
    CustomLayer(DynamicExpression expr,
                std::vector<std::string> inputNames,
                std::vector<std::string> outputNames,
                const TensorPlacement& placement,
                const std::vector<std::shared_ptr<Parameter>>& parameters,
                bool inferenceOnly,
                int64_t stampedId = -1,
                bool useFastMath = false);

    // TrainableLayer
    void forward(Optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override;
    void backward(Optional<Tensor> errorInput, uint32_t batchSize = 0) override;

    // Compute feature output on the data stream
    void computeFeatureOut(uint32_t connectionNumber) override;

    // Gradient-update stream synchronization is handled by backward().
    void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) override;

    // Error-output backward work runs on the data stream.
    Optional<Event> computeErrorOut(uint32_t connectionNumber) override;

    Optional<Tensor> createFeatureOutputTensor() override;
    Optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) override;

    void connectToNextLayer(Layer* nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override;
    Optional<Tensor> connectToPreviousLayer(
        Layer* previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override;
    void replaceErrorInput(Optional<Tensor> oldErrorInput, Optional<Tensor> newErrorInput) override;
    void initialize() override;

    uint64_t flopCountForward() override;
    uint64_t flopCountBackward() override;

    void setLayerName(const std::string& name) { customLayerName = name; }
    std::string getLayerType() override { return "CustomLayer<" + customLayerName + ">"; }

    bool isBackPropStub() override;

   protected:
    void compileImpl() override;
    Parameter::StorageContext buildParameterStorageContext() const override;

   private:
    static void validatePortNames(const std::vector<std::string>& names, const std::string& what);
    uint32_t primaryInputPort() const;
    Stream& computeStream();
    const Stream& computeStream() const;
    void ensurePortStorageAllocated();
    void clearForwardArrivalBookkeeping();
    void clearBackwardArrivalBookkeeping();

    PreparedDynamicExpression::TensorMap buildForwardInputs();
    PreparedDynamicExpression::TensorMap buildForwardOutputs() const;
    PreparedDynamicExpression::TensorMap buildBackwardAdditionalInputs() const;
    PreparedDynamicExpression::TensorMap buildBackwardInputGradOutputs() const;

    Optional<Tensor> inferFeatureOutputTensor(uint32_t outputPortIndex);
    void validatePreparedExpressionInputs(const PreparedDynamicExpression& prepared);
    void validateStampedOutputNames(const StampedExecutionPlan& stamped, const std::vector<std::string>& expectedNames, const char* phase);
    void synchronizeComputeStreamForForwardInputs();

    std::string errorInputNameForOutput(uint32_t outputPortIndex) const;
    std::string errorOutputNameForInput(uint32_t inputPortIndex) const;

   private:
    DynamicExpression layerDefinitionExpression;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

    bool useFastMath = false;

    std::string customLayerName = "UnnamedType";

    std::unordered_map<std::string, uint32_t> inputNameToPort;
    std::unordered_map<std::string, uint32_t> outputNameToPort;

    std::shared_ptr<PreparedDynamicExpression> forwardPrepared;
    std::shared_ptr<StampedExecutionPlan> forwardStamped;
    std::shared_ptr<StampedExecutionPlan> backwardErrorStamped;
    std::shared_ptr<StampedExecutionPlan> backwardWeightsClearStamped;
    std::shared_ptr<StampedExecutionPlan> backwardWeightsAccumulateStamped;

    std::unordered_map<std::string, Tensor> forwardInputsByName;
    std::unordered_map<std::string, Tensor> forwardOutputsByName;
    std::unordered_map<std::string, Tensor> backwardAdditionalInputsByName;
    std::unordered_map<std::string, Tensor> backwardInputGradOutputsByName;

    std::set<unsigned long> allForwardInputTensorIds;
    std::set<unsigned long> stillWaitingForForwardInputTensorIds;
    std::set<unsigned long> allBackwardErrorInputTensorIds;
    std::set<unsigned long> stillWaitingForBackwardErrorInputTensorIds;

    std::vector<Optional<Tensor>> featureInputsConnectedForPorts;
    std::vector<Optional<Tensor>> featureOutputsConnectedForPorts;
    std::vector<Optional<Tensor>> errorInputsConnectedForPorts;
    std::vector<Optional<Tensor>> errorOutputsConnectedForPorts;
};

}  // namespace ThorImplementation
