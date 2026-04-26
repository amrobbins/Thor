#pragma once

#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

    // Named-port form. Connection types are encoded as:
    //   input connection:  applicationIndex * inputNames.size() + inputPortIndex
    //   output connection: applicationIndex * outputNames.size() + outputPortIndex
    //
    // Input and output names could be taken from the expression, but then a name collision
    // between a parameter with either an input or output could not be detected.
    // The thinking is that needing to know the input and output names before layer compile time
    // is necessary anyway, so that the proper connections can be made at the graph level
    // before compiling it, so requiring input names and output names to be specified
    // helps with readablity, explicitness and safety, without taking anything away
    // from dynamic compile time logic.
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
    void pruneUpstreamErrorOutputsForApplication(uint32_t applicationIndex);

   private:
    struct ApplicationState {
        std::set<unsigned long> allForwardInputTensorIds;
        std::set<unsigned long> stillWaitingForForwardInputTensorIds;
        std::set<unsigned long> allBackwardErrorInputTensorIds;
        std::set<unsigned long> stillWaitingForBackwardErrorInputTensorIds;

        bool forwardRanThisPass = false;
        bool backwardRanThisPass = false;
        bool backwardGradientPatternCompiled = false;

        std::set<unsigned long> expectedBackwardErrorInputTensorIds;
        std::unordered_map<std::string, std::string> upstreamInputNamesByOutput;
        std::unordered_set<std::string> upstreamOutputNames;
        std::unordered_set<std::string> activeParameterTargetNames;

        std::shared_ptr<PreparedDynamicExpression> forwardPrepared;
        std::shared_ptr<StampedExecutionPlan> forwardStamped;
        std::shared_ptr<StampedExecutionPlan> backwardErrorStamped;
        std::shared_ptr<StampedExecutionPlan> backwardWeightsClearStamped;
        std::shared_ptr<StampedExecutionPlan> backwardWeightsAccumulateStamped;

        std::unordered_map<std::string, Tensor> forwardInputsByName;
        std::unordered_map<std::string, Tensor> forwardOutputsByName;
        std::unordered_map<std::string, Tensor> backwardAdditionalInputsByName;
        std::unordered_map<std::string, Tensor> backwardInputGradOutputsByName;
    };

    struct DecodedConnection {
        uint32_t applicationIndex;
        uint32_t portIndex;
    };

    static void validatePortNames(const std::vector<std::string>& names, const std::string& what);

    uint32_t inputFlatIndex(uint32_t applicationIndex, uint32_t inputPortIndex) const;
    uint32_t outputFlatIndex(uint32_t applicationIndex, uint32_t outputPortIndex) const;
    DecodedConnection decodeInputConnectionType(int connectionType) const;
    DecodedConnection decodeOutputConnectionType(int connectionType) const;

    uint32_t primaryInputFlatIndex(uint32_t applicationIndex) const;
    Stream& computeStream(uint32_t applicationIndex);
    const Stream& computeStream(uint32_t applicationIndex) const;
    Stream& computeStream();
    const Stream& computeStream() const;

    void ensureApplicationStorageAllocated(uint32_t applicationIndex);
    void ensurePortStorageAllocated();
    bool applicationHasAllInputPortsConnected(uint32_t applicationIndex) const;
    void requireApplicationInputInterfaceConnected(uint32_t applicationIndex) const;
    void clearForwardArrivalBookkeeping(uint32_t applicationIndex);
    void clearForwardArrivalBookkeeping();
    void clearBackwardArrivalBookkeeping(uint32_t applicationIndex);
    void clearBackwardArrivalBookkeeping();
    bool applicationHasAnyDownstreamBackprop(uint32_t applicationIndex) const;
    void recordEffectiveParameterBatchSizeForApplication(uint32_t applicationIndex, uint32_t batchSize);

    PreparedDynamicExpression::TensorMap buildForwardInputs(uint32_t applicationIndex);
    PreparedDynamicExpression::TensorMap buildForwardOutputs(uint32_t applicationIndex) const;
    PreparedDynamicExpression::TensorMap buildBackwardAdditionalInputs(uint32_t applicationIndex) const;
    PreparedDynamicExpression::TensorMap buildBackwardInputGradOutputs(uint32_t applicationIndex) const;

    Optional<Tensor> inferFeatureOutputTensor(uint32_t applicationIndex, uint32_t outputPortIndex);
    void validatePreparedExpressionInputs(const PreparedDynamicExpression& prepared);
    void validateStampedOutputNames(const StampedExecutionPlan& stamped, const std::vector<std::string>& expectedNames, const char* phase);
    void synchronizeComputeStreamForForwardInputs(uint32_t applicationIndex);

    std::string errorInputNameForOutput(uint32_t outputPortIndex) const;
    std::string errorOutputNameForInput(uint32_t inputPortIndex) const;

   private:
    DynamicExpression layerDefinitionExpression;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

    bool useFastMath = false;

    std::string customLayerName = "UnnamedType";

    bool clearGradientFirstThisBackwardPass = false;
    uint32_t numBackwardApplications = 0;
    uint32_t numBackwardApplicationsCompletedThisPass = 0;
    std::unordered_map<std::string, uint64_t> effectiveBatchSizeByParameterName;

    std::unordered_map<std::string, uint32_t> inputNameToPort;
    std::unordered_map<std::string, uint32_t> outputNameToPort;

    std::vector<ApplicationState> applications;

    std::vector<Optional<Tensor>> featureInputsConnectedForPorts;
    std::vector<Optional<Tensor>> featureOutputsConnectedForPorts;
    std::vector<Optional<Tensor>> errorInputsConnectedForPorts;
    std::vector<Optional<Tensor>> errorOutputsConnectedForPorts;
};

}  // namespace ThorImplementation
