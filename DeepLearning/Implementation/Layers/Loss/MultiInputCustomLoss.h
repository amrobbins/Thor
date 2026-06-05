#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/Expression/DynamicExpression.h"

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace ThorImplementation {

class MultiInputCustomLoss : public Layer {
   public:
    MultiInputCustomLoss(DynamicExpression lossExpression,
                         DynamicExpression gradientExpression,
                         std::vector<std::string> inputNames,
                         std::vector<std::optional<std::string>> gradientNames,
                         std::string lossName = "loss",
                         DataType lossDataType = DataType::FP32);

    ~MultiInputCustomLoss() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override;
    void connectToNextLayer(Layer* nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override;
    std::optional<Tensor> connectToPreviousLayer(
        Layer* previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override;
    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override;
    void compileImpl() override;
    void cleanup() override;
    void initialize() override;

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override;
    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override;

    std::string getType() override { return "MultiInputCustomLoss"; }
    std::optional<Tensor> getErrorOutput(uint32_t inputIndex) const;
    std::vector<std::optional<Tensor>> getErrorOutputs() const { return errorOutputs; }
    Stream getStream() override;

   protected:
    using TensorMap = std::unordered_map<std::string, Tensor>;

    TensorMap buildLossInputs() const;
    TensorMap buildLossOutputs() const;
    TensorMap buildGradientOutputs() const;

    void validateExpressionOutputNames(const DynamicExpression& expression,
                                       const std::set<std::string>& expectedOutputNames,
                                       const std::string& what) const;
    std::pair<std::vector<uint64_t>, DataType> inferExpressionOutputDescriptor(const DynamicExpression& expression,
                                                                                const std::string& outputName,
                                                                                const std::string& what) const;

    DynamicExpression lossExpression;
    DynamicExpression gradientExpression;
    std::vector<std::string> inputNames;
    std::vector<std::optional<std::string>> gradientNames;
    std::string lossName;
    DataType lossDataType;

    std::vector<std::optional<Tensor>> featureInputs;
    std::vector<std::optional<Tensor>> errorOutputs;
    std::vector<Stream> inputStreams;
    std::vector<std::optional<Layer*>> previousLayers;

    std::set<unsigned long> allForwardInputTensorIds;
    std::set<unsigned long> stillWaitingForForwardInputTensorIds;
    uint32_t currentBatchSize = 0;

    std::shared_ptr<PreparedDynamicExpression> lossPrepared;
    std::shared_ptr<StampedExecutionPlan> lossStamped;
    std::function<void(Stream&)> lossPreRunHook;

    std::shared_ptr<PreparedDynamicExpression> gradientPrepared;
    std::shared_ptr<StampedExecutionPlan> gradientStamped;
    std::function<void(Stream&)> gradientPreRunHook;

   private:
    static std::string joinNames(const std::set<std::string>& names);
    static std::set<std::string> presentNames(const std::vector<std::optional<std::string>>& names);
    static DataType findOutputDType(const std::shared_ptr<CompiledOutputs>& compiledOutputs, const std::string& outputName);
    uint32_t requireInputIndexFromConnectionType(int connectionType) const;
    Stream& computeStream();
    const Stream& computeStream() const;
    void synchronizeComputeStreamForInputs();
    void resetForwardBookkeeping();

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override;
    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override;
    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) override;
};

}  // namespace ThorImplementation
