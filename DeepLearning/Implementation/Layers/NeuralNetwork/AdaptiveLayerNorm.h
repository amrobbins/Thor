#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <array>
#include <optional>
#include <set>
#include <vector>

namespace ThorImplementation {

class AdaptiveLayerNorm : public Layer {
   public:
    enum InputPort : uint32_t { DATA = 0, SCALE = 1, BIAS = 2, NUM_INPUT_PORTS = 3 };

    ~AdaptiveLayerNorm() override;

    AdaptiveLayerNorm(const TensorPlacement& placement,
                      bool inferenceOnly,
                      std::vector<uint64_t> normalizedShape,
                      std::optional<double> epsilon = std::nullopt,
                      std::optional<TensorDescriptor::DataType> scaleBiasDataType = std::nullopt,
                      int64_t stampedId = -1);

    std::string getLayerType() { return "AdaptiveLayerNorm"; }
    std::string getType() override { return getLayerType(); }

    const std::vector<uint64_t>& getNormalizedShape() const { return normalizedShape; }
    uint64_t getNormalizedFeatureCount() const { return normalizedFeatureCount; }
    double getEpsilon() const { return epsilon; }
    TensorDescriptor::DataType getScaleBiasDataType() const { return scaleBiasDataType; }

    void setEpsilon(double value);

    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) override;

    std::optional<Tensor> connectToPreviousLayer(
        Layer* previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override;
    void connectToNextLayer(Layer* nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override;
    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override;

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override;
    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override;

    TensorPlacement getPlacement() override;
    bool isBackPropStub() override;

    uint64_t floatingPointOperationsPerExampleForward() override;
    uint64_t floatingPointOperationsPerExampleBackward() override;

    void initialize() override;
    void cleanup() override;

   protected:
    void compileImpl() override;

   private:
    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override;
    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override;

    static uint64_t checkedNormalizedFeatureCount(const std::vector<uint64_t>& normalizedShape);
    static uint32_t decodeInputConnectionType(int connectionType);
    void validateConfiguredInput(const Tensor& input) const;
    uint64_t computeBatchSize(const Tensor& input) const;
    uint64_t computeLeadingFeatureCount(const Tensor& input) const;
    void validateScaleBiasInput(const Tensor& tensor, const Tensor& data, const char* name) const;
    void validateAllConnectedInputs() const;
    bool anyErrorOutput() const;
    Stream& computeStream();
    const Stream& computeStream() const;
    void pruneUpstreamErrorOutputs();
    void resetForwardArrivalBookkeeping();

    TensorPlacement placement;
    int64_t stampedId = -1;
    std::vector<uint64_t> normalizedShape;
    uint64_t normalizedFeatureCount = 0;
    uint64_t batchSize = 0;
    uint64_t leadingFeatureCount = 0;
    double epsilon = 1.0e-5;
    TensorDescriptor::DataType scaleBiasDataType = TensorDescriptor::DataType::FP32;

    std::array<std::optional<Tensor>, NUM_INPUT_PORTS> adaptiveFeatureInputs;
    std::array<std::optional<Tensor>, NUM_INPUT_PORTS> adaptiveErrorOutputs;
    std::array<std::optional<Layer*>, NUM_INPUT_PORTS> adaptivePreviousLayers;
    std::array<std::optional<Stream>, NUM_INPUT_PORTS> adaptiveStreams;

    std::set<unsigned long> allForwardInputTensorIds;
    std::set<unsigned long> stillWaitingForForwardInputTensorIds;

    Tensor saveMean;
    Tensor saveInvVariance;
    std::array<std::optional<Tensor>, NUM_INPUT_PORTS> scratchErrorOutputs;
};

}  // namespace ThorImplementation
