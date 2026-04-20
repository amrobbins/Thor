#pragma once

#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

namespace ThorImplementation {
class CustomLayer : public TrainableLayer {
   public:
    virtual ~CustomLayer() = default;

    // FIXME: Initially only supporting 1 output, I could support multiple via Outputs, but then how to connect layers?
    //   Well I do support multiple outputs. Get single working first!
    //   Also start with just 1 input
    //   So version 1 support: 1 input, 1 output, N parameters.
    CustomLayer(DynamicExpression expr,
                const TensorPlacement& placement,
                const std::vector<std::shared_ptr<Parameter>>& parameters,
                bool inferenceOnly,
                int64_t stampedId = -1,
                bool useFastMath = false);

    // TrainableLayer
    // void createWeights(Tensor featureInput) override { /*NOP*/ } - part of parameter now.
    // void createGradients() override { /*NOP*/ }

    // Compute feature output on the data stream
    void computeFeatureOut(uint32_t connectionNumber) override;

    // Gradient-update stream synchronization is handled by TrainableLayer::backward().
    void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) override;

    // Error-output backward work runs on the regular data stream.
    Optional<Event> computeErrorOut(uint32_t connectionNumber, bool clearWeightsGradientFirstIfFused) override;

    Optional<Tensor> createFeatureOutputTensor() override;
    Optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) override;

    uint64_t flopCountForward() override;
    uint64_t flopCountBackward() override;

    std::string getLayerType() override { return "CustomLayer"; }

   protected:
    void compileImpl() override;

   private:
    virtual Optional<Tensor> stampForward(uint32_t connectionNumber);
    virtual Optional<Tensor> stampBackward(uint32_t connectionNumber);
    std::unordered_map<std::string, Tensor> buildForwardInputs(const Tensor& dataIn);
    void validatePreparedExpressionInputs(const PreparedDynamicExpression& prepared);

   private:
    DynamicExpression layerDefinitionExpression;
    std::string inputName;

    bool useFastMath = false;

    std::vector<std::shared_ptr<FusedEquation>> forwardEq;
    std::vector<std::shared_ptr<FusedEquation>> backwardClearEq;
    std::vector<std::shared_ptr<FusedEquation>> backwardAccumulateEq;

    std::vector<std::shared_ptr<StampedExecutionPlan>> forwardStamped;
    std::vector<std::shared_ptr<StampedExecutionPlan>> backwardClearStamped;
    std::vector<std::shared_ptr<StampedExecutionPlan>> backwardAccumulateStamped;

    std::vector<std::unordered_map<std::string, Tensor>> forwardInputs;
    std::vector<std::unordered_map<std::string, Tensor>> forwardOutputs;
    std::vector<std::unordered_map<std::string, Tensor>> backwardInputs;
    std::vector<std::unordered_map<std::string, Tensor>> backwardOutputs;

    std::vector<std::unordered_map<std::string, Tensor>> forwardInputsByConnection;
    std::vector<std::shared_ptr<PreparedDynamicExpression>> forwardPreparedByConnection;
    std::vector<std::shared_ptr<StampedExecutionPlan>> forwardStampedByConnection;

    std::vector<std::shared_ptr<StampedExecutionPlan>> backwardErrorStampedByConnection;
    std::vector<std::shared_ptr<StampedExecutionPlan>> backwardWeightsClearStampedByConnection;
    std::vector<std::shared_ptr<StampedExecutionPlan>> backwardWeightsAccumulateStampedByConnection;
    std::vector<std::unordered_map<std::string, Tensor>> backwardOutputsByConnection;

    const std::string RESERVED_GRAD_PREFIX = "__grad_";
    const std::string featureInName = "feature_input";
    const std::string featureOutName = "feature_output";
    const std::string errorInName = RESERVED_GRAD_PREFIX + featureOutName;
    std::string errorOutName() const { return inputName + "_grad"; }
    std::vector<Event> gradientAccumAvailableEvents;
};

}  // namespace ThorImplementation
