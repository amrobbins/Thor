#pragma once

#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {
class CustomLayer : public TrainableLayer {
   public:
    using DataType = TensorDescriptor::DataType;

    virtual ~CustomLayer() = default;

    // FIXME: Initially only supporting 1 output, I could support multiple via Outputs, but then how to connect layers?
    //   Well I do support multiple outputs. Get single working first!
    //   Also start with just 1 input
    //   So version 1 support: 1 input, 1 output, N parameters.
    CustomLayer(DynamicExpression expr,
                // V1 Assumption: Exactly 1 input. V2 could be multiple or none even. feature inputs would need to be a map string->tensor.
                const std::string& inputName,
                const std::vector<std::shared_ptr<Parameter>>& parameters,
                int deviceNum,
                bool useFastMath,
                int64_t stampedId = -1);

    // TrainableLayer
    // void createWeights(Tensor featureInput) override { /*NOP*/ } - part of parameter now.
    // void createGradients() override { /*NOP*/ }

    // Compute feature output on the data stream
    void computeFeatureOut(uint32_t connectionNumber) override;

    // Error in is up-to-date by the end of the data stream.
    // Gradient update stream must wait for that.
    void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) override;

    // Error in is up-to-date by the end of the data stream.
    Optional<Event> computeErrorOut(uint32_t connectionNumber) override;

    virtual Optional<Tensor> createFeatureOutputTensor();
    virtual Optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber);

    std::string getLayerType() override { return "CustomLayer"; }

    bool canFuseBackwardEoutWgrad() const;

   private:
    void compileImpl() override;
    virtual Optional<Tensor> stampForward(uint32_t connectionNumber);
    virtual Optional<Tensor> stampBackward(uint32_t connectionNumber);
    std::unordered_map<std::string, Tensor> buildForwardInputs(const Tensor& dataIn);
    void validatePreparedExpressionInputs(const PreparedDynamicExpression& prepared);

   private:
    DynamicExpression layerDefinitionExpression;
    std::string inputName;

    int deviceNum = 0;
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
    const std::string featureInName;
    const std::string featureOutName = "output";
    const std::string errorInName = RESERVED_GRAD_PREFIX + featureOutName;
    const std::string errorOutName;
    std::vector<Event> gradientAccumAvailableEvents;

    bool fuseBackwardEoutWgrad = false;
};

}  // namespace ThorImplementation
