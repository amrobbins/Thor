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
                std::vector<std::shared_ptr<Parameter>> parameters,
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
    void accumulateGradient(uint32_t connectionNumber, bool clearGradientFirst) override { /* NOP */ };

    // Error in is up-to-date by the end of the data stream.
    Optional<Event> computeErrorOut(uint32_t connectionNumber) override;

    // TrainableWeightsBiasesLayer abstract methods
    virtual void computeGradient(Optional<Tensor> weightsGradient,
                                 Optional<Tensor> biasesGradient,
                                 Optional<Tensor> featureIn,
                                 Optional<Tensor> errorIn,
                                 Stream gradientUpdateStream,
                                 bool accumulateGradient) { /*NOP*/ }

    virtual void infer(Optional<Tensor> inputTensor,
                       Optional<Tensor> outputTensor,
                       Stream stream,
                       unsigned int connectionNumber,
                       Tensor weights,
                       Optional<Tensor> biases);

    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient);

    virtual Optional<Tensor> createFeatureOutputTensor();
    virtual Optional<Tensor> createErrorOutputTensor(bool backPropagateError);

   private:
    void compileImpl() override;
    virtual void stampForward(uint32_t connectionNumber, Tensor featureInput);
    virtual void stampBackward(uint32_t connectionNumber, Tensor featureInput, Tensor errorInput);
    std::unordered_map<std::string, Tensor> buildForwardInputs(const Tensor& dataIn);
    std::unordered_map<std::string, Tensor> buildForwardOutputs(const Tensor& dataIn);
    std::unordered_map<std::string, Tensor> buildBackwardInputs(const Tensor& dataIn, const Tensor& errorIn);

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

    const std::string RESERVED_GRAD_PREFIX = "__grad_";
    const std::string featureInName;
    const std::string featureOutName = "output";
    const std::string errorInName = RESERVED_GRAD_PREFIX + featureOutName;
    const std::string errorOutName;
    std::vector<Event> gradientAccumAvailableEvents;
};

}  // namespace ThorImplementation
