#pragma once

#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {
class CustomLayer : public TrainableWeightsBiasesLayer, public Parameterizable {
   public:
    using DataType = TensorDescriptor::DataType;

    virtual ~CustomLayer() = default;

    // FIXME: Initially only supporting 1 output, I could support multiple via Outputs, but then how to connect layers?
    //   Well I do support multiple outputs. Get single working first!
    //   Also start with just 1 input
    //   So version 1 support: 1 input, 1 output, N parameters.
    CustomLayer(Expression expr,
                // V1 Assumption: Exactly 1 input. V2 could be multiple or none even.
                const std::string& inputName,
                std::vector<std::shared_ptr<Parameter>> parameters,
                int deviceNum,
                bool useFastMath,
                int64_t stampedId = -1);

    // TrainableWeightsBiasesLayer abstract methods
    virtual void createWeightsIfNecessary() { /*NOP*/ }
    virtual void computeWeightsGradient(Optional<Tensor> weightsGradient,
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
    virtual Optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber);

   private:
    virtual void compileImpl();
    virtual void stampForward(Tensor featureInput);
    virtual void stampBackward(Tensor featureInput, Tensor errorInput);
    std::unordered_map<std::string, Tensor> buildForwardInputs(const Tensor& dataIn);
    std::unordered_map<std::string, Tensor> buildBackwardInputs(const Tensor& dataIn, const Tensor& errorIn);

   private:
    Expression expr;
    std::string inputName;
    std::vector<std::string> parameterNames;

    int deviceNum = 0;
    bool useFastMath = false;

    uint32_t batchSize;

    std::shared_ptr<FusedEquation> forwardEq = nullptr;
    std::shared_ptr<FusedEquation> backwardEq = nullptr;

    std::shared_ptr<StampedExecutionPlan> forwardStamped = nullptr;
    std::shared_ptr<StampedExecutionPlan> backwardStamped = nullptr;

    std::unordered_map<std::string, Tensor> forwardInputs;
    std::unordered_map<std::string, Tensor> backwardInputs;

    const std::string RESERVED_GRAD_PREFIX = "__grad_";
    const std::string featureInName;
    const std::string featureOutName = "output";
    const std::string errorInName = RESERVED_GRAD_PREFIX + featureOutName;
    const std::string errorOutName;
};

}  // namespace ThorImplementation
