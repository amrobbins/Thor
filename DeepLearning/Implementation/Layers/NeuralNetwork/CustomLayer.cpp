// #include "DeepLearning/Implementation/Layers/NeuralNetwork/CustomLayer.h"
//
// #include <stdexcept>
//
// using namespace std;
//
// namespace ThorImplementation {
//
// CustomLayer::CustomLayer(DynamicExpression expr,
//                          const string& inputName,
//                          vector<shared_ptr<Parameter>> parameters,
//                          int deviceNum,
//                          bool useFastMath,
//                          int64_t stampedId)
//     : TrainableLayer(stampedId),
//       layerDefinitionExpression(std::move(expr)),
//       inputName(inputName),
//       deviceNum(deviceNum),
//       useFastMath(useFastMath),
//       featureInName(inputName),
//       errorOutName(inputName + "_grad") {
//     set<string> expressionInputs = expr.getInputNames();
//     if (inputName.length() >= 2 && inputName[0] == '_' && inputName[1] == '_')
//         throw runtime_error("Custom layer input names cannot start with __ that is reserved. Input name " + inputName + " is illegal.");
//     if (inputName.empty())
//         throw runtime_error("Custom layer input sent empty name");
//     if (!expressionInputs.contains(inputName)) {
//         string expectedInputNamesString;
//         for (const auto& name : expressionInputs)
//             expectedInputNamesString += name + " ";
//         throw runtime_error("Provided input name: " + inputName +
//                             " is not part of the expression, the expression has inputs: " + expectedInputNamesString);
//     }
//     for (const auto& param : parameters) {
//         string paramName = param->getName();
//         if (!expressionInputs.contains(paramName)) {
//             string expectedInputNamesString;
//             for (const auto& name : expressionInputs)
//                 expectedInputNamesString += name + " ";
//             throw runtime_error("Provided parameter name: " + paramName +
//                                 " is not an used by the expression, the expression has inputs: " + expectedInputNamesString);
//         }
//         addParam(param);  // verifies name uniqueness
//     }
//     const uint32_t numInputsSent = parameters.size() + 1;
//     if (numInputsSent != expressionInputs.size()) {
//         string expectedInputNamesString;
//         for (const auto& name : expressionInputs)
//             expectedInputNamesString += name + " ";
//         string actualInputNamesString = inputName + " ";
//         for (const auto& param : parameters) {
//             expectedInputNamesString += param->getName() + " ";
//         }
//         throw runtime_error("Wrong number of inputs and parameters for the expression, sent " + to_string(numInputsSent) + " expected " +
//                             to_string(expressionInputs.size()) + ". Expected inputs: " + expectedInputNamesString +
//                             " Actual inputs: " + actualInputNamesString);
//     }
// }
//
// void CustomLayer::compileImpl() { TrainableLayer::compileImpl(); }
//
// void CustomLayer::stampForward(uint32_t connectionNumber, Tensor featureInput) {
//     assert(connectionNumber == forwardStamped.size());
//     assert(forwardInputs.size() == forwardStamped.size());
//     forwardInputs.push_back(buildForwardInputs(featureInput));
//     StampedExecutionPlan layerEquationStamped = layerDefinitionExpression.stamp(forwardInputs.back(), streams[connectionNumber]);
//     forwardStamped.push_back(make_shared<StampedExecutionPlan>(layerEquationStamped));
//     // FIXME: I need a stamp backwards in DynamicExpression
// }
//
// void CustomLayer::stampBackward(uint32_t connectionNumber, Tensor featureInput, Tensor errorInput) {
//     if (isInferenceOnly())
//         return;
//
//     // Then compile back prop, when there is a needed gradient (e.g. errorOut or parameter update)
//     vector<string> backwardTargets;
//
//     if (!isBackPropStub()) {
//         // Compute error out
//         backwardTargets.push_back(inputName);
//     }
//     for (const auto& param : parameters) {
//         if (param->isTrainable()) {
//             backwardTargets.push_back(param->getName());
//         }
//     }
//     if (backwardTargets.empty()) {
//         backwardAccumulateEq = nullptr;
//         backwardClearEq = nullptr;
//         return;
//     }
//
//     unordered_map<string, string> featureOutputNameToErrorInputName;
//     featureOutputNameToErrorInputName[featureOutName] = errorInName;
//     backwardClearEq = std::make_shared<FusedEquation>(forwardEq->compileBackward(backwardTargets, featureOutputNameToErrorInputName));
//     backwardAccumulateEq =
//         std::make_shared<FusedEquation>(forwardEq->compileBackward(backwardTargets, featureOutputNameToErrorInputName, true));
//     backwardInputs = buildBackwardInputs(featureInput, errorInput);
//
//     // computeErrorOut is called when error in is ready on the data stream
//     // gradient stream is then synced with the data stream before starting gradient computation
//     // Here we compute both errorOut and gradient at once, so we do it on the '''data stream'''.
//     // But two data streams can't both be accumulating into a shared gradient buffer at once,
//     // so events are used to serialize gradient accumulation between the data streams.
//     // and CustomLayer::accumulateGradient(...) is a no-op.
//     // FIXME: Explicity pass the parameter gradient buffers for those parameters where training is enabled. Reuse them on both stamps.
//     //        Otherwise would lose gradients when re-stamping, unless it was explicitly fetched first.
//     //        This is cleaner cause have fixed tensors provided as parameters.
//     backwardClearStamped = make_shared<StampedExecutionPlan>(backwardClearEq->stamp(backwardInputs, streams[connectionNumber]));
//
//     // Bind the shared gradient accumulators that both backward variants should write into.
//     backwardOutputs.clear();
//     for (const string& backwardTarget : backwardTargets) {
//         string backwardGrad = backwardTarget + "_grad";
//         backwardOutputs[backwardGrad] = backwardClearStamped->output(backwardGrad);
//     }
//     backwardAccumulateStamped = make_shared<StampedExecutionPlan>(
//         backwardAccumulateEq->stamp(backwardInputs, backwardOutputs, streams[connectionNumber]);
// }
//
// // Note: A featureInput is guaranteed to be connected before createFeatureOutputTensor() is called.
// Optional<Tensor> CustomLayer::createFeatureOutputTensor() {
//     // Feature input is already connected
//     stampForward(featureInputs[0]);
//     // Error input is not yet connected, since it needs to be shaped like the feature output that I am creating here.
//     return forwardStamped->output(featureOutName);
// }
//
// Optional<Tensor> CustomLayer::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
//     if (backPropagateError && !isInferenceOnly()) {
//         assert(errorInputs.size() > connectionNumber);
//         assert(errorInputs[connectionNumber].isPresent());
//         stampBackward(featureInputs[connectionNumber], errorInputs[connectionNumber]);
//         return backwardStamped->output(errorOutName);
//     } else {
//         return Optional<Tensor>::empty();
//     }
// }
//
// unordered_map<string, Tensor> CustomLayer::buildForwardInputs(const Tensor& dataIn) {
//     unordered_map<string, Tensor> inputs;
//     inputs[inputName] = dataIn;
//     for (const auto& parameter : parameters) {
//         inputs[parameter->getName()] = parameter->getStorage();
//     }
//     return inputs;
// }
//
// unordered_map<string, Tensor> CustomLayer::buildBackwardInputs(const Tensor& dataIn, const Tensor& errorIn) {
//     unordered_map<string, Tensor> inputs = buildForwardInputs(dataIn);
//     inputs[errorInName] = errorIn;
//     return inputs;
// }
//
// void CustomLayer::computeFeatureOut(uint32_t connectionNumber) {
//     if (featureOutputs.empty())
//         throw runtime_error("CustomLayer::infer requires an output tensor.");
//     if (featureOutputs[0].isEmpty())
//         throw runtime_error("CustomLayer::infer requires a present output tensor.");
//
//     // V1 Assumption: Exactly 1 input. V2 could be multiple or none even.
//     if (featureInputs.empty())
//         throw runtime_error("V1 CustomLayer::infer requires an input tensor.");
//     if (featureInputs[0].isEmpty())
//         throw runtime_error("V1 CustomLayer::infer requires a present input tensor.");
//
//     forwardStamped->run();
// }
//
// }  // namespace ThorImplementation
