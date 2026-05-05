#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"

#include <cstdint>
#include <limits>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

uint32_t cloneExpressionSubtreeWithSubstitution(const ThorImplementation::PhysicalExpression& src,
                                                uint32_t srcNodeIndex,
                                                const std::string& substituteInputName,
                                                uint32_t substituteNodeIndex,
                                                ThorImplementation::PhysicalExpression& dst,
                                                std::unordered_map<uint32_t, uint32_t>& oldToNew) {
    using ThorImplementation::ExprNode;
    using ThorImplementation::ExprOp;
    using ThorImplementation::Expression;

    auto it = oldToNew.find(srcNodeIndex);
    if (it != oldToNew.end()) {
        return it->second;
    }
    if (srcNodeIndex >= src.nodes.size()) {
        throw std::runtime_error("Epilogue expression node index is out of range.");
    }

    const ExprNode& srcNode = src.nodes[srcNodeIndex];
    if (srcNode.op == ExprOp::INPUT) {
        if (srcNode.input_slot >= src.inputs.size()) {
            throw std::runtime_error("Epilogue expression input slot is out of range.");
        }
        const ThorImplementation::NamedInput& input = src.inputs[srcNode.input_slot];
        if (input.name != substituteInputName) {
            throw std::runtime_error("FullyConnected epilogue expression contains unsupported input '" + input.name + "'.");
        }
        oldToNew[srcNodeIndex] = substituteNodeIndex;
        return substituteNodeIndex;
    }
    if (srcNode.op == ExprOp::RUNTIME_SCALAR || srcNode.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        throw std::runtime_error("FullyConnected epilogue expression cannot contain runtime scalar inputs.");
    }

    ExprNode newNode = srcNode;
    if (Expression::isUnaryOp(srcNode.op)) {
        newNode.lhs = cloneExpressionSubtreeWithSubstitution(src, srcNode.lhs, substituteInputName, substituteNodeIndex, dst, oldToNew);
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isBinaryOp(srcNode.op)) {
        newNode.lhs = cloneExpressionSubtreeWithSubstitution(src, srcNode.lhs, substituteInputName, substituteNodeIndex, dst, oldToNew);
        newNode.rhs = cloneExpressionSubtreeWithSubstitution(src, srcNode.rhs, substituteInputName, substituteNodeIndex, dst, oldToNew);
        newNode.aux = UINT32_MAX;
    } else if (Expression::isTernaryOp(srcNode.op)) {
        newNode.lhs = cloneExpressionSubtreeWithSubstitution(src, srcNode.lhs, substituteInputName, substituteNodeIndex, dst, oldToNew);
        newNode.rhs = cloneExpressionSubtreeWithSubstitution(src, srcNode.rhs, substituteInputName, substituteNodeIndex, dst, oldToNew);
        newNode.aux = cloneExpressionSubtreeWithSubstitution(src, srcNode.aux, substituteInputName, substituteNodeIndex, dst, oldToNew);
        if (srcNode.alpha_node != UINT32_MAX) {
            newNode.alpha_node = cloneExpressionSubtreeWithSubstitution(
                src, srcNode.alpha_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
        if (srcNode.beta_node != UINT32_MAX) {
            newNode.beta_node = cloneExpressionSubtreeWithSubstitution(
                src, srcNode.beta_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
    } else if (Expression::isLeafOp(srcNode.op)) {
        // constants/fill nodes have no children and can be copied directly.
    } else {
        throw std::runtime_error("Unsupported op while applying FullyConnected epilogue expression: " +
                                 std::to_string(static_cast<int>(srcNode.op)));
    }

    const uint32_t newIndex = static_cast<uint32_t>(dst.nodes.size());
    dst.nodes.push_back(std::move(newNode));
    oldToNew[srcNodeIndex] = newIndex;
    return newIndex;
}

ThorImplementation::Expression applyFullyConnectedEpilogue(const ThorImplementation::Expression& input,
                                                           const ThorImplementation::ExpressionDefinition& epilogue) {
    FullyConnected::validateEpilogueDefinition(epilogue);

    ThorImplementation::PhysicalExpression inputPhysical = input.expression();
    if (inputPhysical.output_node >= inputPhysical.nodes.size()) {
        throw std::runtime_error("FullyConnected epilogue input expression has an invalid output node.");
    }

    auto composed = std::make_shared<ThorImplementation::PhysicalExpression>();
    composed->inputs = inputPhysical.inputs;
    composed->nodes = inputPhysical.nodes;

    const uint32_t epilogueRoot = epilogue.outputs.outputs.front().node_idx;
    std::unordered_map<uint32_t, uint32_t> oldToNew;
    const uint32_t composedRoot = cloneExpressionSubtreeWithSubstitution(*epilogue.outputs.expr,
                                                                         epilogueRoot,
                                                                         FullyConnected::epilogueInputName(),
                                                                         inputPhysical.output_node,
                                                                         *composed,
                                                                         oldToNew);
    composed->output_node = composedRoot;
    return ThorImplementation::Expression::fromPhysicalNode(std::move(composed), composedRoot);
}

ThorImplementation::DynamicExpression buildFullyConnectedExpression(bool hasBias,
                                                                    ThorImplementation::TensorPlacement placement,
                                                                    Tensor::DataType weightsDataType,
                                                                    Tensor::DataType computeDataType,
                                                                    Tensor::DataType outputDataType,
                                                                    std::shared_ptr<Thor::Activation> activation,
                                                                    Optional<ThorImplementation::ExpressionDefinition> epilogue) {
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    std::vector<std::string> expectedInputNames = {"feature_input", "weights"};
    if (hasBias) {
        expectedInputNames.push_back("biases");
    }

    return DynamicExpression(
        std::move(expectedInputNames),
        {"feature_output"},
        [hasBias, placement, weightsDataType, computeDataType, outputDataType, activation = std::move(activation), epilogue](
            const DynamicExpression::TensorMap& inputs,
            const DynamicExpression::TensorMap& outputs,
            Stream& stream) -> DynamicExpressionBuild {
            (void)stream;

            Tensor featureInputTensor = inputs.at("feature_input");
            const Tensor& wTensor = inputs.at("weights");
            if (wTensor.getDimensions().size() != 2) {
                throw std::runtime_error("FullyConnected weights tensor must be rank 2.");
            }
            if (wTensor.getDataType() != weightsDataType) {
                throw std::runtime_error("FullyConnected weights tensor dtype does not match weightsDataType.");
            }
            if (wTensor.getPlacement() != placement) {
                throw std::runtime_error("FullyConnected weights tensor placement does not match the layer placement.");
            }

            std::vector<uint64_t> featureInputDimensions = featureInputTensor.getDimensions();
            if (featureInputDimensions.size() < 2) {
                throw std::runtime_error("FullyConnected dynamic expression requires a feature input tensor with batch plus at least one feature dimension.");
            }
            if (featureInputTensor.getPlacement() != placement) {
                throw std::runtime_error("FullyConnected feature input placement does not match the layer placement.");
            }

            // Treat any rank > 2 input as [batch, flattened_features] for the matrix multiply, without touching the
            // original Tensor object owned by the surrounding graph. Tensor is a lightweight metadata/storage alias,
            // so this reshape changes only this DynamicExpression's logical view.
            if (featureInputDimensions.size() > 2) {
                const uint64_t batchSize = featureInputDimensions[0];
                if (batchSize == 0) {
                    throw std::runtime_error("FullyConnected runtime batch dimension must be non-zero.");
                }
                uint64_t flattenedFeatures = 1;
                for (uint32_t i = 1; i < featureInputDimensions.size(); ++i) {
                    if (featureInputDimensions[i] == 0) {
                        throw std::runtime_error("FullyConnected runtime feature dimensions must be non-zero.");
                    }
                    if (flattenedFeatures > std::numeric_limits<uint64_t>::max() / featureInputDimensions[i]) {
                        throw std::runtime_error("FullyConnected flattened feature count overflows uint64_t.");
                    }
                    flattenedFeatures *= featureInputDimensions[i];
                }
                featureInputTensor.reshape({batchSize, flattenedFeatures});
                featureInputDimensions = featureInputTensor.getDimensions();
            }

            if (featureInputDimensions.size() != 2) {
                throw std::runtime_error("FullyConnected logical feature input tensor must be rank 2 after flattening.");
            }
            if (featureInputDimensions[0] == 0 || featureInputDimensions[1] == 0) {
                throw std::runtime_error("FullyConnected logical feature input tensor dimensions must be non-zero.");
            }
            if (featureInputDimensions[1] != wTensor.getDimensions()[0]) {
                throw std::runtime_error("FullyConnected input feature count does not match weights rows.");
            }
            if (outputs.contains("feature_output")) {
                const Tensor& featureOutputTensor = outputs.at("feature_output");
                if (featureOutputTensor.getDimensions().size() != 2) {
                    throw std::runtime_error("FullyConnected feature output tensor must be rank 2.");
                }
                if (featureOutputTensor.getDimensions()[0] != featureInputDimensions[0] ||
                    featureOutputTensor.getDimensions()[1] != wTensor.getDimensions()[1]) {
                    throw std::runtime_error("FullyConnected feature output tensor dimensions are incompatible with the matmul output.");
                }
                if (featureOutputTensor.getDataType() != outputDataType) {
                    throw std::runtime_error("FullyConnected feature output tensor dtype does not match outputDataType.");
                }
                if (featureOutputTensor.getPlacement() != placement) {
                    throw std::runtime_error("FullyConnected feature output tensor placement does not match the layer placement.");
                }
            }

            auto fin = Expression::input("feature_input", featureInputTensor.getDataType(), featureInputTensor.getDataType());
            auto w = Expression::input("weights", weightsDataType, weightsDataType);

            // [batch, in_features] @ [in_features, out_features]
            Expression fout = Expression::matmul(fin, w, false, false, computeDataType, outputDataType);

            if (hasBias) {
                const Tensor& bTensor = inputs.at("biases");
                if (bTensor.getDimensions().size() != 1 || bTensor.getDimensions()[0] != wTensor.getDimensions()[1]) {
                    throw std::runtime_error("FullyConnected biases tensor dimensions are incompatible with the weights tensor.");
                }
                if (bTensor.getDataType() != weightsDataType) {
                    throw std::runtime_error("FullyConnected biases tensor dtype does not match weightsDataType.");
                }
                if (bTensor.getPlacement() != placement) {
                    throw std::runtime_error("FullyConnected biases tensor placement does not match the layer placement.");
                }

                auto b = Expression::input("biases", weightsDataType, weightsDataType);

                // Broadcast [out_features] over batch.
                fout = fout + b;
            }

            if (activation != nullptr) {
                fout = activation->toExpression(fout);
            }
            if (epilogue.isPresent()) {
                fout = applyFullyConnectedEpilogue(fout, epilogue.get());
            }

            // The API layer's declared output tensor dtype is authoritative.
            fout = fout.withOutputDType(outputDataType);

            auto expressionOutputs = Expression::outputs({{"feature_output", fout}});

            DynamicExpression::TensorMap stampInputs = inputs;
            stampInputs["feature_input"] = featureInputTensor;

            return DynamicExpressionBuild{
                std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
                stampInputs,
                {},
                outputs,
                {},
            };
        });
}

}  // namespace

std::shared_ptr<ThorImplementation::Layer> FullyConnected::stamp(ThorImplementation::TensorPlacement placement,
                                                                 std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                                 std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                                 Thor::Tensor connectingApiTensor,
                                                                 const bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;

    assert(initialized);
    assert(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

    // Note: Network notices when a layer has already been stamped and only adds a connection; it does not re-stamp the layer.
    std::shared_ptr<ThorImplementation::CustomLayer> physicalFullyConnected = std::make_shared<ThorImplementation::CustomLayer>(
        buildFullyConnectedExpression(hasBias, placement, weightsDataType, computeDataType, outputDataType, activation, epilogue),
        placement,
        ThorImplementation::FullyConnected::defineParameters(numOutputFeatures, hasBias, weightsDataType),
        inferenceOnly,
        getId(),
        false);
    physicalFullyConnected->setLayerName(getLayerType());

    return physicalFullyConnected;
}

json FullyConnected::architectureJson() const {
    // Multi-layers will only serialize the single layer, itself.
    // The other layers will each serialize themselves when walking the api level layer graph that has been added to the network

    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "fully_connected";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["num_output_features"] = numOutputFeatures;
    j["has_bias"] = hasBias;
    j["weights_data_type"] = weightsDataType;
    j["compute_data_type"] = computeDataType;
    j["output_data_type"] = outputDataType;
    if (epilogue.isPresent()) {
        j["epilogue"] = epilogue.get().architectureJson();
    }

    // Input connections
    json inputs = json::array();
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        inputs.push_back(featureInputs[i].architectureJson());
    }
    j["inputs"] = inputs;

    // Output connections
    json outputs = json::array();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        outputs.push_back(featureOutputs[i].architectureJson());
    }
    j["outputs"] = outputs;

    if (weightsInitializer != nullptr) {
        j["weights_initializer"] = weightsInitializer->architectureJson();
    }
    if (biasesInitializer != nullptr) {
        j["biases_initializer"] = biasesInitializer->architectureJson();
    }

    if (hasOptimizer()) {
        j["weights_optimizer"] = weightsOptimizer->architectureJson();
        if (hasBias) {
            j["biases_optimizer"] = biasesOptimizer->architectureJson();
        }
    }

    return j;
}

json FullyConnected::serialize(thor_file::TarWriter& archiveWriter,
                               Stream stream,
                               bool saveOptimizerState,
                               ThorImplementation::StampedNetwork& stampedNetwork) const {
    // Multi-layers will only serialize the single layer, itself.
    // The other layers will each serialize themselves when walking the api level layer graph that has been added to the network
    json j = architectureJson();

    string layerName = string("layer") + to_string(getId());

    // Dump the weights to a file and record its name
    shared_ptr<ThorImplementation::TrainableLayer> twbLayer = nullptr;
    shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetwork.getPhysicalLayerFromApiLayer(getId());
    twbLayer = dynamic_pointer_cast<ThorImplementation::TrainableLayer>(physicalLayer);
    assert(twbLayer != nullptr);

    ThorImplementation::Tensor weights;
    ThorImplementation::Tensor biases;
    string weightsFile;
    string biasesFile;
    if (twbLayer != nullptr) {
        if (hasBias) {
            biasesFile = (layerName + "_biases.gds");
            j["biases_tensor"] = biasesFile;
            biases = twbLayer->getParameter("biases")->getStorage().get();
            archiveWriter.addArchiveFile(biasesFile, biases);
        }

        weightsFile = (layerName + "_weights.gds");
        j["weights_tensor"] = weightsFile;
        weights = twbLayer->getParameter("weights")->getStorage();
        archiveWriter.addArchiveFile(weightsFile, weights);
    }

    if (hasOptimizer()) {
        j["weights_optimizer"] = weightsOptimizer->serialize(archiveWriter,
                                                             stream,
                                                             twbLayer->getParameter("weights")->getOptimizer(),
                                                             string("layer") + to_string(getId()),
                                                             saveOptimizerState);
        if (hasBias) {
            j["biases_optimizer"] = biasesOptimizer->serialize(archiveWriter,
                                                               stream,
                                                               twbLayer->getParameter("biases")->getOptimizer(),
                                                               string("layer") + to_string(getId()),
                                                               saveOptimizerState);
        }
    }

    return j;
}

void FullyConnected::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    // FIXME
    // if (j.at("version").get<std::string>() != "1.0.0")
    //     throw runtime_error("Unsupported version in FullyConnected::deserialize: " + j["version"].get<std::string>());
    // if (j.at("layer_type").get<std::string>() != "fully_connected")
    //     throw runtime_error("Layer type mismatch in FullyConnected::deserialize: " + j.at("layer_type").get<std::string>());
    //
    // uint32_t numOutputFeatures = j.at("num_output_features").get<uint32_t>();
    // bool hasBias = j.at("has_bias").get<bool>();
    //
    // vector<Tensor> featureInputs;
    // for (const json &input : j["inputs"]) {
    //     uint64_t originalTensorId = input.at("id").get<uint64_t>();
    //     Tensor tensor = network->getApiTensorByOriginalId(originalTensorId);
    //     featureInputs.push_back(tensor);
    // }
    //
    // vector<Tensor> featureOutputs;
    // for (const json &output : j["outputs"]) {
    //     featureOutputs.push_back(Tensor::deserialize(output));
    // }
    //
    // FullyConnected fullyConnected = FullyConnected();
    // fullyConnected.numOutputFeatures = numOutputFeatures;
    // fullyConnected.hasBias = hasBias;
    // fullyConnected.featureInputs = featureInputs;
    // fullyConnected.standaloneFCFeatureInputs = featureInputs;
    // for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i) {
    //     fullyConnected.featureOutputs.push_back(featureOutputs[i]);
    //     fullyConnected.standaloneFCFeatureOutputs.push_back(featureOutputs[i]);
    //     fullyConnected.outputTensorFromInputTensor[fullyConnected.featureInputs[i]] = fullyConnected.featureOutputs.back();
    //     fullyConnected.inputTensorFromOutputTensor[fullyConnected.featureOutputs.back()] = fullyConnected.featureInputs[i];
    // }
    // fullyConnected.archiveReader = archiveReader;
    // if (j.contains("weights_tensor")) {
    //     fullyConnected.weightsFile = j.at("weights_tensor").get<string>();
    //     if (hasBias)
    //         fullyConnected.biasesFile = j.at("biases_tensor").get<string>();
    // }
    //
    // if (j.contains("weights_initializer")) {
    //     fullyConnected.weightsInitializer = Initializer::deserialize(j.at("weights_initializer"));
    // }
    // if (j.contains("biases_initializer")) {
    //     fullyConnected.biasesInitializer = Initializer::deserialize(j.at("biases_initializer"));
    // }
    //
    // if (j.contains("weights_optimizer")) {
    //     fullyConnected.weightsOptimizer = Optimizer::deserialize(archiveReader, j.at("weights_optimizer"), network);
    // }
    // if (j.contains("biases_optimizer")) {
    //     fullyConnected.weightsOptimizer = Optimizer::deserialize(archiveReader, j.at("biases_optimizer"), network);
    // }
    //
    // fullyConnected.initialized = true;
    // fullyConnected.addToNetwork(network);
}

vector<Event> FullyConnected::initialize(shared_ptr<ThorImplementation::TrainableLayer> physicalLayer,
                                         bool isFirstStamp,
                                         shared_ptr<ThorImplementation::TrainableLayer> sisterPhysicalLayer,
                                         Optional<Event> sisterPhysicalLayerLoadedEvent) {
    vector<Event> initDoneEvents =
        TrainableLayer::initialize(physicalLayer, isFirstStamp, sisterPhysicalLayer, sisterPhysicalLayerLoadedEvent);

    // FIXME
    //
    // // Weights are set right now, based on 1 of 3 methods:
    // // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
    // //      * So this is once per GPU since multiple stamps on the same GPU share the weights
    // // 2. Copy from a file - when loading a saved network
    // // 3. Run an initializer to set the weights - on an untrained network
    // if (!isFirstStamp) {
    //     // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
    //     assert(sisterPhysicalLayer != nullptr);
    //     ThorImplementation::Tensor weights = physicalLayer->getParameter("weights")->getStorage();
    //     Stream stream = Stream::getNextDownloadStream(weights.getPlacement().getDeviceNum());
    //     if (sisterPhysicalLayerLoadedEvent.isPresent())
    //         stream.waitEvent(sisterPhysicalLayerLoadedEvent);
    //     weights.copyFromAsync(sisterPhysicalLayer->getParameter("weights")->getStorage(), stream);
    //     if (hasBias) {
    //         ThorImplementation::Tensor biases = physicalLayer->getParameter("biases")->getStorage();
    //         Optional<ThorImplementation::Tensor> sisterLayerBiases = sisterPhysicalLayer->getParameter("biases")->getStorage();
    //         assert(sisterLayerBiases.isPresent());
    //         biases.copyFromAsync(sisterLayerBiases.get(), stream);
    //     }
    //
    //     initDoneEvents.push_back(stream.putEvent(false, true));
    // } else if (weightsFile.isPresent()) {
    //     // 2. Copy from a file - when loading a saved network
    //     assert(archiveReader != nullptr);
    //     assert(physicalLayer->getParameter("weights")->getStorage().get().getPlacement().getMemDevice() ==
    //            ThorImplementation::TensorPlacement::MemDevices::GPU);
    //     Stream stream =
    //         Stream::getNextUploadStream(physicalLayer->getParameter("weights")->getStorage().get().getPlacement().getDeviceNum());
    //
    //     ThorImplementation::Tensor weights = physicalLayer->getParameter("weights")->getStorage();
    //     archiveReader->registerReadRequest(weightsFile.get(), weights);
    //     if (hasBias) {
    //         assert(biasesFile.isPresent());
    //         ThorImplementation::Tensor biases = physicalLayer->getParameter("biases")->getStorage().get();
    //         archiveReader->registerReadRequest(biasesFile.get(), biases);
    //     }
    //
    //     // Can't use the file later, it may not still be there
    //     archiveReader = nullptr;
    //     weightsFile = Optional<string>::empty();
    //     biasesFile = Optional<string>::empty();
    // } else {
    //     // FIXME: This needs to be updated to use Parameter's. It should be moved to API Thor::TrainableLayer
    //     // // 3. Run an initializer to set the weights - on an untrained network
    //     // assert(weightsInitializer != nullptr);
    //     // if (hasBias)
    //     //     assert(biasInitializer != nullptr);
    //     //
    //     // Optional<Event> initDoneEvent;
    //     //
    //     // initDoneEvent = weightsInitializer->initialize(physicalLayer->getParameter("weights")->getStorage(), physicalLayer.get());
    //     // if (initDoneEvent.isPresent())
    //     //     initDoneEvents.push_back(initDoneEvent);
    //     //
    //     // if (physicalLayer->getParameter("biases")->getStorage().isPresent()) {
    //     //     initDoneEvent = biasInitializer->initialize(physicalLayer->getParameter("biases")->getStorage().get(),
    //     physicalLayer.get());
    //     //     if (initDoneEvent.isPresent())
    //     //         initDoneEvents.push_back(initDoneEvent);
    //     // }
    // }
    //
    // // if (hasOptimizer()) {
    // //     // Initialize the optimizer - it will follow the same process as above.
    // //     shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = physicalLayer->getOptimizer();
    // //     shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer =
    // //         sisterPhysicalLayer ? sisterPhysicalLayer->getOptimizer() : nullptr;
    // //
    // //     vector<Event> optimizerInitDoneEvents =
    // //         optimizer->initialize(physicalOptimizer, isFirstStamp, physicalSisterOptimizer, sisterPhysicalLayerLoadedEvent);
    // //     for (uint32_t i = 0; i < optimizerInitDoneEvents.size(); ++i)
    // //         initDoneEvents.push_back(optimizerInitDoneEvents[i]);
    // // }

    return initDoneEvents;
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("fully_connected", &Thor::FullyConnected::deserialize);
    return true;
}();
}  // namespace
