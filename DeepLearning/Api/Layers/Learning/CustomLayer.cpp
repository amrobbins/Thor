#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace std;
using json = nlohmann::json;

using DynamicExpression = ThorImplementation::DynamicExpression;
using DataType = ThorImplementation::DataType;
using PhysicalTensor = ThorImplementation::Tensor;
using PhysicalTensorMap = std::unordered_map<std::string, PhysicalTensor>;

namespace {

PhysicalTensor makeFakePlacedTensor(const Thor::Tensor& apiTensor, uint64_t batchSize) {
    std::vector<uint64_t> fakeDims;
    fakeDims.reserve(apiTensor.getDimensions().size() + 1);
    fakeDims.push_back(batchSize);
    for (uint64_t dim : apiTensor.getDimensions()) {
        fakeDims.push_back(dim);
    }

    ThorImplementation::TensorPlacement placement(ThorImplementation::TensorPlacement::MemDevices::CPU, 0);
    ThorImplementation::TensorDescriptor descriptor(apiTensor.getDataType(), fakeDims);
    return PhysicalTensor(placement, descriptor);
}

Thor::Tensor logicalTensorFromFakeOutput(const std::vector<uint64_t>& fakeOutputDims, DataType dtype, uint64_t expectedBatchSize) {
    if (fakeOutputDims.empty() || fakeOutputDims.front() != expectedBatchSize) {
        throw std::runtime_error("CustomLayer expression output must preserve the physical batch dimension. Expected batch " +
                                 std::to_string(expectedBatchSize) + ", got " +
                                 (fakeOutputDims.empty() ? std::string("<no batch dimension>") : std::to_string(fakeOutputDims.front())) + ".");
    }
    return Thor::Tensor(dtype, std::vector<uint64_t>(fakeOutputDims.begin() + 1, fakeOutputDims.end()));
}

bool isSymbolicShapeField(const std::string& key) {
    return key == "reshape_dims" || key == "view_dims";
}

bool generalizeBatchDependentJson(json& result,
                                  const json& batchOne,
                                  const json& batchTwo,
                                  const std::string& path,
                                  std::string& rejectionReason) {
    if (batchOne.type() != batchTwo.type()) {
        rejectionReason = path + " changed JSON type between batch-1 and batch-2 builds.";
        return false;
    }

    if (batchOne.is_object()) {
        if (batchOne.size() != batchTwo.size()) {
            rejectionReason = path + " changed object fields between batch-1 and batch-2 builds.";
            return false;
        }
        for (auto it = batchOne.begin(); it != batchOne.end(); ++it) {
            if (!batchTwo.contains(it.key())) {
                rejectionReason = path + " lost field '" + it.key() + "' in the batch-2 build.";
                return false;
            }
            if (it.key() == "canonical_hash") {
                result[it.key()] = "";
                continue;
            }
            if (isSymbolicShapeField(it.key()) && it.value().is_array()) {
                const json& oneArray = it.value();
                const json& twoArray = batchTwo.at(it.key());
                if (!twoArray.is_array() || oneArray.size() != twoArray.size()) {
                    rejectionReason = path + "." + it.key() + " changed rank between batch-1 and batch-2 builds.";
                    return false;
                }

                std::vector<size_t> changedIndices;
                for (size_t i = 0; i < oneArray.size(); ++i) {
                    if (oneArray[i] != twoArray[i])
                        changedIndices.push_back(i);
                }

                result[it.key()] = oneArray;
                if (changedIndices.empty())
                    continue;

                if (changedIndices.size() == 1) {
                    const size_t changedIndex = changedIndices.front();
                    const json& oneDimension = oneArray[changedIndex];
                    const json& twoDimension = twoArray[changedIndex];
                    if (oneDimension.is_number_unsigned() && twoDimension.is_number_unsigned()) {
                        if (it.key() == "reshape_dims") {
                            // COPY_DIM is relative to the reshape's immediate source axis, which may already be flattened.
                            // A varying reshape dimension is instead reconstructed from source numel and the remaining fixed
                            // dimensions. This handles both [batch, sequence, hidden] -> [batch * sequence, hidden] and the
                            // inverse reshape without treating the synthetic batch value as a literal.
                            // FusedEquation validation below recompiles this generalized definition against both probes, so
                            // this rewrite is accepted only when it preserves the original shape for batch 1 and batch 2.
                            result[it.key()][changedIndex] = std::numeric_limits<uint64_t>::max();
                            continue;
                        }
                        if (changedIndex == 0 && oneDimension.get<uint64_t>() == 1 && twoDimension.get<uint64_t>() == 2) {
                            // Expression COPY_DIM is the internal symbolic reference to the source dimension at this axis.
                            // This is valid for a strided view preserving CustomLayer's leading physical batch dimension.
                            result[it.key()][changedIndex] = 0;
                            continue;
                        }
                    }
                }

                const size_t firstChangedIndex = changedIndices.front();
                rejectionReason = path + "." + it.key() + "[" + std::to_string(firstChangedIndex) +
                                  "] changed in a way Thor cannot serialize symbolically. A strided view may only copy the "
                                  "leading placement-batch dimension automatically, and a reshape may additionally infer one "
                                  "batch-derived dimension from its source numel.";
                return false;
            }
            if (!generalizeBatchDependentJson(result[it.key()], it.value(), batchTwo.at(it.key()), path + "." + it.key(), rejectionReason))
                return false;
        }
        return true;
    }

    if (batchOne.is_array()) {
        if (batchOne.size() != batchTwo.size()) {
            rejectionReason = path + " changed array length between batch-1 and batch-2 builds.";
            return false;
        }
        for (size_t i = 0; i < batchOne.size(); ++i) {
            if (!generalizeBatchDependentJson(result[i], batchOne[i], batchTwo[i], path + "[" + std::to_string(i) + "]", rejectionReason))
                return false;
        }
        return true;
    }

    if (batchOne != batchTwo) {
        rejectionReason = path + " captured a concrete batch-dependent value (batch-1=" + batchOne.dump() +
                          ", batch-2=" + batchTwo.dump() + ").";
        return false;
    }
    return true;
}

}  // namespace

namespace Thor {

CustomLayer::CustomLayer(DynamicExpression expr,
                         const std::vector<TensorMap>& inputInterfaces,
                         std::vector<std::shared_ptr<ParameterSpecification>> parameters)
    : CustomLayer(std::move(expr), {}, {}, inputInterfaces, {}, std::move(parameters)) {}

CustomLayer::CustomLayer(DynamicExpression expr,
                         std::vector<std::string> inputNames,
                         std::vector<std::string> outputNames,
                         const std::vector<TensorMap>& inputInterfaces,
                         const std::vector<TensorMap>& outputInterfaces,
                         std::vector<std::shared_ptr<ParameterSpecification>> parameters)
    : CustomLayer(std::move(expr),
                  std::move(inputNames),
                  std::move(outputNames),
                  inputInterfaces,
                  outputInterfaces,
                  std::move(parameters),
                  SerializationContract::REQUIRE_EXPRESSION_DEFINITION) {}

CustomLayer::CustomLayer(DynamicExpression expr,
                         std::vector<std::string> inputNames,
                         std::vector<std::string> outputNames,
                         const std::vector<TensorMap>& inputInterfaces,
                         const std::vector<TensorMap>& outputInterfaces,
                         std::vector<std::shared_ptr<ParameterSpecification>> parameters,
                         SerializationContract serializationContract)
    : TrainableLayer(std::move(parameters)), expr(std::move(expr)) {
    if (inputNames.empty())
        inputNames = this->expr.getExpectedInputNames();
    if (outputNames.empty())
        outputNames = this->expr.getExpectedOutputNames();

    this->inputNames = std::move(inputNames);
    this->outputNames = std::move(outputNames);

    if (inputInterfaces.empty())
        throw runtime_error("Cannot create a CustomLayer with zero input interfaces.");

    validateInputInterfacesMatchExpression();
    validateOutputInterfacesMatchExpression();
    validateParameterNames();
    validateExpressionCanBindFeatureAndParameterInputs();

    for (const TensorMap& inputInterface : inputInterfaces) {
        validateTensorInterface(inputInterface, "input");
        validateInterfaceNames(inputInterface, this->inputNames, "input");
    }

    // Ensure no two interfaces are exactly the same. Aliasing a tensor across otherwise-distinct interfaces is allowed,
    // but exact duplicate interfaces make getOutputInterface(inputInterface) ambiguous.
    for (uint32_t i = 0; i < inputInterfaces.size() - 1; ++i) {
        for (uint32_t j = i + 1; j < inputInterfaces.size(); ++j) {
            if (interfaceMatches(inputInterfaces[i], inputInterfaces[j])) {
                throw runtime_error("CustomLayer: An input interface was connected more than once.");
            }
        }
    }

    // Ensure shape and dtype equivalence between all corresponding tensors across all interfaces.
    const TensorMap& referenceInterface = inputInterfaces.front();
    for (const std::string& name : this->inputNames) {
        const Tensor& referenceTensor = referenceInterface.at(name);
        const auto referenceDataType = referenceTensor.getDataType();
        const auto referenceDimensions = referenceTensor.getDimensions();

        for (uint32_t interfaceIndex = 1; interfaceIndex < inputInterfaces.size(); ++interfaceIndex) {
            const Tensor& tensor = inputInterfaces[interfaceIndex].at(name);

            if (tensor.getDataType() != referenceDataType || tensor.getDimensions() != referenceDimensions) {
                std::ostringstream oss;
                oss << "CustomLayer input tensor '" << name << "' must have the same shape and dtype across all "
                    << "input interfaces. Interface 0 has " << referenceTensor.getDescriptorString() << ", but interface " << interfaceIndex
                    << " has " << tensor.getDescriptorString() << ".";
                throw runtime_error(oss.str());
            }
        }
    }

    assignInputInterfaces(inputInterfaces);
    if (outputInterfaces.empty()) {
        materializeOutputInterfacesFromInputInterfaces();
    } else {
        // Explicit logical output descriptors decouple API output-shape construction from physical expression construction.
        // The physical builder is still probed below before the layer is accepted, so serialization failures cannot surface after training.
        assignOutputInterfaces(outputInterfaces);
    }

    const TensorMap& referenceOutputInterface = this->outputInterfaces.front();
    for (const std::string& name : this->outputNames) {
        const Tensor& referenceTensor = referenceOutputInterface.at(name);
        for (uint32_t interfaceIndex = 1; interfaceIndex < this->outputInterfaces.size(); ++interfaceIndex) {
            const Tensor& tensor = this->outputInterfaces[interfaceIndex].at(name);
            if (tensor.getDataType() != referenceTensor.getDataType() || tensor.getDimensions() != referenceTensor.getDimensions()) {
                std::ostringstream oss;
                oss << "CustomLayer output tensor '" << name << "' must have the same logical shape and dtype across all output "
                    << "interfaces. Interface 0 has " << referenceTensor.getDescriptorString() << ", but interface " << interfaceIndex
                    << " has " << tensor.getDescriptorString() << ".";
                throw runtime_error(oss.str());
            }
        }
    }

    if (serializationContract == SerializationContract::REQUIRE_EXPRESSION_DEFINITION) {
        analyzeSerializableExpression(referenceInterface);
        if (serializableExpressionDefinition == nullptr) {
            throw runtime_error(
                "CustomLayer construction rejected because Thor could not derive a verified batch-polymorphic symbolic "
                "ExpressionDefinition. A CustomLayer must be serializable before it can be added to a model. " +
                (serializationRejectionReason.empty() ? std::string("No serializable expression definition was supplied.")
                                                      : serializationRejectionReason) +
                " Use a native declarative layer, or return an explicitly serializable symbolic expression definition.");
        }
    }

    initialized = true;
}

std::string CustomLayer::joinNames(const std::set<std::string>& names) {
    if (names.empty())
        return "<none>";

    std::ostringstream oss;
    bool first = true;
    for (const auto& name : names) {
        if (!first)
            oss << ", ";
        oss << name;
        first = false;
    }
    return oss.str();
}

uint32_t CustomLayer::encodeInputConnection(uint32_t interfaceIndex, uint32_t inputPortIndex) const {
    return interfaceIndex * static_cast<uint32_t>(inputNames.size()) + inputPortIndex;
}

uint32_t CustomLayer::encodeOutputConnection(uint32_t interfaceIndex, uint32_t outputPortIndex) const {
    return interfaceIndex * static_cast<uint32_t>(outputNames.size()) + outputPortIndex;
}

void CustomLayer::validateTensorInterface(const TensorMap& tensorInterface, const std::string& what) {
    if (tensorInterface.empty()) {
        throw runtime_error("CustomLayer requires at least one tensor in each " + what + " interface.");
    }

    for (const auto& [name, tensor] : tensorInterface) {
        if (name.empty()) {
            throw runtime_error("CustomLayer " + what + " name cannot be empty.");
        }
        if (!tensor.isInitialized()) {
            throw runtime_error("CustomLayer " + what + " tensor for name '" + name + "' is not initialized.");
        }
    }
}

void CustomLayer::validateInterfaceNames(const TensorMap& tensorInterface,
                                         const std::vector<std::string>& expectedNames,
                                         const std::string& what) {
    std::set<std::string> actualNames;
    for (const auto& [name, tensor] : tensorInterface) {
        (void)tensor;
        actualNames.insert(name);
    }

    std::set<std::string> expectedNameSet(expectedNames.begin(), expectedNames.end());
    if (actualNames != expectedNameSet) {
        throw runtime_error("CustomLayer " + what + " interface name mismatch. Expected {" + joinNames(expectedNameSet) + "}, got {" +
                            joinNames(actualNames) + "}.");
    }
}

void CustomLayer::validateInputInterfacesMatchExpression() const {
    if (inputNames.empty()) {
        throw runtime_error("CustomLayer must declare at least one feature input name.");
    }
}

void CustomLayer::validateOutputInterfacesMatchExpression() const {
    if (outputNames.empty()) {
        throw runtime_error("CustomLayer must declare at least one output name.");
    }
}

void CustomLayer::validateParameterNames() const {
    std::set<std::string> seen;
    std::set<std::string> featureInputNames(inputNames.begin(), inputNames.end());
    for (const auto& parameter : parameters) {
        if (parameter == nullptr) {
            throw runtime_error("CustomLayer contains a null ParameterSpecification.");
        }

        const std::string& name = parameter->getName();
        if (!seen.insert(name).second) {
            throw runtime_error("CustomLayer received duplicate Parameter name '" + name + "'.");
        }
        if (featureInputNames.contains(name)) {
            throw runtime_error("CustomLayer Parameter name '" + name + "' conflicts with a feature input name.");
        }
    }
}

void CustomLayer::validateExpressionCanBindFeatureAndParameterInputs() const {
    const std::vector<std::string>& expectedInputNames = expr.getExpectedInputNames();
    if (expectedInputNames.empty())
        return;

    std::set<std::string> expected(expectedInputNames.begin(), expectedInputNames.end());
    for (const std::string& name : inputNames) {
        if (!expected.contains(name)) {
            throw runtime_error("CustomLayer expression expected input names do not include feature input '" + name + "'.");
        }
    }
    for (const auto& parameter : parameters) {
        if (parameter == nullptr)
            throw runtime_error("CustomLayer contains a null ParameterSpecification.");
        const std::string& name = parameter->getName();
        if (!expected.contains(name)) {
            throw runtime_error("CustomLayer expression expected input names do not include parameter '" + name + "'.");
        }
    }

    const std::vector<std::string>& expectedOutputNames = expr.getExpectedOutputNames();
    if (!expectedOutputNames.empty()) {
        std::set<std::string> expectedOutputs(expectedOutputNames.begin(), expectedOutputNames.end());
        for (const std::string& name : outputNames) {
            if (!expectedOutputs.contains(name)) {
                throw runtime_error("CustomLayer expression expected output names do not include output '" + name + "'.");
            }
        }
    }
}

void CustomLayer::assignInputInterfaces(const std::vector<TensorMap>& inputInterfaces) {
    if (inputInterfaces.empty()) {
        throw runtime_error("CustomLayer requires at least one input interface.");
    }

    featureInputs.clear();
    this->inputInterfaces.clear();
    connectedInputPortIndicesByInterface.clear();
    emittedOutputInterface.clear();
    inputBindingsByTensorOriginalId.clear();
    nextInputBindingConnectionCursorByTensorOriginalId.clear();

    for (uint32_t interfaceIndex = 0; interfaceIndex < inputInterfaces.size(); ++interfaceIndex) {
        const TensorMap& inputInterface = inputInterfaces[interfaceIndex];
        validateTensorInterface(inputInterface, "input");
        validateInterfaceNames(inputInterface, inputNames, "input");

        this->inputInterfaces.push_back(inputInterface);
        connectedInputPortIndicesByInterface.emplace_back();
        emittedOutputInterface.push_back(false);

        for (uint32_t inputPortIndex = 0; inputPortIndex < inputNames.size(); ++inputPortIndex) {
            const std::string& name = inputNames[inputPortIndex];
            const Tensor& tensor = inputInterface.at(name);

            featureInputs.push_back(tensor);
            inputBindingsByTensorOriginalId[tensor.getOriginalId()].push_back(InputBinding{interfaceIndex, inputPortIndex, name});
        }
    }
}

CustomLayer::SerializationProbe CustomLayer::buildExpressionForBatch(const TensorMap& inputInterface,
                                                                       uint64_t batchSize,
                                                                       const TensorMap* outputInterface) const {
    PhysicalTensorMap fakeFeatureInputs;
    for (const auto& [name, apiTensor] : inputInterface) {
        fakeFeatureInputs.emplace(name, makeFakePlacedTensor(apiTensor, batchSize));
    }

    ThorImplementation::PhysicalParameter::StorageContext fakeStorageContext(fakeFeatureInputs);
    PhysicalTensorMap fakeAllInputs = fakeFeatureInputs;
    for (const auto& apiParameter : parameters) {
        std::shared_ptr<ThorImplementation::PhysicalParameter> physicalParameter = apiParameter->stamp();
        physicalParameter->compileStorage(fakeStorageContext);
        std::optional<PhysicalTensor> storage = physicalParameter->getStorage();
        if (!storage.has_value()) {
            throw std::runtime_error("CustomLayer failed to infer parameter storage for '" + apiParameter->getName() + "'.");
        }
        const bool inserted = fakeAllInputs.emplace(apiParameter->getName(), storage.value()).second;
        if (!inserted) {
            throw std::runtime_error("CustomLayer Parameter name '" + apiParameter->getName() +
                                     "' conflicts with a feature input name.");
        }
    }

    PhysicalTensorMap fakeOutputs;
    if (outputInterface != nullptr) {
        for (const auto& [name, apiTensor] : *outputInterface) {
            fakeOutputs.emplace(name, makeFakePlacedTensor(apiTensor, batchSize));
        }
    }

    Stream fakeStream(0, Stream::Priority::REGULAR);
    ThorImplementation::DynamicExpressionBuild build = expr.build(fakeAllInputs, fakeOutputs, fakeStream);
    return SerializationProbe{std::move(build), std::move(fakeAllInputs), std::move(fakeOutputs)};
}

CustomLayer::TensorMap CustomLayer::inferOutputInterfaceFromInputInterface(const TensorMap& inputInterface) {
    const SerializationProbe batchOne = buildExpressionForBatch(inputInterface, 1, nullptr);
    const SerializationProbe batchTwo = buildExpressionForBatch(inputInterface, 2, nullptr);

    const auto batchOneShapes =
        batchOne.build.equation->getOutputShapes(batchOne.build.stamp_inputs, batchOne.build.tensor_scalar_inputs);
    const auto batchTwoShapes =
        batchTwo.build.equation->getOutputShapes(batchTwo.build.stamp_inputs, batchTwo.build.tensor_scalar_inputs);
    const auto batchOneDTypes = batchOne.build.equation->getOutputDataTypes(batchOne.build.stamp_inputs);
    const auto batchTwoDTypes = batchTwo.build.equation->getOutputDataTypes(batchTwo.build.stamp_inputs);

    TensorMap inferredOutputs;
    for (const std::string& outputName : outputNames) {
        auto shapeOneIt = batchOneShapes.find(outputName);
        auto shapeTwoIt = batchTwoShapes.find(outputName);
        auto dtypeOneIt = batchOneDTypes.find(outputName);
        auto dtypeTwoIt = batchTwoDTypes.find(outputName);
        if (shapeOneIt == batchOneShapes.end() || shapeTwoIt == batchTwoShapes.end()) {
            throw std::runtime_error("CustomLayer failed to infer output shape for '" + outputName + "'.");
        }
        if (dtypeOneIt == batchOneDTypes.end() || dtypeTwoIt == batchTwoDTypes.end()) {
            throw std::runtime_error("CustomLayer failed to infer output dtype for '" + outputName + "'.");
        }
        if (dtypeOneIt->second != dtypeTwoIt->second) {
            throw std::runtime_error("CustomLayer output dtype for '" + outputName +
                                     "' changed between batch-1 and batch-2 shape inference.");
        }

        Thor::Tensor one = logicalTensorFromFakeOutput(shapeOneIt->second, dtypeOneIt->second, 1);
        Thor::Tensor two = logicalTensorFromFakeOutput(shapeTwoIt->second, dtypeTwoIt->second, 2);
        if (one.getDimensions() != two.getDimensions()) {
            throw std::runtime_error("CustomLayer logical output shape for '" + outputName +
                                     "' depends on the synthetic physical batch size. Batch-1 inferred " +
                                     one.getDescriptorString() + ", batch-2 inferred " + two.getDescriptorString() + ".");
        }
        inferredOutputs.emplace(outputName, std::move(one));
    }

    return inferredOutputs;
}

void CustomLayer::analyzeSerializableExpression(const TensorMap& inputInterface) const {
    serializationAnalysisPerformed = true;
    serializableExpressionDefinition.reset();
    serializationRejectionReason.clear();

    try {
        THOR_THROW_IF_FALSE(!outputInterfaces.empty());
        const TensorMap& outputInterface = outputInterfaces.front();
        const SerializationProbe batchOne = buildExpressionForBatch(inputInterface, 1, &outputInterface);
        const SerializationProbe batchTwo = buildExpressionForBatch(inputInterface, 2, &outputInterface);
        analyzeSerializableExpression(batchOne, batchTwo);
    } catch (const std::exception& error) {
        serializableExpressionDefinition.reset();
        serializationRejectionReason = "batch-polymorphism probe failed: " + std::string(error.what());
    }
}

void CustomLayer::analyzeSerializableExpression(const SerializationProbe& batchOne,
                                                const SerializationProbe& batchTwo) const {
    serializationAnalysisPerformed = true;
    serializableExpressionDefinition.reset();
    serializationRejectionReason.clear();

    auto validatePureSerializableBuild = [&](const SerializationProbe& probe, uint64_t batchSize) -> bool {
        const ThorImplementation::DynamicExpressionBuild& build = probe.build;
        if (!build.tensor_scalar_inputs.empty()) {
            serializationRejectionReason =
                "the batch-" + std::to_string(batchSize) +
                " builder uses tensor-backed runtime scalars, which an ExpressionDefinition alone cannot reconstruct";
            return false;
        }
        if (!build.requested_output_shapes.empty()) {
            serializationRejectionReason =
                "the batch-" + std::to_string(batchSize) +
                " builder supplies runtime output-shape overrides, which an ExpressionDefinition alone cannot reconstruct";
            return false;
        }
        if (build.pre_forward_hook) {
            serializationRejectionReason =
                "the batch-" + std::to_string(batchSize) +
                " builder installs a pre-forward callback, which cannot be serialized as an ExpressionDefinition";
            return false;
        }
        for (const auto& [name, actual] : build.stamp_inputs) {
            auto source = probe.sourceInputs.find(name);
            if (source == probe.sourceInputs.end() || actual != source->second) {
                serializationRejectionReason =
                    "the batch-" + std::to_string(batchSize) + " builder remaps or synthesizes tensor input '" + name +
                    "'; saved CustomLayer expressions must bind declared inputs directly";
                return false;
            }
        }
        for (const auto& [name, actual] : build.preallocated_outputs) {
            auto source = probe.sourceOutputs.find(name);
            if (source == probe.sourceOutputs.end() || actual != source->second) {
                serializationRejectionReason =
                    "the batch-" + std::to_string(batchSize) + " builder remaps or synthesizes preallocated output '" + name +
                    "'; saved CustomLayer expressions must bind caller-provided outputs directly";
                return false;
            }
        }

        const auto actualShapes = build.equation->getOutputShapes(build.stamp_inputs, build.tensor_scalar_inputs);
        const auto actualDTypes = build.equation->getOutputDataTypes(build.stamp_inputs);
        const TensorMap& declaredOutputs = outputInterfaces.front();
        for (const std::string& outputName : outputNames) {
            auto declared = declaredOutputs.find(outputName);
            auto actualShape = actualShapes.find(outputName);
            auto actualDType = actualDTypes.find(outputName);
            if (declared == declaredOutputs.end() || actualShape == actualShapes.end() || actualDType == actualDTypes.end()) {
                serializationRejectionReason =
                    "the batch-" + std::to_string(batchSize) + " builder did not produce declared output '" + outputName + "'";
                return false;
            }
            std::vector<uint64_t> expectedDimensions{batchSize};
            const std::vector<uint64_t>& logicalDimensions = declared->second.getDimensions();
            expectedDimensions.insert(expectedDimensions.end(), logicalDimensions.begin(), logicalDimensions.end());
            if (actualShape->second != expectedDimensions || actualDType->second != declared->second.getDataType()) {
                serializationRejectionReason =
                    "the batch-" + std::to_string(batchSize) + " builder output '" + outputName +
                    "' does not match its logical TensorSpec after adding the placement batch dimension";
                return false;
            }
        }
        return true;
    };

    if (!validatePureSerializableBuild(batchOne, 1) || !validatePureSerializableBuild(batchTwo, 2))
        return;

    if (batchOne.build.serialized_definition == nullptr || batchTwo.build.serialized_definition == nullptr) {
        serializationRejectionReason =
            "the builder did not provide an ExpressionDefinition for both batch-polymorphism probes";
        return;
    }

    auto validateDefinitionAgainstProbes = [&](const ThorImplementation::ExpressionDefinition& definition) {
        auto validateProbe = [&](const SerializationProbe& probe, uint64_t batchSize) {
            const ThorImplementation::DynamicExpressionBuild& original = probe.build;
            ThorImplementation::FusedEquation serializedEquation = ThorImplementation::FusedEquation::compile(definition.outputs, 0);
            const auto serializedShapes = serializedEquation.getOutputShapes(original.stamp_inputs, original.tensor_scalar_inputs);
            const auto originalShapes = original.equation->getOutputShapes(original.stamp_inputs, original.tensor_scalar_inputs);
            const auto serializedDTypes = serializedEquation.getOutputDataTypes(original.stamp_inputs);
            const auto originalDTypes = original.equation->getOutputDataTypes(original.stamp_inputs);
            for (const std::string& outputName : outputNames) {
                if (!serializedShapes.contains(outputName) || !originalShapes.contains(outputName) ||
                    serializedShapes.at(outputName) != originalShapes.at(outputName) ||
                    !serializedDTypes.contains(outputName) || !originalDTypes.contains(outputName) ||
                    serializedDTypes.at(outputName) != originalDTypes.at(outputName)) {
                    throw std::runtime_error("serialized expression changed output '" + outputName +
                                             "' for batch " + std::to_string(batchSize));
                }
            }
        };
        validateProbe(batchOne, 1);
        validateProbe(batchTwo, 2);
    };

    const json batchOneJson = batchOne.build.serialized_definition->architectureJson();
    const json batchTwoJson = batchTwo.build.serialized_definition->architectureJson();
    if (batchOneJson == batchTwoJson) {
        try {
            ThorImplementation::ExpressionDefinition definition = *batchOne.build.serialized_definition;
            definition.validate();
            validateDefinitionAgainstProbes(definition);
            serializableExpressionDefinition =
                std::make_shared<ThorImplementation::ExpressionDefinition>(std::move(definition));
        } catch (const std::exception& error) {
            serializationRejectionReason = error.what();
        }
        return;
    }
    if ((batchOneJson.contains("cuda_kernels") && !batchOneJson.at("cuda_kernels").empty()) ||
        (batchTwoJson.contains("cuda_kernels") && !batchTwoJson.at("cuda_kernels").empty())) {
        serializationRejectionReason =
            "batch-dependent CUDA-kernel expressions require an explicitly supplied symbolic ExpressionDefinition; "
            "Thor will not rewrite a signed CUDA expression during batch-polymorphism analysis";
        return;
    }

    json generalized = batchOneJson;
    std::string rejectionReason;
    if (!generalizeBatchDependentJson(generalized, batchOneJson, batchTwoJson, "expression", rejectionReason)) {
        serializationRejectionReason = std::move(rejectionReason);
        return;
    }

    try {
        ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::deserialize(generalized);
        definition.validate();
        validateDefinitionAgainstProbes(definition);
        serializableExpressionDefinition = std::make_shared<ThorImplementation::ExpressionDefinition>(std::move(definition));
    } catch (const std::exception& error) {
        serializationRejectionReason = error.what();
    }
}

void CustomLayer::materializeOutputInterfacesFromInputInterfaces() {
    TensorMap inferredOutputs = inferOutputInterfaceFromInputInterface(inputInterfaces.front());

    std::vector<TensorMap> materialized;
    materialized.reserve(inputInterfaces.size());
    for (size_t interfaceIndex = 0; interfaceIndex < inputInterfaces.size(); ++interfaceIndex) {
        (void)interfaceIndex;
        TensorMap outputInterface;
        for (const std::string& outputName : outputNames) {
            auto it = inferredOutputs.find(outputName);
            if (it == inferredOutputs.end()) {
                throw std::runtime_error("CustomLayer failed to infer output tensor for '" + outputName + "'.");
            }
            outputInterface[outputName] = it->second.clone();
        }
        materialized.push_back(std::move(outputInterface));
    }

    assignOutputInterfaces(materialized);
}

void CustomLayer::assignOutputInterfaces(const std::vector<TensorMap>& outputInterfaces) {
    featureOutputs.clear();
    this->outputInterfaces.clear();

    for (const TensorMap& outputInterface : outputInterfaces) {
        validateTensorInterface(outputInterface, "output");
        validateInterfaceNames(outputInterface, outputNames, "output");
        this->outputInterfaces.push_back(outputInterface);

        for (const std::string& name : outputNames) {
            featureOutputs.push_back(outputInterface.at(name));
        }
    }
}

bool CustomLayer::interfaceMatches(const TensorMap& subset, const TensorMap& superset) {
    if (subset.size() != superset.size())
        return false;

    for (const auto& [name, tensor] : subset) {
        const auto& found = superset.find(name);
        if (found == superset.end() || found->second != tensor)
            return false;
    }
    return true;
}

CustomLayer::TensorMap CustomLayer::getInputInterface(uint32_t interfaceIndex) const {
    if (interfaceIndex >= inputInterfaces.size()) {
        throw runtime_error("CustomLayer input interface index out of range.");
    }
    return inputInterfaces[interfaceIndex];
}

CustomLayer::TensorMap CustomLayer::getOutputInterfaceByIndex(uint32_t interfaceIndex) const {
    if (interfaceIndex >= outputInterfaces.size()) {
        throw runtime_error("CustomLayer output interface index out of range.");
    }
    return outputInterfaces[interfaceIndex];
}

Tensor CustomLayer::getOutput(const std::string& outputName, uint32_t interfaceIndex) const {
    if (interfaceIndex >= outputInterfaces.size()) {
        throw runtime_error("CustomLayer output interface index out of range.");
    }

    const TensorMap& outputInterface = outputInterfaces[interfaceIndex];
    auto found = outputInterface.find(outputName);
    if (found == outputInterface.end()) {
        throw runtime_error("CustomLayer has no output named '" + outputName + "'.");
    }
    return found->second;
}

CustomLayer::TensorMap CustomLayer::getOutputInterface(const TensorMap& inputInterface) const {
    validateInterfaceNames(inputInterface, inputNames, "input");

    bool foundMatch = false;
    uint32_t matchedIndex = 0;
    for (uint32_t i = 0; i < inputInterfaces.size(); ++i) {
        if (!interfaceMatches(inputInterfaces[i], inputInterface)) {
            continue;
        }
        if (foundMatch) {
            throw runtime_error("Cannot get output interface because the input interface matches more than one CustomLayer interface.");
        }
        foundMatch = true;
        matchedIndex = i;
    }

    if (!foundMatch) {
        throw runtime_error(
            "Cannot get output interface from the inputInterface that was sent,"
            " because the input interface that was sent is not in the list of connected input interfaces.");
    }

    if (matchedIndex >= outputInterfaces.size()) {
        throw runtime_error("CustomLayer output interface has not been materialized yet for the requested input interface.");
    }

    return outputInterfaces[matchedIndex];
}

int CustomLayer::getConnectionType(Tensor connectingTensor) const {
    const uint64_t originalId = connectingTensor.getOriginalId();
    auto inputIt = inputBindingsByTensorOriginalId.find(originalId);
    if (inputIt != inputBindingsByTensorOriginalId.end()) {
        const std::vector<InputBinding>& bindings = inputIt->second;
        THOR_THROW_IF_FALSE(!bindings.empty());

        uint32_t& cursor = nextInputBindingConnectionCursorByTensorOriginalId[originalId];
        const InputBinding& binding = bindings[cursor % bindings.size()];
        ++cursor;
        return static_cast<int>(encodeInputConnection(binding.interfaceIndex, binding.inputPortIndex));
    }

    for (uint32_t interfaceIndex = 0; interfaceIndex < outputInterfaces.size(); ++interfaceIndex) {
        for (uint32_t outputPortIndex = 0; outputPortIndex < outputNames.size(); ++outputPortIndex) {
            const std::string& name = outputNames[outputPortIndex];
            if (outputInterfaces[interfaceIndex].at(name) == connectingTensor) {
                return static_cast<int>(encodeOutputConnection(interfaceIndex, outputPortIndex));
            }
        }
    }

    throw runtime_error("Tensor is not connected to this CustomLayer.");
}

void CustomLayer::informThatInputConnectionMade(Tensor inputTensor) {
    auto it = inputBindingsByTensorOriginalId.find(inputTensor.getOriginalId());
    if (it == inputBindingsByTensorOriginalId.end()) {
        throw runtime_error("CustomLayer informed of connection for unknown input tensor.");
    }

    for (const InputBinding& binding : it->second) {
        THOR_THROW_IF_FALSE(binding.interfaceIndex < connectedInputPortIndicesByInterface.size());
        connectedInputPortIndicesByInterface[binding.interfaceIndex].insert(binding.inputPortIndex);
    }
}

void CustomLayer::resetGraphTraversalState() {
    for (std::set<uint32_t>& connectedInputPortIndices : connectedInputPortIndicesByInterface) {
        connectedInputPortIndices.clear();
    }
    std::fill(emittedOutputInterface.begin(), emittedOutputInterface.end(), false);
    nextInputBindingConnectionCursorByTensorOriginalId.clear();
}

std::vector<Tensor> CustomLayer::getOutputsFromInput(Tensor inputTensor) {
    auto it = inputBindingsByTensorOriginalId.find(inputTensor.getOriginalId());
    if (it == inputBindingsByTensorOriginalId.end()) {
        throw runtime_error("CustomLayer asked for outputs from unknown input tensor.");
    }

    std::vector<Tensor> outputs;

    std::set<uint32_t> candidateInterfaces;
    for (const InputBinding& binding : it->second) {
        candidateInterfaces.insert(binding.interfaceIndex);
    }

    for (uint32_t interfaceIndex : candidateInterfaces) {
        if (interfaceIndex >= connectedInputPortIndicesByInterface.size()) {
            continue;
        }
        if (emittedOutputInterface[interfaceIndex]) {
            continue;
        }
        if (connectedInputPortIndicesByInterface[interfaceIndex].size() != inputNames.size()) {
            continue;
        }
        if (interfaceIndex >= outputInterfaces.size()) {
            continue;
        }

        emittedOutputInterface[interfaceIndex] = true;
        outputs.reserve(outputs.size() + outputNames.size());
        for (const std::string& name : outputNames) {
            outputs.push_back(outputInterfaces[interfaceIndex].at(name));
        }
    }

    return outputs;
}

uint64_t CustomLayer::getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;

    uint64_t totalBytes = 0;
    std::set<uint64_t> countedInputOriginalIds;
    for (const Tensor& tensor : featureInputs) {
        if (countedInputOriginalIds.insert(tensor.getOriginalId()).second) {
            totalBytes += tensor.getTotalSizeInBytes();
        }
    }
    for (const Tensor& tensor : featureOutputs)
        totalBytes += tensor.getTotalSizeInBytes();
    return totalBytes * std::max<uint32_t>(1, batchSize);
}

std::shared_ptr<ThorImplementation::Layer> CustomLayer::stamp(ThorImplementation::TensorPlacement placement,
                                                              std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                              std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                              Thor::Tensor connectingApiTensor,
                                                              const bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;

    if (!inputBindingsByTensorOriginalId.contains(connectingApiTensor.getOriginalId())) {
        throw runtime_error("CustomLayer::stamp called with a tensor that is not one of its declared inputs.");
    }

    std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>> physicalParameters;
    physicalParameters.reserve(parameters.size());
    for (const auto& parameter : parameters) {
        if (parameter == nullptr)
            throw runtime_error("CustomLayer contains a null Parameter.");
        physicalParameters.push_back(parameter->stamp());
    }

    std::vector<ThorImplementation::CustomLayer::DeclaredOutputDescriptor> declaredOutputDescriptors;
    declaredOutputDescriptors.reserve(outputNames.size());
    const TensorMap& outputInterface = outputInterfaces.front();
    for (const std::string& outputName : outputNames) {
        const Tensor& outputTensor = outputInterface.at(outputName);
        declaredOutputDescriptors.push_back(
            ThorImplementation::CustomLayer::DeclaredOutputDescriptor{outputTensor.getDataType(), outputTensor.getDimensions()});
    }

    auto physicalLayer = std::make_shared<ThorImplementation::CustomLayer>(
        expr,
        inputNames,
        outputNames,
        placement,
        physicalParameters,
        inferenceOnly,
        Layer::getId(),
        std::move(declaredOutputDescriptors));
    physicalLayer->setLayerName(getLayerType());
    return physicalLayer;
}

json CustomLayer::architectureJson() const {
    if (!serializationAnalysisPerformed || serializableExpressionDefinition == nullptr) {
        throw logic_error(
            "CustomLayer serialization invariant violated: generic CustomLayer instances must have a verified symbolic "
            "ExpressionDefinition before they are added to a model.");
    }

    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = "1.0.0";
    j["layer_type"] = "custom_layer";
    j["input_names"] = inputNames;
    j["output_names"] = outputNames;
    j["input_interfaces"] = json::array();
    j["output_interfaces"] = json::array();
    j["parameters"] = json::object();

    for (const TensorMap& inputInterface : inputInterfaces) {
        json interfaceJson;
        for (const std::string& name : inputNames) {
            interfaceJson[name] = inputInterface.at(name).architectureJson();
        }
        j["input_interfaces"].push_back(interfaceJson);
    }

    for (const TensorMap& outputInterface : outputInterfaces) {
        json interfaceJson;
        for (const std::string& name : outputNames) {
            interfaceJson[name] = outputInterface.at(name).architectureJson();
        }
        j["output_interfaces"].push_back(interfaceJson);
    }

    for (const auto& parameter : parameters) {
        if (parameter != nullptr)
            j["parameters"][parameter->getName()] = parameter->architectureJson();
    }

    j["expression"] = serializableExpressionDefinition->architectureJson();

    return j;
}

void CustomLayer::deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in CustomLayer::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "custom_layer")
        throw runtime_error("Layer type mismatch in CustomLayer::deserialize: " + j.at("layer_type").get<std::string>());

    std::vector<std::string> inputNames = j.at("input_names").get<std::vector<std::string>>();
    std::vector<std::string> outputNames = j.at("output_names").get<std::vector<std::string>>();
    ThorImplementation::ExpressionDefinition expressionDefinition = ThorImplementation::ExpressionDefinition::deserialize(
        j.at("expression"),
        network != nullptr && network->allowUnsafeLoadedCudaKernelSourceCompilation(),
        network != nullptr ? network->trustedLoadedCudaKernelPublicKey() : std::string{},
        network != nullptr ? network->trustedLoadedCudaKernelSourceDecryptionKey() : std::string{});

    std::vector<TensorMap> inputInterfaces;
    for (const json& interfaceJson : j.at("input_interfaces")) {
        TensorMap inputInterface;
        for (const std::string& name : inputNames) {
            const json& tensorJson = interfaceJson.at(name);
            uint64_t originalTensorId = tensorJson.at("id").get<uint64_t>();
            inputInterface.emplace(name, network->getApiTensorByOriginalId(originalTensorId));
        }
        inputInterfaces.push_back(std::move(inputInterface));
    }

    std::vector<TensorMap> outputInterfaces;
    for (const json& interfaceJson : j.at("output_interfaces")) {
        TensorMap outputInterface;
        for (const std::string& name : outputNames) {
            outputInterface.emplace(name, Tensor::deserialize(interfaceJson.at(name)));
        }
        outputInterfaces.push_back(std::move(outputInterface));
    }

    const json& parametersJson = j.at("parameters");
    if (!parametersJson.is_object()) {
        throw runtime_error("CustomLayer parameters must be an object keyed by parameter name.");
    }
    std::vector<std::shared_ptr<ParameterSpecification>> parameters;
    for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
        ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
        parameters.push_back(std::make_shared<ParameterSpecification>(std::move(parameter)));
    }

    CustomLayer customLayer(DynamicExpression::fromExpressionDefinition(expressionDefinition),
                            std::move(inputNames),
                            std::move(outputNames),
                            inputInterfaces,
                            outputInterfaces,
                            std::move(parameters));
    customLayer.initialized = true;
    customLayer.addToNetwork(network);
}

json CustomLayer::serialize(thor_file::TarWriter& archiveWriter,
                            Stream stream,
                            bool saveOptimizerState,
                            ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, "layer" + to_string(getId()));
    return j;
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("custom_layer", &Thor::CustomLayer::deserialize);
    return true;
}();
}  // namespace
