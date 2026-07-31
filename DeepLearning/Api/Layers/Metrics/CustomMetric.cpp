#include "DeepLearning/Api/Layers/Metrics/CustomMetric.h"

#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/FusedEquation.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

using PhysicalTensor = ThorImplementation::Tensor;
using PhysicalTensorMap = std::unordered_map<std::string, PhysicalTensor>;
using CompiledOutputs = ThorImplementation::CompiledOutputs;
using CompiledExecutionStage = ThorImplementation::CompiledExecutionStage;
using CompiledStageOutput = ThorImplementation::CompiledStageOutput;

std::string joinNames(const std::set<std::string>& names) {
    if (names.empty())
        return "<none>";

    std::ostringstream oss;
    bool first = true;
    for (const std::string& name : names) {
        if (!first)
            oss << ", ";
        oss << name;
        first = false;
    }
    return oss.str();
}

std::set<std::string> toNameSet(const std::vector<std::string>& names) { return std::set<std::string>(names.begin(), names.end()); }

std::set<std::string> expectedMetricOutputNames(const std::string& metricName, MetricAggregation aggregation) {
    std::set<std::string> names{metricName};
    if (aggregation == MetricAggregation::RATIO) {
        names.insert(Thor::METRIC_AGGREGATION_NUMERATOR_NAME);
        names.insert(Thor::METRIC_AGGREGATION_DENOMINATOR_NAME);
    }
    return names;
}

bool isScalarShape(const std::vector<uint64_t>& dimensions) {
    return std::all_of(dimensions.begin(), dimensions.end(), [](uint64_t dimension) { return dimension == 1; });
}

std::optional<DataType> finalOutputDType(const std::shared_ptr<CompiledOutputs>& compiledOutputs,
                                        const std::string& outputName) {
    for (const CompiledExecutionStage& stage : compiledOutputs->stages) {
        for (size_t outputIndex = 0; outputIndex < stage.outputs.size(); ++outputIndex) {
            if (stage.outputs[outputIndex].name == outputName)
                return stage.outputDType(outputIndex);
        }
    }
    for (const CompiledStageOutput& finalOutput : compiledOutputs->final_outputs) {
        if (finalOutput.name != outputName)
            continue;
        for (const CompiledExecutionStage& stage : compiledOutputs->stages) {
            for (size_t outputIdx = 0; outputIdx < stage.outputs.size(); ++outputIdx) {
                if (stage.outputs[outputIdx].value_id == finalOutput.value_id)
                    return stage.outputDType(outputIdx);
            }
        }
    }
    return std::nullopt;
}

}  // namespace

CustomMetric::CustomMetric(ThorImplementation::DynamicExpression expr,
                           Tensor predictions,
                           Tensor labels,
                           MetricAggregation aggregation,
                           std::string predictionsName,
                           std::string labelsName,
                           std::string metricName,
                           std::optional<Tensor> metricTensor,
                           std::string displayName,
                           bool usesBatchValidity)
    : expr(std::move(expr)),
      predictionsName(std::move(predictionsName)),
      labelsName(std::move(labelsName)),
      metricName(std::move(metricName)),
      displayName(std::move(displayName)),
      aggregation(aggregation),
      batchValidityMaskUsed(usesBatchValidity) {
    validateName(this->predictionsName, "predictions input");
    validateName(this->labelsName, "labels input");
    validateName(this->metricName, "metric output");
    if (this->predictionsName == this->labelsName)
        throw runtime_error("CustomMetric predictions and labels input names must be distinct.");
    if (this->displayName.empty())
        this->displayName = "Metric";

    const std::vector<std::string>& expectedInputs = this->expr.getExpectedInputNames();
    if (!expectedInputs.empty()) {
        std::set<std::string> expected(expectedInputs.begin(), expectedInputs.end());
        std::set<std::string> actual{this->predictionsName, this->labelsName};
        if (batchValidityMaskUsed)
            actual.insert(Thor::BATCH_VALIDITY_MASK_NAME);
        if (expected != actual) {
            throw runtime_error("CustomMetric expression input name mismatch. Expected {" + joinNames(expected) + "}, got {" +
                                joinNames(actual) + "}.");
        }
    }

    const std::vector<std::string>& expectedOutputs = this->expr.getExpectedOutputNames();
    if (!expectedOutputs.empty()) {
        const std::set<std::string> expected(expectedOutputs.begin(), expectedOutputs.end());
        const std::set<std::string> actual = expectedMetricOutputNames(this->metricName, this->aggregation);
        if (expected != actual) {
            throw runtime_error("CustomMetric expression output name mismatch. Expected {" + joinNames(expected) + "}, got {" +
                                joinNames(actual) + "}.");
        }
    }

    featureInput = std::move(predictions);
    labelsTensor = std::move(labels);

    Tensor inferredMetricTensor = inferMetricTensor();
    if (metricTensor.has_value()) {
        if (metricTensor.value().getDataType() != inferredMetricTensor.getDataType() ||
            metricTensor.value().getDimensions() != inferredMetricTensor.getDimensions()) {
            throw runtime_error("CustomMetric metric tensor must match the expression output descriptor. Expected " +
                                inferredMetricTensor.getDescriptorString() + ", got " +
                                metricTensor.value().getDescriptorString() + ".");
        }
        this->metricTensor = metricTensor.value();
    } else {
        this->metricTensor = inferredMetricTensor;
    }
    initialized = true;
}

void CustomMetric::validateName(const std::string& name, const std::string& what) {
    if (name.empty())
        throw runtime_error("CustomMetric " + what + " name cannot be empty.");
    if (name.length() >= 2 && name[0] == '_' && name[1] == '_')
        throw runtime_error("CustomMetric " + what + " names cannot start with __ that is reserved. Name " + name + " is illegal.");
}

PhysicalTensor CustomMetric::makeFakePlacedTensor(const Tensor& apiTensor) {
    std::vector<uint64_t> fakeDims;
    fakeDims.reserve(apiTensor.getDimensions().size() + 1);
    fakeDims.push_back(1);
    for (uint64_t dim : apiTensor.getDimensions())
        fakeDims.push_back(dim);

    ThorImplementation::TensorPlacement placement(ThorImplementation::TensorPlacement::MemDevices::CPU, 0);
    ThorImplementation::TensorDescriptor descriptor(apiTensor.getDataType(), fakeDims);
    return PhysicalTensor(placement, descriptor);
}

Tensor CustomMetric::logicalMetricTensorFromFakeOutput(const std::vector<uint64_t>& fakeOutputDims, DataType dtype) {
    return Tensor(dtype, fakeOutputDims.empty() ? std::vector<uint64_t>{1} : fakeOutputDims);
}

Tensor CustomMetric::inferMetricTensor() const {
    PhysicalTensorMap fakeInputs;
    fakeInputs.emplace(predictionsName, makeFakePlacedTensor(getPredictions()));
    fakeInputs.emplace(labelsName, makeFakePlacedTensor(getLabels()));
    if (batchValidityMaskUsed) {
        PhysicalTensor predictions = fakeInputs.at(predictionsName);
        std::vector<uint64_t> maskDimensions = predictions.getDimensions();
        for (size_t axis = 1; axis < maskDimensions.size(); ++axis)
            maskDimensions[axis] = 1;
        fakeInputs.emplace(Thor::BATCH_VALIDITY_MASK_NAME,
                           PhysicalTensor(predictions.getPlacement(),
                                          ThorImplementation::TensorDescriptor(DataType::FP32, maskDimensions)));
    }

    Stream fakeStream(0, Stream::Priority::REGULAR);
    ThorImplementation::DynamicExpressionBuild build = expr.build(fakeInputs, {}, fakeStream);

    const std::set<std::string> actualOutputNames = toNameSet(build.equation->getOutputNames());
    const std::set<std::string> expectedOutputNames = expectedMetricOutputNames(metricName, aggregation);
    if (actualOutputNames != expectedOutputNames) {
        throw runtime_error("CustomMetric expression output name mismatch. Expected {" + joinNames(expectedOutputNames) + "}, got {" +
                            joinNames(actualOutputNames) + "}.");
    }

    std::unordered_map<std::string, std::vector<uint64_t>> fakeOutputShapes =
        build.equation->getOutputShapes(build.stamp_inputs, build.tensor_scalar_inputs);
    std::shared_ptr<CompiledOutputs> compiledOutputs = build.equation->compileForInputs(build.stamp_inputs, {}, build.tensor_scalar_inputs);

    std::optional<DataType> metricDType = finalOutputDType(compiledOutputs, metricName);

    if (aggregation == MetricAggregation::RATIO) {
        for (const char* statisticName : {Thor::METRIC_AGGREGATION_NUMERATOR_NAME,
                                          Thor::METRIC_AGGREGATION_DENOMINATOR_NAME}) {
            auto statisticShapeIt = fakeOutputShapes.find(statisticName);
            if (statisticShapeIt == fakeOutputShapes.end())
                throw runtime_error("CustomMetric failed to infer output shape for '" + std::string(statisticName) + "'.");
            if (!isScalarShape(statisticShapeIt->second))
                throw runtime_error("CustomMetric ratio statistic output '" + std::string(statisticName) + "' must be scalar.");
            const std::optional<DataType> statisticDType = finalOutputDType(compiledOutputs, statisticName);
            if (!statisticDType.has_value())
                throw runtime_error("CustomMetric failed to infer output dtype for '" + std::string(statisticName) + "'.");
            if (statisticDType.value() != DataType::FP32)
                throw runtime_error("CustomMetric ratio statistic output '" + std::string(statisticName) + "' must be FP32.");
        }
    }

    auto shapeIt = fakeOutputShapes.find(metricName);
    if (shapeIt == fakeOutputShapes.end())
        throw runtime_error("CustomMetric failed to infer output shape for '" + metricName + "'.");
    if (aggregation == MetricAggregation::RATIO && !isScalarShape(shapeIt->second))
        throw runtime_error("CustomMetric ratio metric output '" + metricName + "' must be scalar.");
    if (!metricDType.has_value())
        throw runtime_error("CustomMetric failed to infer output dtype for '" + metricName + "'.");

    return logicalMetricTensorFromFakeOutput(shapeIt->second, metricDType.value());
}

std::shared_ptr<ThorImplementation::Layer> CustomMetric::stamp(ThorImplementation::TensorPlacement placement,
                                                               std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                               std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                               Thor::Tensor connectingApiTensor,
                                                               const bool inferenceOnly) const {
    (void)placement;
    (void)drivingLayer;
    (void)drivingApiLayer;
    (void)inferenceOnly;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value() || connectingApiTensor == labelsTensor);

    const std::optional<std::string> validityMaskName =
        batchValidityMaskUsed
            ? std::optional<std::string>(Thor::BATCH_VALIDITY_MASK_NAME)
            : std::nullopt;
    return std::make_shared<ThorImplementation::CustomMetric>(
        expr, predictionsName, labelsName, metricName, displayName, aggregation, validityMaskName);
}

uint64_t CustomMetric::getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                             ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    const uint64_t validityMaskBytes =
        batchValidityMaskUsed ? static_cast<uint64_t>(batchSize) * sizeof(float) : 0;
    const uint64_t ratioStatisticBytes = aggregation == MetricAggregation::RATIO ? 4 * sizeof(float) : 0;
    return metricTensor.getTotalSizeInBytes() + validityMaskBytes + ratioStatisticBytes;
}

json CustomMetric::architectureJson() const {
    json j = Metric::architectureJson();
    j["layer_type"] = "custom_metric";
    j["predictions_name"] = predictionsName;
    j["labels_name"] = labelsName;
    j["metric_name"] = metricName;
    j["display_name"] = displayName;
    j["uses_batch_validity"] = batchValidityMaskUsed;

    auto serializedDefinition = expr.getSerializedDefinition();
    if (serializedDefinition == nullptr) {
        throw runtime_error(
            "CustomMetric expression is not serializable. Construct it from a ThorImplementation::ExpressionDefinition or "
            "DynamicExpression::fromExpressionDefinition(...). Arbitrary DynamicExpression builders cannot be saved.");
    }
    j["expression"] = serializedDefinition->architectureJson();
    return j;
}

void CustomMetric::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in CustomMetric::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "custom_metric")
        throw runtime_error("Layer type mismatch in CustomMetric::deserialize: " + j.at("layer_type").get<std::string>());

    const std::string predictionsName = j.value("predictions_name", std::string("predictions"));
    const std::string labelsName = j.value("labels_name", std::string("labels"));
    const std::string metricName = j.value("metric_name", std::string("metric"));
    const std::string displayName = j.value("display_name", std::string("Metric"));
    if (j.contains("supports_partial_batches"))
        throw runtime_error("CustomMetric field supports_partial_batches is no longer supported; use uses_batch_validity.");
    if (j.contains("uses_batch_validity_mask"))
        throw runtime_error("CustomMetric field uses_batch_validity_mask is no longer supported; use uses_batch_validity.");
    const bool usesBatchValidity = j.at("uses_batch_validity").get<bool>();
    const MetricAggregation aggregation = j.at("aggregation").get<MetricAggregation>();

    const uint64_t predictionsOriginalId = j.at("predictions").at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(predictionsOriginalId);

    const uint64_t labelsOriginalId = j.at("labels").at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(labelsOriginalId);

    Tensor metricTensor = Tensor::deserialize(j.at("metric").get<json>());

    ThorImplementation::ExpressionDefinition expressionDefinition = ThorImplementation::ExpressionDefinition::deserialize(
        j.at("expression"),
        network != nullptr && network->allowUnsafeLoadedCudaKernelSourceCompilation(),
        network != nullptr ? network->trustedLoadedCudaKernelPublicKey() : std::string{},
        network != nullptr ? network->trustedLoadedCudaKernelSourceDecryptionKey() : std::string{});

    CustomMetric customMetric(ThorImplementation::DynamicExpression::fromExpressionDefinition(expressionDefinition),
                              predictions,
                              labels,
                              aggregation,
                              predictionsName,
                              labelsName,
                              metricName,
                              metricTensor,
                              displayName,
                              usesBatchValidity);
    customMetric.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Metric::register_layer("custom_metric", &Thor::CustomMetric::deserialize);
    return true;
}();
}  // namespace
