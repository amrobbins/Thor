#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/LaplaceNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedBroadcast.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kLocationName = "location";
constexpr const char* kScaleName = "scale";
constexpr const char* kTargetName = "target";
constexpr const char* kExampleWeightsName = "example_weights";
constexpr const char* kLossName = "loss";
constexpr const char* kLocationGradientName = "location_grad";
constexpr const char* kScaleGradientName = "scale_grad";
constexpr double kLogTwo = 0.693147180559945309417232121458;

void validateFloatingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported LaplaceNLLLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateExampleWeights(Tensor location, Tensor scale, Tensor target, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == location || exampleWeights.value() == scale || exampleWeights.value() == target)
        throw runtime_error("LaplaceNLLLoss example_weights tensor must be distinct from location, scale, and target.");
    validateFloatingDType("example_weights", exampleWeights.value().getDataType());
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != location.getDimensions()) {
        throw runtime_error("LaplaceNLLLoss example_weights dimensions must be [1] for per-example weights or match location dimensions.");
    }
}

vector<uint64_t> raggedPackedScalarWeightBroadcastDimensions(const vector<uint64_t>& locationValueDimensions) {
    if (locationValueDimensions.empty()) throw invalid_argument("LaplaceNLLLoss ragged location values must have a packed-capacity dimension.");
    vector<uint64_t> dimensions(locationValueDimensions.size(), 1);
    dimensions[0] = locationValueDimensions[0];
    return dimensions;
}

ThorImplementation::Expression safeScale(const ThorImplementation::Expression& scale, float eps) {
    return scale.max(ThorImplementation::Expression(eps));
}

ThorImplementation::Expression signOf(const ThorImplementation::Expression& value) {
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression positive(1.0f);
    ThorImplementation::Expression negative(-1.0f);
    return ThorImplementation::Expression::where(
        value > zero, positive, ThorImplementation::Expression::where(value < zero, negative, zero));
}

ThorImplementation::Expression laplaceLoss(const ThorImplementation::Expression& location,
                                           const ThorImplementation::Expression& scaleOrLogScale,
                                           const ThorImplementation::Expression& target,
                                           bool logScale,
                                           float eps) {
    ThorImplementation::Expression absDiff = (location - target).abs();
    if (logScale) {
        return ThorImplementation::Expression(kLogTwo) + scaleOrLogScale +
               absDiff * (ThorImplementation::Expression(0.0) - scaleOrLogScale).exp();
    }
    ThorImplementation::Expression scale = safeScale(scaleOrLogScale, eps);
    return ThorImplementation::Expression(kLogTwo) + scale.ln() + absDiff / scale;
}

ThorImplementation::DynamicExpression makeLaplaceNLLLossExpression(DataType lossDataType,
                                                                   bool logScale,
                                                                   float eps,
                                                                   bool weighted,
                                                                   optional<vector<uint64_t>> raggedLocationValueDimensions = nullopt) {
    validateFloatingDType("loss", lossDataType);
    ThorImplementation::Expression location = ThorImplementation::Expression::input(kLocationName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scale = ThorImplementation::Expression::input(kScaleName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss = laplaceLoss(location, scale, target, logScale, eps);
    if (weighted) {
        ThorImplementation::Expression exampleWeights =
            ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
        if (raggedLocationValueDimensions.has_value())
            exampleWeights = exampleWeights.reshape(raggedPackedScalarWeightBroadcastDimensions(raggedLocationValueDimensions.value()));
        loss = loss * exampleWeights;
    }
    loss = loss.withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeLaplaceNLLGradientExpression(DataType locationDType,
                                                                       DataType scaleDType,
                                                                       bool logScale,
                                                                       float eps,
                                                                       bool weighted,
                                                                       optional<vector<uint64_t>> raggedLocationValueDimensions = nullopt) {
    validateFloatingDType("location", locationDType);
    validateFloatingDType("scale", scaleDType);

    ThorImplementation::Expression location = ThorImplementation::Expression::input(kLocationName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scaleInput = ThorImplementation::Expression::input(kScaleName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression diff = location - target;
    ThorImplementation::Expression absDiff = diff.abs();

    ThorImplementation::Expression locationGradient = [&]() {
        if (logScale) {
            ThorImplementation::Expression inverseScale = (ThorImplementation::Expression(0.0) - scaleInput).exp();
            return signOf(diff) * inverseScale;
        }
        ThorImplementation::Expression scale = safeScale(scaleInput, eps);
        return signOf(diff) / scale;
    }();

    ThorImplementation::Expression scaleGradient = [&]() {
        if (logScale) {
            ThorImplementation::Expression inverseScale = (ThorImplementation::Expression(0.0) - scaleInput).exp();
            return ThorImplementation::Expression(1.0) - absDiff * inverseScale;
        }
        ThorImplementation::Expression scale = safeScale(scaleInput, eps);
        return ThorImplementation::Expression(1.0) / scale - absDiff / (scale * scale);
    }();

    if (weighted) {
        ThorImplementation::Expression exampleWeights =
            ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
        if (raggedLocationValueDimensions.has_value())
            exampleWeights = exampleWeights.reshape(raggedPackedScalarWeightBroadcastDimensions(raggedLocationValueDimensions.value()));
        locationGradient = locationGradient * exampleWeights;
        scaleGradient = scaleGradient * exampleWeights;
    }

    ThorImplementation::Expression lossScale(ThorImplementation::Loss::getLossScalingFactor());
    locationGradient = (locationGradient * lossScale).withOutputDType(locationDType);
    scaleGradient = (scaleGradient * lossScale).withOutputDType(scaleDType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kLocationGradientName, locationGradient}, {kScaleGradientName, scaleGradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void LaplaceNLLLoss::buildSupportLayersAndAddToNetwork() {
    validateFloatingDType("location", predictionsTensor.getDataType());
    validateFloatingDType("scale", scaleTensor.getDataType());
    validateFloatingDType("target", labelsTensor.getDataType());
    validateExampleWeights(predictionsTensor, scaleTensor, labelsTensor, exampleWeightsTensor);
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions() == scaleTensor.getDimensions());
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions() == labelsTensor.getDimensions());
    THOR_THROW_IF_FALSE(eps > 0.0f);

    if (isRagged()) {
        if (lossShape == LossShape::PER_OUTPUT)
            throw invalid_argument("LaplaceNLLLoss LossShape::PER_OUTPUT is undefined for ragged sequences.");

        optional<RaggedTensor> broadcastWeights;
        if (exampleWeightsTensor.has_value()) {
            if (exampleWeightsTensor->getDimensions() != vector<uint64_t>{1})
                throw invalid_argument("LaplaceNLLLoss ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
            TypeConverter weightConverter = TypeConverter::Builder()
                                                .network(*network)
                                                .featureInput(exampleWeightsTensor.value())
                                                .newDataType(DataType::FP32)
                                                .build();
            SegmentedBroadcast weightBroadcast = SegmentedBroadcast::Builder()
                                                     .network(*network)
                                                     .featureInput(weightConverter.getFeatureOutput().value())
                                                     .partitionInput(raggedPredictionsTensor.value())
                                                     .build();
            broadcastWeights = weightBroadcast.getRaggedFeatureOutput();
        }

        const vector<uint64_t> dims = raggedPredictionsTensor->getValuesDimensions();
        RaggedCustomLoss::Builder builder;
        builder.network(*network)
            .predictions(raggedPredictionsTensor.value())
            .labels(raggedLabelsTensor.value())
            .secondaryInput(raggedScaleTensor.value(), kScaleName, kScaleGradientName)
            .lossExpression(makeLaplaceNLLLossExpression(lossDataType,
                                                         logScale,
                                                         eps,
                                                         broadcastWeights.has_value(),
                                                         broadcastWeights.has_value()
                                                             ? optional<vector<uint64_t>>(dims)
                                                             : nullopt))
            .gradientExpression(makeLaplaceNLLGradientExpression(predictionsTensor.getDataType(),
                                                                 scaleTensor.getDataType(),
                                                                 logScale,
                                                                 eps,
                                                                 broadcastWeights.has_value(),
                                                                 broadcastWeights.has_value()
                                                                     ? optional<vector<uint64_t>>(dims)
                                                                     : nullopt))
            .predictionsName(kLocationName)
            .labelsName(kTargetName)
            .lossName(kLossName)
            .gradientName(kLocationGradientName)
            .lossDataType(lossDataType)
            .lossWeight(lossWeight.value_or(1.0f));
        if (broadcastWeights.has_value())
            builder.exampleWeights(broadcastWeights.value()).exampleWeightsName(kExampleWeightsName);

        RaggedCustomLoss rawLaplaceNLLLoss = builder.build();
        raggedRawLossTensor = rawLaplaceNLLLoss.getRaggedRawLoss();
        lossShaperInput = raggedRawLossTensor->getValues();
        if (lossShape == LossShape::NONE) {
            lossTensor = lossShaperInput;
            Stub::Builder().network(*network).inputTensor(lossShaperInput).build();
        } else if (lossShape == LossShape::RAW) {
            lossTensor = lossShaperInput;
        } else if (lossShape == LossShape::PER_EXAMPLE) {
            lossTensor = RaggedLossShaper::Builder()
                             .network(*network)
                             .lossInput(raggedRawLossTensor.value())
                             .reportsPerExampleLoss()
                             .build()
                             .getLossOutput();
        } else if (lossShape == LossShape::BATCH) {
            lossTensor = RaggedLossShaper::Builder()
                             .network(*network)
                             .lossInput(raggedRawLossTensor.value())
                             .reportsBatchLoss()
                             .build()
                             .getLossOutput();
        } else {
            THOR_UNREACHABLE();
        }
        return;
    }

    MultiInputCustomLoss::Builder builder;
    builder.network(*network)
        .lossExpression(makeLaplaceNLLLossExpression(lossDataType, logScale, eps, exampleWeightsTensor.has_value()))
        .gradientExpression(makeLaplaceNLLGradientExpression(predictionsTensor.getDataType(),
                                                             scaleTensor.getDataType(),
                                                             logScale,
                                                             eps,
                                                             exampleWeightsTensor.has_value()))
        .input(kLocationName, predictionsTensor, std::string(kLocationGradientName))
        .auxiliaryInput(kTargetName, labelsTensor)
        .input(kScaleName, scaleTensor, std::string(kScaleGradientName))
        .lossName(kLossName)
        .lossDataType(lossDataType)
        .lossWeight(lossWeight.value_or(1.0f))
        .reportsRawLoss();
    if (exampleWeightsTensor.has_value())
        builder.auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value());

    MultiInputCustomLoss rawLaplaceNLLLoss = builder.build();
    lossShaperInput = rawLaplaceNLLLoss.getLoss();
    finalizeLossReporting();
}

json LaplaceNLLLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "laplace_nll_loss";
    j["loss_shape"] = lossShape;
    j["scale_tensor"] = scaleTensor.architectureJson();
    j["log_scale"] = logScale;
    j["eps"] = eps;
    if (isRagged()) {
        j["ragged_predictions"] = raggedPredictionsTensor->architectureJson();
        j["ragged_labels"] = raggedLabelsTensor->architectureJson();
        j["ragged_scale"] = raggedScaleTensor->architectureJson();
        if (raggedRawLossTensor.has_value()) j["ragged_raw_loss"] = raggedRawLossTensor->architectureJson();
    }
    return j;
}

void LaplaceNLLLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in LaplaceNLLLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "laplace_nll_loss")
        throw runtime_error("Layer type mismatch in LaplaceNLLLoss::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor location = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "LaplaceNLLLoss");
        RaggedTensor target = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "LaplaceNLLLoss");
        RaggedTensor scale = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_scale"), network, "LaplaceNLLLoss");
        Builder builder;
        builder.network(*network)
            .location(location)
            .scale(scale)
            .target(target)
            .logScale(j.value("log_scale", true))
            .eps(j.value("eps", 1.0e-8f))
            .lossDataType(j.at("loss_data_type").get<DataType>())
            .lossWeight(ThorImplementation::lossWeightFromJson(j).value_or(1.0f));
        if (j.contains("example_weights_tensor"))
            builder.exampleWeights(network->getApiTensorByOriginalId(j["example_weights_tensor"].at("id").get<uint64_t>()));
        switch (j.at("loss_shape").get<LossShape>()) {
            case LossShape::NONE: builder.reportsNoLoss(); break;
            case LossShape::RAW: builder.reportsRawLoss(); break;
            case LossShape::PER_EXAMPLE: builder.reportsPerExampleLoss(); break;
            case LossShape::BATCH: builder.reportsBatchLoss(); break;
            case LossShape::PER_OUTPUT: throw runtime_error("Serialized ragged LaplaceNLLLoss cannot use PER_OUTPUT.");
        }
        (void)builder.build();
        return;
    }

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor location = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor target = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["scale_tensor"].at("id").get<uint64_t>();
    Tensor scale = network->getApiTensorByOriginalId(originalTensorId);

    LaplaceNLLLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.logScale = j.value("log_scale", true);
    loss.eps = j.value("eps", 1.0e-8f);
    loss.predictionsTensor = location;
    loss.labelsTensor = target;
    loss.scaleTensor = scale;
    if (j.contains("example_weights_tensor")) {
        originalTensorId = j["example_weights_tensor"].at("id").get<uint64_t>();
        loss.exampleWeightsTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("laplace_nll_loss", &Thor::LaplaceNLLLoss::deserialize);
    return true;
}();
}  // namespace
