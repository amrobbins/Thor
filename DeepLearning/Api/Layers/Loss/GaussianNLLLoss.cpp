#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/GaussianNLLLoss.h"
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

constexpr const char* kMeanName = "mean";
constexpr const char* kTargetName = "target";
constexpr const char* kVarianceName = "variance";
constexpr const char* kExampleWeightsName = "example_weights";
constexpr const char* kLossName = "loss";
constexpr const char* kMeanGradientName = "mean_grad";
constexpr const char* kVarianceGradientName = "variance_grad";
constexpr double kLogTwoPi = 1.837877066409345483560659472811;

void validateFloatingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported GaussianNLLLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateGaussianDTypes(DataType meanDType, DataType targetDType, DataType varianceDType) {
    validateFloatingDType("mean", meanDType);
    validateFloatingDType("target", targetDType);
    validateFloatingDType("variance", varianceDType);
}

void validateExampleWeights(Tensor mean, Tensor target, Tensor variance, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == mean || exampleWeights.value() == target || exampleWeights.value() == variance)
        throw runtime_error("GaussianNLLLoss example_weights tensor must be distinct from mean, target, and variance.");
    validateFloatingDType("example_weights", exampleWeights.value().getDataType());
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != mean.getDimensions()) {
        throw runtime_error(
            "GaussianNLLLoss example_weights dimensions must be [1] for per-example weights or match mean dimensions.");
    }
}

vector<uint64_t> raggedPackedScalarWeightBroadcastDimensions(const vector<uint64_t>& meanValueDimensions) {
    if (meanValueDimensions.empty()) throw invalid_argument("GaussianNLLLoss ragged mean values must have a packed-capacity dimension.");
    vector<uint64_t> dimensions(meanValueDimensions.size(), 1);
    dimensions[0] = meanValueDimensions[0];
    return dimensions;
}

ThorImplementation::Expression clampedVariance(const ThorImplementation::Expression& variance, float eps) {
    return variance.max(ThorImplementation::Expression(eps));
}

ThorImplementation::Expression gaussianLoss(const ThorImplementation::Expression& mean,
                                            const ThorImplementation::Expression& target,
                                            const ThorImplementation::Expression& varianceOrLogVariance,
                                            bool logVariance,
                                            bool full,
                                            float eps) {
    ThorImplementation::Expression diff = mean - target;
    ThorImplementation::Expression loss = [&]() {
        if (logVariance) {
            return ThorImplementation::Expression(0.5) *
                   (varianceOrLogVariance + diff * diff * (ThorImplementation::Expression(0.0) - varianceOrLogVariance).exp());
        }
        ThorImplementation::Expression safeVariance = clampedVariance(varianceOrLogVariance, eps);
        return ThorImplementation::Expression(0.5) * (safeVariance.ln() + (diff * diff) / safeVariance);
    }();
    if (full)
        loss = loss + ThorImplementation::Expression(0.5 * kLogTwoPi);
    return loss;
}

ThorImplementation::DynamicExpression makeGaussianNLLLossExpression(DataType lossDataType,
                                                                    bool logVariance,
                                                                    bool full,
                                                                    float eps,
                                                                    bool weighted,
                                                                    optional<vector<uint64_t>> raggedMeanValueDimensions = nullopt) {
    validateFloatingDType("loss", lossDataType);

    ThorImplementation::Expression mean = ThorImplementation::Expression::input(kMeanName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression variance = ThorImplementation::Expression::input(kVarianceName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss = gaussianLoss(mean, target, variance, logVariance, full, eps);
    if (weighted) {
        ThorImplementation::Expression exampleWeights =
            ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
        if (raggedMeanValueDimensions.has_value())
            exampleWeights = exampleWeights.reshape(raggedPackedScalarWeightBroadcastDimensions(raggedMeanValueDimensions.value()));
        loss = loss * exampleWeights;
    }
    loss = loss.withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeGaussianNLLGradientExpression(DataType meanDType,
                                                                        DataType varianceDType,
                                                                        bool logVariance,
                                                                        float eps,
                                                                        bool weighted,
                                                                        optional<vector<uint64_t>> raggedMeanValueDimensions = nullopt) {
    validateFloatingDType("mean", meanDType);
    validateFloatingDType("variance", varianceDType);

    ThorImplementation::Expression mean = ThorImplementation::Expression::input(kMeanName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression variance = ThorImplementation::Expression::input(kVarianceName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression diff = mean - target;
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());

    ThorImplementation::Expression meanGradient = [&]() {
        if (logVariance)
            return diff * (ThorImplementation::Expression(0.0) - variance).exp();
        ThorImplementation::Expression safeVariance = clampedVariance(variance, eps);
        return diff / safeVariance;
    }();

    ThorImplementation::Expression varianceGradient = [&]() {
        if (logVariance) {
            return ThorImplementation::Expression(0.5) *
                   (ThorImplementation::Expression(1.0) - diff * diff * (ThorImplementation::Expression(0.0) - variance).exp());
        }
        ThorImplementation::Expression safeVariance = clampedVariance(variance, eps);
        return ThorImplementation::Expression(0.5) *
               ((ThorImplementation::Expression(1.0) / safeVariance) - ((diff * diff) / (safeVariance * safeVariance)));
    }();

    if (weighted) {
        ThorImplementation::Expression exampleWeights =
            ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
        if (raggedMeanValueDimensions.has_value())
            exampleWeights = exampleWeights.reshape(raggedPackedScalarWeightBroadcastDimensions(raggedMeanValueDimensions.value()));
        meanGradient = meanGradient * exampleWeights;
        varianceGradient = varianceGradient * exampleWeights;
    }

    meanGradient = (meanGradient * scale).withOutputDType(meanDType);
    varianceGradient = (varianceGradient * scale).withOutputDType(varianceDType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kMeanGradientName, meanGradient}, {kVarianceGradientName, varianceGradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void GaussianNLLLoss::buildSupportLayersAndAddToNetwork() {
    validateGaussianDTypes(predictionsTensor.getDataType(), labelsTensor.getDataType(), varianceTensor.getDataType());
    validateExampleWeights(predictionsTensor, labelsTensor, varianceTensor, exampleWeightsTensor);
    THOR_THROW_IF_FALSE(eps > 0.0f);

    if (isRagged()) {
        if (lossShape == LossShape::PER_OUTPUT)
            throw invalid_argument("GaussianNLLLoss LossShape::PER_OUTPUT is undefined for ragged sequences.");

        optional<RaggedTensor> broadcastWeights;
        if (exampleWeightsTensor.has_value()) {
            if (exampleWeightsTensor->getDimensions() != vector<uint64_t>{1})
                throw invalid_argument("GaussianNLLLoss ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
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
            .secondaryInput(raggedVarianceTensor.value(), kVarianceName, kVarianceGradientName)
            .lossExpression(makeGaussianNLLLossExpression(lossDataType,
                                                          logVariance,
                                                          full,
                                                          eps,
                                                          broadcastWeights.has_value(),
                                                          broadcastWeights.has_value()
                                                              ? optional<vector<uint64_t>>(dims)
                                                              : nullopt))
            .gradientExpression(makeGaussianNLLGradientExpression(predictionsTensor.getDataType(),
                                                                  varianceTensor.getDataType(),
                                                                  logVariance,
                                                                  eps,
                                                                  broadcastWeights.has_value(),
                                                                  broadcastWeights.has_value()
                                                                      ? optional<vector<uint64_t>>(dims)
                                                                      : nullopt))
            .predictionsName(kMeanName)
            .labelsName(kTargetName)
            .lossName(kLossName)
            .gradientName(kMeanGradientName)
            .lossDataType(lossDataType)
            .lossWeight(lossWeight.value_or(1.0f));
        if (broadcastWeights.has_value())
            builder.exampleWeights(broadcastWeights.value()).exampleWeightsName(kExampleWeightsName);

        RaggedCustomLoss rawGaussianNLLLoss = builder.build();
        raggedRawLossTensor = rawGaussianNLLLoss.getRaggedRawLoss();
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
        .lossExpression(makeGaussianNLLLossExpression(lossDataType, logVariance, full, eps, exampleWeightsTensor.has_value()))
        .gradientExpression(makeGaussianNLLGradientExpression(predictionsTensor.getDataType(),
                                                              varianceTensor.getDataType(),
                                                              logVariance,
                                                              eps,
                                                              exampleWeightsTensor.has_value()))
        .input(kMeanName, predictionsTensor, std::string(kMeanGradientName))
        .auxiliaryInput(kTargetName, labelsTensor)
        .input(kVarianceName, varianceTensor, std::string(kVarianceGradientName))
        .lossName(kLossName)
        .lossDataType(lossDataType)
        .lossWeight(lossWeight.value_or(1.0f))
        .reportsRawLoss();
    if (exampleWeightsTensor.has_value())
        builder.auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value());

    MultiInputCustomLoss rawGaussianNLLLoss = builder.build();
    lossShaperInput = rawGaussianNLLLoss.getLoss();
    finalizeLossReporting();
}

json GaussianNLLLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "gaussian_nll_loss";
    j["loss_shape"] = lossShape;
    j["variance_tensor"] = varianceTensor.architectureJson();
    j["log_variance"] = logVariance;
    j["full"] = full;
    j["eps"] = eps;
    if (isRagged()) {
        j["ragged_predictions"] = raggedPredictionsTensor->architectureJson();
        j["ragged_labels"] = raggedLabelsTensor->architectureJson();
        j["ragged_variance"] = raggedVarianceTensor->architectureJson();
        if (raggedRawLossTensor.has_value()) j["ragged_raw_loss"] = raggedRawLossTensor->architectureJson();
    }
    return j;
}

void GaussianNLLLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in GaussianNLLLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "gaussian_nll_loss")
        throw runtime_error("Layer type mismatch in GaussianNLLLoss::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor mean = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "GaussianNLLLoss");
        RaggedTensor target = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "GaussianNLLLoss");
        RaggedTensor variance = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_variance"), network, "GaussianNLLLoss");
        Builder builder;
        builder.network(*network)
            .mean(mean)
            .target(target)
            .variance(variance)
            .logVariance(j.value("log_variance", false))
            .full(j.value("full", false))
            .eps(j.value("eps", 1.0e-6f))
            .lossDataType(j.at("loss_data_type").get<DataType>())
            .lossWeight(ThorImplementation::lossWeightFromJson(j).value_or(1.0f));
        if (j.contains("example_weights_tensor"))
            builder.exampleWeights(network->getApiTensorByOriginalId(j["example_weights_tensor"].at("id").get<uint64_t>()));
        switch (j.at("loss_shape").get<LossShape>()) {
            case LossShape::NONE: builder.reportsNoLoss(); break;
            case LossShape::RAW: builder.reportsRawLoss(); break;
            case LossShape::PER_EXAMPLE: builder.reportsPerExampleLoss(); break;
            case LossShape::BATCH: builder.reportsBatchLoss(); break;
            case LossShape::PER_OUTPUT: throw runtime_error("Serialized ragged GaussianNLLLoss cannot use PER_OUTPUT.");
        }
        (void)builder.build();
        return;
    }

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["variance_tensor"].at("id").get<uint64_t>();
    Tensor variance = network->getApiTensorByOriginalId(originalTensorId);

    GaussianNLLLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.logVariance = j.value("log_variance", false);
    loss.full = j.value("full", false);
    loss.eps = j.value("eps", 1.0e-6f);
    loss.predictionsTensor = predictions;
    loss.labelsTensor = labels;
    loss.varianceTensor = variance;
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
    Thor::Loss::register_layer("gaussian_nll_loss", &Thor::GaussianNLLLoss::deserialize);
    return true;
}();
}  // namespace
