#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/StudentTNLLLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kLocationName = "location";
constexpr const char* kLogScaleName = "log_scale";
constexpr const char* kTargetName = "target";
constexpr const char* kLogDegreesOfFreedomName = "log_degrees_of_freedom";
constexpr const char* kExampleWeightsName = "example_weights";
constexpr const char* kLossName = "loss";
constexpr const char* kLocationGradientName = "location_grad";
constexpr const char* kLogScaleGradientName = "log_scale_grad";
constexpr const char* kLogDegreesOfFreedomGradientName = "log_degrees_of_freedom_grad";
constexpr double kLogPi = 1.1447298858494001741434273513531;

void validateFloatingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported StudentTNLLLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateExampleWeights(Tensor location,
                            Tensor logScale,
                            Tensor target,
                            optional<Tensor> logDegreesOfFreedom,
                            optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == location || exampleWeights.value() == logScale || exampleWeights.value() == target ||
        (logDegreesOfFreedom.has_value() && exampleWeights.value() == logDegreesOfFreedom.value())) {
        throw runtime_error(
            "StudentTNLLLoss example_weights tensor must be distinct from location, log_scale, target, and learned degrees of freedom.");
    }
    validateFloatingDType("example_weights", exampleWeights.value().getDataType());
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != location.getDimensions()) {
        throw runtime_error(
            "StudentTNLLLoss example_weights dimensions must be [1] for per-example weights or match location dimensions.");
    }
}

ThorImplementation::Expression degreesOfFreedomExpression(optional<ThorImplementation::Expression> learnedLogDegreesOfFreedom,
                                                           float fixedDegreesOfFreedom,
                                                           float minimumDegreesOfFreedom) {
    if (learnedLogDegreesOfFreedom.has_value())
        return ThorImplementation::Expression(minimumDegreesOfFreedom) + learnedLogDegreesOfFreedom.value().exp();
    return ThorImplementation::Expression(fixedDegreesOfFreedom);
}

ThorImplementation::Expression studentTNLL(const ThorImplementation::Expression& location,
                                           const ThorImplementation::Expression& logScale,
                                           const ThorImplementation::Expression& target,
                                           optional<ThorImplementation::Expression> learnedLogDegreesOfFreedom,
                                           float fixedDegreesOfFreedom,
                                           float minimumDegreesOfFreedom) {
    ThorImplementation::Expression degreesOfFreedom =
        degreesOfFreedomExpression(learnedLogDegreesOfFreedom, fixedDegreesOfFreedom, minimumDegreesOfFreedom);
    ThorImplementation::Expression logDegreesOfFreedom =
        learnedLogDegreesOfFreedom.has_value() && minimumDegreesOfFreedom == 0.0f
            ? learnedLogDegreesOfFreedom.value()
            : degreesOfFreedom.ln();
    ThorImplementation::Expression inverseScale = (ThorImplementation::Expression(0.0) - logScale).exp();
    ThorImplementation::Expression standardizedResidual = (location - target) * inverseScale;
    ThorImplementation::Expression residualSquared = standardizedResidual * standardizedResidual;
    ThorImplementation::Expression halfDegreesOfFreedom = degreesOfFreedom * ThorImplementation::Expression(0.5);
    ThorImplementation::Expression halfDegreesOfFreedomPlusOne =
        (degreesOfFreedom + ThorImplementation::Expression(1.0)) * ThorImplementation::Expression(0.5);

    return logScale + halfDegreesOfFreedom.lgamma() - halfDegreesOfFreedomPlusOne.lgamma() +
           ThorImplementation::Expression(0.5) * (logDegreesOfFreedom + ThorImplementation::Expression(kLogPi)) +
           halfDegreesOfFreedomPlusOne * (residualSquared / degreesOfFreedom).log1p();
}

ThorImplementation::DynamicExpression makeStudentTNLLLossExpression(DataType lossDataType,
                                                                    bool learnedDegreesOfFreedom,
                                                                    float fixedDegreesOfFreedom,
                                                                    float minimumDegreesOfFreedom,
                                                                    bool weighted) {
    validateFloatingDType("loss", lossDataType);
    ThorImplementation::Expression location = ThorImplementation::Expression::input(kLocationName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression logScale = ThorImplementation::Expression::input(kLogScaleName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    optional<ThorImplementation::Expression> logDegreesOfFreedom;
    if (learnedDegreesOfFreedom) {
        logDegreesOfFreedom =
            ThorImplementation::Expression::input(kLogDegreesOfFreedomName, DataType::FP32, DataType::FP32);
    }
    ThorImplementation::Expression loss =
        studentTNLL(location, logScale, target, logDegreesOfFreedom, fixedDegreesOfFreedom, minimumDegreesOfFreedom);
    if (weighted) {
        ThorImplementation::Expression exampleWeights =
            ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
        loss = loss * exampleWeights;
    }
    loss = loss.withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeStudentTNLLGradientExpression(DataType locationDType,
                                                                        DataType logScaleDType,
                                                                        optional<DataType> logDegreesOfFreedomDType,
                                                                        float fixedDegreesOfFreedom,
                                                                        float minimumDegreesOfFreedom,
                                                                        bool weighted) {
    validateFloatingDType("location", locationDType);
    validateFloatingDType("log_scale", logScaleDType);
    if (logDegreesOfFreedomDType.has_value())
        validateFloatingDType("log_degrees_of_freedom", logDegreesOfFreedomDType.value());

    ThorImplementation::Expression location = ThorImplementation::Expression::input(kLocationName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression logScale = ThorImplementation::Expression::input(kLogScaleName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    optional<ThorImplementation::Expression> logDegreesOfFreedom;
    if (logDegreesOfFreedomDType.has_value()) {
        logDegreesOfFreedom =
            ThorImplementation::Expression::input(kLogDegreesOfFreedomName, DataType::FP32, DataType::FP32);
    }

    ThorImplementation::Expression degreesOfFreedom =
        degreesOfFreedomExpression(logDegreesOfFreedom, fixedDegreesOfFreedom, minimumDegreesOfFreedom);
    ThorImplementation::Expression inverseScale = (ThorImplementation::Expression(0.0) - logScale).exp();
    ThorImplementation::Expression standardizedResidual = (location - target) * inverseScale;
    ThorImplementation::Expression residualSquared = standardizedResidual * standardizedResidual;
    ThorImplementation::Expression denominator = degreesOfFreedom + residualSquared;
    ThorImplementation::Expression degreesOfFreedomPlusOne = degreesOfFreedom + ThorImplementation::Expression(1.0);

    ThorImplementation::Expression locationGradient =
        degreesOfFreedomPlusOne * standardizedResidual * inverseScale / denominator;
    ThorImplementation::Expression logScaleGradient =
        ThorImplementation::Expression(1.0) - degreesOfFreedomPlusOne * residualSquared / denominator;

    optional<ThorImplementation::Expression> logDegreesOfFreedomGradient;
    if (logDegreesOfFreedom.has_value()) {
        ThorImplementation::Expression halfDegreesOfFreedom = degreesOfFreedom * ThorImplementation::Expression(0.5);
        ThorImplementation::Expression halfDegreesOfFreedomPlusOne =
            degreesOfFreedomPlusOne * ThorImplementation::Expression(0.5);
        if (minimumDegreesOfFreedom == 0.0f) {
            logDegreesOfFreedomGradient = ThorImplementation::Expression(0.5) *
                (degreesOfFreedom * (halfDegreesOfFreedom.digamma() - halfDegreesOfFreedomPlusOne.digamma()) +
                 ThorImplementation::Expression(1.0) +
                 degreesOfFreedom * (residualSquared / degreesOfFreedom).log1p() -
                 degreesOfFreedomPlusOne * residualSquared / denominator);
        } else {
            ThorImplementation::Expression degreesOfFreedomExcess = logDegreesOfFreedom.value().exp();
            ThorImplementation::Expression dLossDNu = ThorImplementation::Expression(0.5) *
                (halfDegreesOfFreedom.digamma() - halfDegreesOfFreedomPlusOne.digamma() +
                 ThorImplementation::Expression(1.0) / degreesOfFreedom +
                 (residualSquared / degreesOfFreedom).log1p() -
                 degreesOfFreedomPlusOne * residualSquared / (degreesOfFreedom * denominator));
            logDegreesOfFreedomGradient = degreesOfFreedomExcess * dLossDNu;
        }
    }

    if (weighted) {
        ThorImplementation::Expression exampleWeights =
            ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
        locationGradient = locationGradient * exampleWeights;
        logScaleGradient = logScaleGradient * exampleWeights;
        if (logDegreesOfFreedomGradient.has_value())
            logDegreesOfFreedomGradient = logDegreesOfFreedomGradient.value() * exampleWeights;
    }

    ThorImplementation::Expression lossScale(ThorImplementation::Loss::getLossScalingFactor());
    locationGradient = (locationGradient * lossScale).withOutputDType(locationDType);
    logScaleGradient = (logScaleGradient * lossScale).withOutputDType(logScaleDType);

    vector<pair<string, ThorImplementation::Expression>> outputs;
    outputs.emplace_back(kLocationGradientName, locationGradient);
    outputs.emplace_back(kLogScaleGradientName, logScaleGradient);
    if (logDegreesOfFreedomGradient.has_value()) {
        outputs.emplace_back(kLogDegreesOfFreedomGradientName,
                             (logDegreesOfFreedomGradient.value() * lossScale)
                                 .withOutputDType(logDegreesOfFreedomDType.value()));
    }

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs(outputs));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void StudentTNLLLoss::buildSupportLayersAndAddToNetwork() {
    validateFloatingDType("location", predictionsTensor.getDataType());
    validateFloatingDType("log_scale", logScaleTensor.getDataType());
    validateFloatingDType("target", labelsTensor.getDataType());
    if (logDegreesOfFreedomTensor.has_value())
        validateFloatingDType("log_degrees_of_freedom", logDegreesOfFreedomTensor.value().getDataType());
    validateExampleWeights(predictionsTensor, logScaleTensor, labelsTensor, logDegreesOfFreedomTensor, exampleWeightsTensor);
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions() == logScaleTensor.getDimensions());
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions() == labelsTensor.getDimensions());
    if (logDegreesOfFreedomTensor.has_value())
        THOR_THROW_IF_FALSE(predictionsTensor.getDimensions() == logDegreesOfFreedomTensor.value().getDimensions());
    THOR_THROW_IF_FALSE(degreesOfFreedom > 0.0f);
    THOR_THROW_IF_FALSE(std::isfinite(minimumDegreesOfFreedom) && minimumDegreesOfFreedom >= 0.0f);
    if (!logDegreesOfFreedomTensor.has_value())
        THOR_THROW_IF_FALSE(degreesOfFreedom > minimumDegreesOfFreedom);

    MultiInputCustomLoss::Builder builder;
    builder.network(*network)
        .lossExpression(makeStudentTNLLLossExpression(lossDataType,
                                                      logDegreesOfFreedomTensor.has_value(),
                                                      degreesOfFreedom,
                                                      minimumDegreesOfFreedom,
                                                      exampleWeightsTensor.has_value()))
        .gradientExpression(makeStudentTNLLGradientExpression(predictionsTensor.getDataType(),
                                                              logScaleTensor.getDataType(),
                                                              logDegreesOfFreedomTensor.has_value()
                                                                  ? optional<DataType>(logDegreesOfFreedomTensor.value().getDataType())
                                                                  : nullopt,
                                                              degreesOfFreedom,
                                                              minimumDegreesOfFreedom,
                                                              exampleWeightsTensor.has_value()))
        .input(kLocationName, predictionsTensor, std::string(kLocationGradientName))
        .input(kLogScaleName, logScaleTensor, std::string(kLogScaleGradientName));
    if (logDegreesOfFreedomTensor.has_value()) {
        builder.input(kLogDegreesOfFreedomName,
                      logDegreesOfFreedomTensor.value(),
                      std::string(kLogDegreesOfFreedomGradientName));
    }
    builder.auxiliaryInput(kTargetName, labelsTensor)
        .lossName(kLossName)
        .lossDataType(lossDataType)
        .lossWeight(lossWeight.value_or(1.0f))
        .reportsRawLoss();
    if (exampleWeightsTensor.has_value())
        builder.auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value());

    MultiInputCustomLoss rawStudentTNLLLoss = builder.build();
    lossShaperInput = rawStudentTNLLLoss.getLoss();
    finalizeLossReporting();
}

json StudentTNLLLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "student_t_nll_loss";
    j["loss_shape"] = lossShape;
    j["log_scale_tensor"] = logScaleTensor.architectureJson();
    j["minimum_degrees_of_freedom"] = minimumDegreesOfFreedom;
    if (logDegreesOfFreedomTensor.has_value())
        j["log_degrees_of_freedom_tensor"] = logDegreesOfFreedomTensor.value().architectureJson();
    else
        j["degrees_of_freedom"] = degreesOfFreedom;
    return j;
}

void StudentTNLLLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in StudentTNLLLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "student_t_nll_loss")
        throw runtime_error("Layer type mismatch in StudentTNLLLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor location = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor target = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["log_scale_tensor"].at("id").get<uint64_t>();
    Tensor logScale = network->getApiTensorByOriginalId(originalTensorId);

    StudentTNLLLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.predictionsTensor = location;
    loss.labelsTensor = target;
    loss.logScaleTensor = logScale;
    loss.minimumDegreesOfFreedom = j.value("minimum_degrees_of_freedom", 0.0f);
    if (j.contains("log_degrees_of_freedom_tensor")) {
        originalTensorId = j["log_degrees_of_freedom_tensor"].at("id").get<uint64_t>();
        loss.logDegreesOfFreedomTensor = network->getApiTensorByOriginalId(originalTensorId);
    } else {
        loss.degreesOfFreedom = j.value("degrees_of_freedom", 3.0f);
    }
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
    Thor::Loss::register_layer("student_t_nll_loss", &Thor::StudentTNLLLoss::deserialize);
    return true;
}();
}  // namespace
