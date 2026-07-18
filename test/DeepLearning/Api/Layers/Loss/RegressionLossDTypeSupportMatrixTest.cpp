#include "DeepLearning/Api/Layers/Loss/AsymmetricPowerLoss.h"
#include "DeepLearning/Api/Layers/Loss/ExpectileLoss.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Layers/Loss/MeanPowerError.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include "gtest/gtest.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Api = Thor;

namespace {

enum class RegressionLossKind { MSE, MAE, MeanPower, Quantile, Expectile, AsymmetricPower };

const std::vector<RegressionLossKind> kLossKinds = {
    RegressionLossKind::MSE,
    RegressionLossKind::MAE,
    RegressionLossKind::MeanPower,
    RegressionLossKind::Quantile,
    RegressionLossKind::Expectile,
    RegressionLossKind::AsymmetricPower,
};

std::string lossName(RegressionLossKind kind) {
    switch (kind) {
        case RegressionLossKind::MSE:
            return "MSE";
        case RegressionLossKind::MAE:
            return "MAE";
        case RegressionLossKind::MeanPower:
            return "MeanPowerError";
        case RegressionLossKind::Quantile:
            return "QuantileLoss";
        case RegressionLossKind::Expectile:
            return "ExpectileLoss";
        case RegressionLossKind::AsymmetricPower:
            return "AsymmetricPowerLoss";
    }
    return "unknown";
}

std::string dtypeName(Api::DataType dtype) { return ThorImplementation::TensorDescriptor::getElementTypeName(dtype); }

Api::Tensor tensor(Api::DataType dtype, uint64_t width = 4) { return Api::Tensor(dtype, {width}); }

Api::Tensor buildLoss(RegressionLossKind kind,
                      Api::DataType predictionsDType,
                      Api::DataType labelsDType,
                      std::optional<Api::DataType> lossDType = std::nullopt,
                      std::optional<Api::DataType> exampleWeightsDType = std::nullopt) {
    Api::Network network("regression_loss_dtype_" + lossName(kind));
    Api::Tensor predictions = tensor(predictionsDType);
    Api::Tensor labels = tensor(labelsDType);
    std::optional<Api::Tensor> exampleWeights;
    if (exampleWeightsDType.has_value())
        exampleWeights = tensor(exampleWeightsDType.value(), 1);

    switch (kind) {
        case RegressionLossKind::MSE: {
            Api::MSE::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).reportsRawLoss();
            if (lossDType.has_value())
                builder.lossDataType(lossDType.value());
            if (exampleWeights.has_value())
                builder.exampleWeights(exampleWeights.value());
            return builder.build().getLoss();
        }
        case RegressionLossKind::MAE: {
            Api::MAE::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).reportsRawLoss();
            if (lossDType.has_value())
                builder.lossDataType(lossDType.value());
            if (exampleWeights.has_value())
                builder.exampleWeights(exampleWeights.value());
            return builder.build().getLoss();
        }
        case RegressionLossKind::MeanPower: {
            Api::MeanPowerError::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).exponent(1.5f).reportsRawLoss();
            if (lossDType.has_value())
                builder.lossDataType(lossDType.value());
            if (exampleWeights.has_value())
                builder.exampleWeights(exampleWeights.value());
            return builder.build().getLoss();
        }
        case RegressionLossKind::Quantile: {
            Api::QuantileLoss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).quantile(0.1f).reportsRawLoss();
            if (lossDType.has_value())
                builder.lossDataType(lossDType.value());
            if (exampleWeights.has_value())
                builder.exampleWeights(exampleWeights.value());
            return builder.build().getLoss();
        }
        case RegressionLossKind::Expectile: {
            Api::ExpectileLoss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).expectile(0.9f).reportsRawLoss();
            if (lossDType.has_value())
                builder.lossDataType(lossDType.value());
            if (exampleWeights.has_value())
                builder.exampleWeights(exampleWeights.value());
            return builder.build().getLoss();
        }
        case RegressionLossKind::AsymmetricPower: {
            Api::AsymmetricPowerLoss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).level(0.9f).exponent(1.5f).reportsRawLoss();
            if (lossDType.has_value())
                builder.lossDataType(lossDType.value());
            if (exampleWeights.has_value())
                builder.exampleWeights(exampleWeights.value());
            return builder.build().getLoss();
        }
    }
    throw std::runtime_error("Unhandled regression loss kind.");
}

}  // namespace

TEST(RegressionLossDTypeSupportMatrix, AcceptsAllDifferentiableThorFloatingPredictionDTypes) {
    const std::vector<Api::DataType> predictionDTypes = {
        Api::DataType::FP8_E4M3, Api::DataType::FP8_E5M2, Api::DataType::FP16, Api::DataType::BF16, Api::DataType::FP32};

    for (RegressionLossKind kind : kLossKinds) {
        for (Api::DataType predictionDType : predictionDTypes) {
            SCOPED_TRACE(lossName(kind) + " predictions=" + dtypeName(predictionDType));
            EXPECT_NO_THROW(buildLoss(kind, predictionDType, Api::DataType::FP32, Api::DataType::FP32));
        }
    }
}

TEST(RegressionLossDTypeSupportMatrix, AcceptsAllExpressionConvertibleLabelDTypes) {
    const std::vector<Api::DataType> labelDTypes = {
        Api::DataType::BOOLEAN, Api::DataType::INT8,      Api::DataType::INT16, Api::DataType::INT32,
        Api::DataType::INT64,   Api::DataType::UINT8,     Api::DataType::UINT16, Api::DataType::UINT32,
        Api::DataType::UINT64,  Api::DataType::FP8_E4M3, Api::DataType::FP8_E5M2, Api::DataType::FP16,
        Api::DataType::BF16,    Api::DataType::FP32,
    };

    for (RegressionLossKind kind : kLossKinds) {
        for (Api::DataType labelDType : labelDTypes) {
            SCOPED_TRACE(lossName(kind) + " labels=" + dtypeName(labelDType));
            EXPECT_NO_THROW(buildLoss(kind, Api::DataType::BF16, labelDType, Api::DataType::FP32));
        }
    }
}

TEST(RegressionLossDTypeSupportMatrix, AcceptsAllDifferentiableThorFloatingExampleWeightDTypes) {
    const std::vector<Api::DataType> weightDTypes = {
        Api::DataType::FP8_E4M3, Api::DataType::FP8_E5M2, Api::DataType::FP16, Api::DataType::BF16, Api::DataType::FP32};

    for (RegressionLossKind kind : kLossKinds) {
        for (Api::DataType weightDType : weightDTypes) {
            SCOPED_TRACE(lossName(kind) + " example_weights=" + dtypeName(weightDType));
            EXPECT_NO_THROW(buildLoss(kind, Api::DataType::BF16, Api::DataType::FP32, Api::DataType::FP32, weightDType));
        }
    }
}

TEST(RegressionLossDTypeSupportMatrix, DefaultsLowPrecisionLossStorageToFp32) {
    for (RegressionLossKind kind : kLossKinds) {
        for (Api::DataType predictionDType :
             {Api::DataType::FP8_E4M3, Api::DataType::FP8_E5M2, Api::DataType::BF16}) {
            SCOPED_TRACE(lossName(kind) + " predictions=" + dtypeName(predictionDType));
            EXPECT_EQ(buildLoss(kind, predictionDType, Api::DataType::FP32).getDataType(), Api::DataType::FP32);
        }
        EXPECT_EQ(buildLoss(kind, Api::DataType::FP16, Api::DataType::FP32).getDataType(), Api::DataType::FP16);
        EXPECT_EQ(buildLoss(kind, Api::DataType::FP32, Api::DataType::FP32).getDataType(), Api::DataType::FP32);
    }
}

TEST(RegressionLossDTypeSupportMatrix, RejectsNonDifferentiablePredictionsFp64LabelsAndUnsupportedLossStorage) {
    for (RegressionLossKind kind : kLossKinds) {
        SCOPED_TRACE(lossName(kind));
        EXPECT_THROW(buildLoss(kind, Api::DataType::INT32, Api::DataType::FP32, Api::DataType::FP32), std::exception);
        EXPECT_THROW(buildLoss(kind, Api::DataType::FP64, Api::DataType::FP32, Api::DataType::FP32), std::exception);
        EXPECT_THROW(buildLoss(kind, Api::DataType::FP32, Api::DataType::FP64, Api::DataType::FP32), std::exception);
        EXPECT_THROW(buildLoss(kind, Api::DataType::FP32, Api::DataType::FP32, Api::DataType::BF16), std::exception);
        EXPECT_THROW(
            buildLoss(kind, Api::DataType::FP32, Api::DataType::FP32, Api::DataType::FP32, Api::DataType::INT32),
            std::exception);
    }
}
