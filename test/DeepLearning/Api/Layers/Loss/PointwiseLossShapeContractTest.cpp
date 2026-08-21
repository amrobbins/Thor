#include "DeepLearning/Api/Layers/Loss/AsymmetricPowerLoss.h"
#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"
#include "DeepLearning/Api/Layers/Loss/BinaryFocalLoss.h"
#include "DeepLearning/Api/Layers/Loss/ExpectileLoss.h"
#include "DeepLearning/Api/Layers/Loss/GammaNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/GaussianNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/LaplaceNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/StudentTNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/HuberLoss.h"
#include "DeepLearning/Api/Layers/Loss/PoissonNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/NegativeBinomialNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.h"
#include "DeepLearning/Api/Layers/Loss/MeanPowerError.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"
#include "DeepLearning/Api/Layers/Loss/SmoothL1Loss.h"
#include "DeepLearning/Api/Layers/Loss/TweedieLoss.h"
#include "DeepLearning/Api/Layers/Loss/ContrastiveLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace Api = Thor;

namespace {

class GraphValidationTestNetwork : public Api::Network {
   public:
    using Api::Network::Network;

    Api::Network::StatusCode evaluateGraphForTest(bool inferenceOnly) { return evaluateGraph(inferenceOnly); }
};

const std::vector<uint64_t> kTensorDimensions = {2, 3, 4};

template <typename BuildRawLoss>
void expectMultidimensionalRawLoss(const std::string& networkName, BuildRawLoss buildRawLoss) {
    Api::Network network(networkName);
    Api::Tensor predictions(Api::DataType::FP32, kTensorDimensions);
    Api::Tensor labels(Api::DataType::FP32, kTensorDimensions);

    const Api::Tensor rawLoss = buildRawLoss(network, predictions, labels);
    EXPECT_EQ(rawLoss.getDimensions(), kTensorDimensions);
}

template <typename BuildRawLoss>
void expectMultidimensionalThreeInputRawLoss(const std::string& networkName, BuildRawLoss buildRawLoss) {
    Api::Network network(networkName);
    Api::Tensor predictions(Api::DataType::FP32, kTensorDimensions);
    Api::Tensor labels(Api::DataType::FP32, kTensorDimensions);
    Api::Tensor auxiliary(Api::DataType::FP32, kTensorDimensions);

    const Api::Tensor rawLoss = buildRawLoss(network, predictions, labels, auxiliary);
    EXPECT_EQ(rawLoss.getDimensions(), kTensorDimensions);
}

}  // namespace

TEST(PointwiseLossShapeContract, CorePointwiseLossesAcceptMultidimensionalPredictionsAndLabels) {
    expectMultidimensionalRawLoss(
        "multidimensional_bce", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::BinaryCrossEntropy::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_binary_focal", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::BinaryFocalLoss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_mape", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::MAPE::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss("multidimensional_mse", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
        return Api::MSE::Builder().network(network).predictions(predictions).labels(labels).reportsRawLoss().build().getLoss();
    });

    expectMultidimensionalRawLoss("multidimensional_mae", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
        return Api::MAE::Builder().network(network).predictions(predictions).labels(labels).reportsRawLoss().build().getLoss();
    });

    expectMultidimensionalRawLoss(
        "multidimensional_mean_power", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::MeanPowerError::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_expectile", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::ExpectileLoss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_quantile", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::QuantileLoss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_asymmetric_power", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::AsymmetricPowerLoss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_huber", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::HuberLoss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_smooth_l1", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::SmoothL1Loss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_poisson_nll", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::PoissonNLLLoss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_gamma_nll", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::GammaNLLLoss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_tweedie", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::TweedieLoss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalRawLoss(
        "multidimensional_contrastive", [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels) {
            return Api::ContrastiveLoss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalThreeInputRawLoss(
        "multidimensional_gaussian_nll",
        [](Api::Network& network, Api::Tensor predictions, Api::Tensor labels, Api::Tensor variance) {
            return Api::GaussianNLLLoss::Builder()
                .network(network)
                .predictions(predictions)
                .labels(labels)
                .variance(variance)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalThreeInputRawLoss(
        "multidimensional_laplace_nll",
        [](Api::Network& network, Api::Tensor location, Api::Tensor labels, Api::Tensor scale) {
            return Api::LaplaceNLLLoss::Builder()
                .network(network)
                .location(location)
                .scale(scale)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalThreeInputRawLoss(
        "multidimensional_student_t_nll",
        [](Api::Network& network, Api::Tensor location, Api::Tensor labels, Api::Tensor logScale) {
            return Api::StudentTNLLLoss::Builder()
                .network(network)
                .location(location)
                .logScale(logScale)
                .labels(labels)
                .degreesOfFreedom(4.0f)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectMultidimensionalThreeInputRawLoss(
        "multidimensional_negative_binomial_nll",
        [](Api::Network& network, Api::Tensor mean, Api::Tensor labels, Api::Tensor dispersion) {
            return Api::NegativeBinomialNLLLoss::Builder()
                .network(network)
                .mean(mean)
                .dispersion(dispersion)
                .labels(labels)
                .reportsRawLoss()
                .build()
                .getLoss();
        });
}

TEST(PointwiseLossShapeContract, BinaryCrossEntropySupportsAllCanonicalReportingShapes) {
    const std::vector<uint64_t> dimensions = {2, 3, 4};

    {
        Api::Network network("multidimensional_bce_batch");
        Api::Tensor predictions(Api::DataType::FP32, dimensions);
        Api::Tensor labels(Api::DataType::FP32, dimensions);
        Api::BinaryCrossEntropy loss =
            Api::BinaryCrossEntropy::Builder().network(network).predictions(predictions).labels(labels).reportsBatchLoss().build();
        EXPECT_EQ(loss.getLoss().getDimensions(), std::vector<uint64_t>({1}));
    }

    {
        Api::Network network("multidimensional_bce_per_example");
        Api::Tensor predictions(Api::DataType::FP32, dimensions);
        Api::Tensor labels(Api::DataType::FP32, dimensions);
        Api::BinaryCrossEntropy loss = Api::BinaryCrossEntropy::Builder()
                                                   .network(network)
                                                   .predictions(predictions)
                                                   .labels(labels)
                                                   .reportsPerExampleLoss()
                                                   .build();
        EXPECT_EQ(loss.getLoss().getDimensions(), std::vector<uint64_t>({1}));
    }

    {
        Api::Network network("multidimensional_bce_per_output");
        Api::Tensor predictions(Api::DataType::FP32, dimensions);
        Api::Tensor labels(Api::DataType::FP32, dimensions);
        Api::BinaryCrossEntropy loss = Api::BinaryCrossEntropy::Builder()
                                                   .network(network)
                                                   .predictions(predictions)
                                                   .labels(labels)
                                                   .reportsPerOutputLoss()
                                                   .build();
        EXPECT_EQ(loss.getLoss().getDimensions(), dimensions);
    }

    {
        Api::Network network("multidimensional_bce_raw");
        Api::Tensor predictions(Api::DataType::FP32, dimensions);
        Api::Tensor labels(Api::DataType::FP32, dimensions);
        Api::BinaryCrossEntropy loss =
            Api::BinaryCrossEntropy::Builder().network(network).predictions(predictions).labels(labels).reportsRawLoss().build();
        EXPECT_EQ(loss.getLoss().getDimensions(), dimensions);
    }
}

TEST(PointwiseLossShapeContract, ReportingReductionsPreservePerOutputTensorLayout) {
    const std::vector<uint64_t> dimensions = {2, 3, 4};

    {
        Api::Network network("multidimensional_mse_batch");
        Api::Tensor predictions(Api::DataType::FP32, dimensions);
        Api::Tensor labels(Api::DataType::FP32, dimensions);
        Api::MSE loss = Api::MSE::Builder().network(network).predictions(predictions).labels(labels).reportsBatchLoss().build();
        EXPECT_EQ(loss.getLoss().getDimensions(), std::vector<uint64_t>({1}));
    }

    {
        Api::Network network("multidimensional_mse_per_example");
        Api::Tensor predictions(Api::DataType::FP32, dimensions);
        Api::Tensor labels(Api::DataType::FP32, dimensions);
        Api::MSE loss = Api::MSE::Builder().network(network).predictions(predictions).labels(labels).reportsPerExampleLoss().build();
        EXPECT_EQ(loss.getLoss().getDimensions(), std::vector<uint64_t>({1}));
    }

    {
        Api::Network network("multidimensional_mse_per_output");
        Api::Tensor predictions(Api::DataType::FP32, dimensions);
        Api::Tensor labels(Api::DataType::FP32, dimensions);
        Api::MSE loss = Api::MSE::Builder().network(network).predictions(predictions).labels(labels).reportsPerOutputLoss().build();
        EXPECT_EQ(loss.getLoss().getDimensions(), dimensions);
    }
}

TEST(PointwiseLossShapeContract, StillRequiresExactPredictionAndLabelShapes) {
    Api::Network network("pointwise_shape_mismatch");
    Api::Tensor predictions(Api::DataType::FP32, {2, 3});
    Api::Tensor labels(Api::DataType::FP32, {6});

    EXPECT_THROW(Api::MSE::Builder().network(network).predictions(predictions).labels(labels).build(), std::logic_error);
}

TEST(PointwiseLossShapeContract, NoReportingKeepsRawTrainingRootAndAvoidsDanglingOutput) {
    GraphValidationTestNetwork network("mse_no_report");
    Api::NetworkInput predictions = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("predictions")
                                        .dimensions({3})
                                        .dataType(Api::DataType::FP32)
                                        .build();
    Api::NetworkInput labels = Api::NetworkInput::Builder()
                                   .network(network)
                                   .name("labels")
                                   .dimensions({3})
                                   .dataType(Api::DataType::FP32)
                                   .build();

    Api::MSE loss = Api::MSE::Builder()
                        .network(network)
                        .predictions(predictions.getFeatureOutput().value())
                        .labels(labels.getFeatureOutput().value())
                        .reportsNoLoss()
                        .build();

    EXPECT_FALSE(loss.reportsLoss());
    EXPECT_THROW((void)loss.getLoss(), std::runtime_error);
    ASSERT_TRUE(loss.getRawLoss().isInitialized());

    const std::vector<Api::Tensor> roots = network.getLossRootTensors();
    ASSERT_EQ(roots.size(), 1u);
    EXPECT_EQ(roots[0].getOriginalId(), loss.getRawLoss().getOriginalId());
    EXPECT_EQ(network.evaluateGraphForTest(false), Api::Network::StatusCode::SUCCESS);
}

TEST(PointwiseLossShapeContract, RequestedRawReportStillMustBeConsumed) {
    GraphValidationTestNetwork network("mse_raw_report_must_be_consumed");
    Api::NetworkInput predictions = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("predictions")
                                        .dimensions({3})
                                        .dataType(Api::DataType::FP32)
                                        .build();
    Api::NetworkInput labels = Api::NetworkInput::Builder()
                                   .network(network)
                                   .name("labels")
                                   .dimensions({3})
                                   .dataType(Api::DataType::FP32)
                                   .build();

    Api::MSE loss = Api::MSE::Builder()
                        .network(network)
                        .predictions(predictions.getFeatureOutput().value())
                        .labels(labels.getFeatureOutput().value())
                        .reportsRawLoss()
                        .build();

    EXPECT_TRUE(loss.reportsLoss());
    EXPECT_TRUE(loss.getLoss().isInitialized());
    EXPECT_EQ(network.evaluateGraphForTest(false), Api::Network::StatusCode::DANGLING_OUTPUT);
}
