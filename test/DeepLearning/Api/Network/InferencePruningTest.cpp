#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/Expression.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

class InferencePruningTestNetwork : public Api::Network {
   public:
    using Api::Network::Network;

    void markLoadedFromArchiveForTest() { loadedFromArchive = true; }

    Api::Network::StatusCode evaluateGraphForTest(bool inferenceOnly) { return evaluateGraph(inferenceOnly); }

    bool containsTensorForTest(const Api::Tensor& tensor) const { return allTensors.count(tensor) != 0; }
};

Impl::DynamicExpression makeTwoOutputExpression() {
    Impl::Expression x = Impl::Expression::input("x", Impl::DataType::FP32, Impl::DataType::FP32);
    Impl::Expression prediction = x + 1.0f;
    Impl::Expression lossOnly = x * 2.0f;
    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(
        Impl::Expression::outputs({{"prediction", prediction}, {"loss_only", lossOnly}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

TEST(NetworkInferencePruning, KeepsOnlyLiveOutputPortsFromMultiOutputLayer) {
    InferencePruningTestNetwork network("multi_output_inference_pruning");

    Api::NetworkInput features = Api::NetworkInput::Builder()
                                     .network(network)
                                     .name("features")
                                     .dimensions({4})
                                     .dataType(Impl::DataType::FP32)
                                     .build();
    Api::NetworkInput labels = Api::NetworkInput::Builder()
                                   .network(network)
                                   .name("labels")
                                   .dimensions({4})
                                   .dataType(Impl::DataType::FP32)
                                   .build();

    Api::CustomLayer multiOutput = Api::CustomLayer::Builder()
                                       .network(network)
                                       .expression(makeTwoOutputExpression())
                                       .inputInterface({{"x", features.getFeatureOutput().value()}})
                                       .build();
    const Api::Tensor prediction = multiOutput.getOutput("prediction");
    const Api::Tensor lossOnly = multiOutput.getOutput("loss_only");

    Api::MSE trainingLoss = Api::MSE::Builder()
                                .network(network)
                                .predictions(lossOnly)
                                .labels(labels.getFeatureOutput().value())
                                .lossDataType(Impl::DataType::FP32)
                                .reportsBatchLoss()
                                .build();

    (void)Api::NetworkOutput::Builder()
        .network(network)
        .name("prediction")
        .inputTensor(prediction)
        .dataType(Impl::DataType::FP32)
        .build();
    (void)Api::NetworkOutput::Builder()
        .network(network)
        .name("training_loss")
        .inputTensor(trainingLoss.getLoss())
        .dataType(Impl::DataType::FP32)
        .build();

    ASSERT_EQ(network.evaluateGraphForTest(/*inferenceOnly=*/false), Api::Network::StatusCode::SUCCESS);

    network.markLoadedFromArchiveForTest();
    EXPECT_EQ(network.getRequiredNetworkInputNamesForOutputs({"prediction"}, /*inferenceOnly=*/true),
              (std::vector<std::string>{"features"}));

    EXPECT_TRUE(network.containsTensorForTest(prediction));
    EXPECT_FALSE(network.containsTensorForTest(lossOnly));
    EXPECT_FALSE(network.containsTensorForTest(labels.getFeatureOutput().value()));
}
