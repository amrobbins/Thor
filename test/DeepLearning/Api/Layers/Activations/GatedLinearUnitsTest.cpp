#include "DeepLearning/Api/Layers/Activations/BilinearGlu.h"
#include "DeepLearning/Api/Layers/Activations/Geglu.h"
#include "DeepLearning/Api/Layers/Activations/Glu.h"
#include "DeepLearning/Api/Layers/Activations/Reglu.h"
#include "DeepLearning/Api/Layers/Activations/Swiglu.h"
#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/Expression.h"

#include "gtest/gtest.h"

#include <memory>
#include <vector>

using namespace Thor;
using namespace std;

namespace {

template <typename ActivationT>
shared_ptr<ActivationT> buildActivation(Network& network, Tensor input) {
    typename ActivationT::Builder builder;
    shared_ptr<Activation> base = builder.network(network).featureInput(input).build();
    shared_ptr<ActivationT> activation = dynamic_pointer_cast<ActivationT>(base);
    EXPECT_NE(activation, nullptr);
    return activation;
}

template <typename ActivationT>
void expectGatedActivationBuilds(const string& expectedLayerType, const string& expectedSerializedType) {
    Network network("testNetwork");
    Tensor featureInput(DataType::FP32, {2, 3, 10});

    shared_ptr<ActivationT> activation = buildActivation<ActivationT>(network, featureInput);
    ASSERT_NE(activation, nullptr);
    ASSERT_TRUE(activation->isInitialized());
    ASSERT_EQ(activation->getLayerType(), expectedLayerType);

    ASSERT_TRUE(activation->getFeatureInput().has_value());
    ASSERT_EQ(activation->getFeatureInput().value(), featureInput);

    ASSERT_TRUE(activation->getFeatureOutput().has_value());
    ASSERT_EQ(activation->getFeatureOutput().value().getDataType(), DataType::FP32);
    ASSERT_EQ(activation->getFeatureOutput().value().getDimensions(), (vector<uint64_t>{2, 3, 5}));

    const auto json = activation->architectureJson();
    ASSERT_EQ(json.at("layer_type").template get<string>(), expectedSerializedType);

    ThorImplementation::Expression input = ThorImplementation::Expression::input("feature_input");
    EXPECT_NO_THROW((void)activation->toExpression(input));
}

}  // namespace

TEST(GatedLinearUnits, BuildHalvesFinalFeatureDimension) {
    expectGatedActivationBuilds<Glu>("Glu", "glu");
    expectGatedActivationBuilds<Reglu>("Reglu", "reglu");
    expectGatedActivationBuilds<Geglu>("Geglu", "geglu");
    expectGatedActivationBuilds<Swiglu>("Swiglu", "swiglu");
    expectGatedActivationBuilds<BilinearGlu>("BilinearGlu", "bilinear_glu");
}

TEST(GatedLinearUnits, RejectOddFinalFeatureDimension) {
    Network network("testNetwork");
    Tensor featureInput(DataType::FP32, {2, 3, 9});

    Glu::Builder builder;
    EXPECT_ANY_THROW((void)builder.network(network).featureInput(featureInput).build());
}
