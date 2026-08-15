#include "DeepLearning/Api/Layers/Activations/Elu.h"
#include "DeepLearning/Api/Layers/Activations/Exponential.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"
#include "DeepLearning/Api/Layers/Activations/HardSwish.h"
#include "DeepLearning/Api/Layers/Activations/HardTanh.h"
#include "DeepLearning/Api/Layers/Activations/Mish.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Relu6.h"
#include "DeepLearning/Api/Layers/Activations/Selu.h"
#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"
#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"
#include "DeepLearning/Api/Layers/Activations/SoftSign.h"
#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Api/Layers/Activations/Swiglu.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Activations/Threshold.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "Utilities/Expression/Expression.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

using ThorImplementation::canonicalize;
using ThorImplementation::Expression;

namespace {

string canonicalExpression(const Expression &expression) { return canonicalize(expression.expression()); }

void expectActivationExpression(const Thor::Activation &activation, const Expression &input, const Expression &expected) {
    EXPECT_EQ(canonicalExpression(activation.toExpression(input)), canonicalExpression(expected)) << activation.getLayerType();
}

}  // namespace

TEST(Activations, ToExpressionDispatchesThroughConcreteActivationOverrides) {
    const Expression input = Expression::input("feature_input");
    const Expression zero(0.0);
    const Expression one(1.0);

    Relu relu;
    expectActivationExpression(relu, input, input.max(zero));

    Sigmoid sigmoid;
    expectActivationExpression(sigmoid, input, input.sigmoid());

    Tanh tanh;
    expectActivationExpression(tanh, input, input.tanh());

    HardSigmoid hardSigmoid;
    expectActivationExpression(hardSigmoid, input, ((input * Expression(0.2)) + Expression(0.5)).min(one).max(zero));

    HardSwish hardSwish;
    expectActivationExpression(hardSwish, input, input.hardSwish());

    HardTanh hardTanh(-0.25, 0.75);
    expectActivationExpression(hardTanh, input, input.hardTanh(-0.25, 0.75));

    Mish mish;
    expectActivationExpression(mish, input, input.mish());

    Relu6 relu6;
    expectActivationExpression(relu6, input, input.relu6());

    Threshold threshold(0.25, -1.0);
    expectActivationExpression(threshold, input, input.threshold(0.25, -1.0));

    SoftPlus softPlus;
    expectActivationExpression(softPlus, input, input.softplus());

    SoftSign softSign;
    expectActivationExpression(softSign, input, input / (input.abs() + one));

    Exponential exponential;
    expectActivationExpression(exponential, input, input.exp());

    Gelu gelu;
    expectActivationExpression(gelu, input, input.gelu());

    Selu selu;
    expectActivationExpression(selu, input, input.selu());

    Swish swish;
    expectActivationExpression(swish, input, input.swish());

    Softmax softmax;
    expectActivationExpression(softmax, input, input.softmax());

    Elu elu(0.25f);
    expectActivationExpression(elu, input, input.elu(0.25f));
}


TEST(Activations, ShapePreservingBuildersInferRaggedFromInputAndPreserveOffsets) {
    Network network("ragged_activation_builders");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({3})
                             .maxTotalValues(8)
                             .batchSize(2)
                             .build();

    auto swish = std::dynamic_pointer_cast<Swish>(Swish::Builder().network(network).featureInput(input).build());
    ASSERT_NE(swish, nullptr);
    ASSERT_TRUE(swish->getUseRagged());
    ASSERT_TRUE(swish->getRaggedFeatureInput().has_value());
    ASSERT_TRUE(swish->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(swish->getRaggedFeatureOutput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(swish->getRaggedFeatureOutput()->getValuesDimensions(), input.getValuesDimensions());
    EXPECT_EQ(swish->getFeatureInputs().size(), 2U);
    EXPECT_EQ(swish->getFeatureInputs()[0], input.getValues());
    EXPECT_EQ(swish->getFeatureInputs()[1], input.getOffsets());

    const json j = swish->architectureJson();
    EXPECT_TRUE(j.at("use_ragged").get<bool>());
    EXPECT_EQ(j.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(), input.getOffsets().getId());
    EXPECT_EQ(j.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>(), input.getOffsets().getId());

    auto gelu = std::dynamic_pointer_cast<Gelu>(Gelu::Builder().network(network).featureInput(input).build());
    ASSERT_NE(gelu, nullptr);
    ASSERT_TRUE(gelu->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(gelu->getRaggedFeatureOutput()->getOffsets(), input.getOffsets());

    EXPECT_FALSE(Softmax().supportsRaggedStandalone());
    EXPECT_TRUE(Swiglu().supportsRaggedStandalone());
}
