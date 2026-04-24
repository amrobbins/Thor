// #include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
// #include "Utilities/Expression/Expression.h"
//
// #include "gtest/gtest.h"
//
// using namespace Thor;
// using Expression = ThorImplementation::Expression;
// using DynamicExpression = ThorImplementation::DynamicExpression;
// using DynamicExpressionBuild = ThorImplementation::DynamicExpressionBuild;
// using FusedEquation = ThorImplementation::FusedEquation;
//
// namespace {
// DynamicExpression makeNoOpDynamicExpression() {
//     return DynamicExpression([](const DynamicExpression::TensorMap& inputs,
//                                 const DynamicExpression::TensorMap& outputs,
//                                 Stream& stream) -> DynamicExpressionBuild {
//         (void)inputs;
//
//         auto eq = std::make_shared<FusedEquation>(FusedEquation::compile(Expression::outputs({}).physicalOutputs(), stream.getGpuNum()));
//
//         return DynamicExpressionBuild{
//             eq,
//             {},  // stamp_inputs
//             {},  // tensor_scalar_inputs
//             outputs,
//             {}  // requested_output_shapes
//         };
//     });
// }
// }  // namespace
//
// TEST(CustomLayerApi, NamedConnectionsMapToPortIndicesAndWaitForAllInputs) {
//     Tensor x(Tensor::DataType::FP32, {8});
//     Tensor y(Tensor::DataType::FP32, {8});
//     Tensor sum(Tensor::DataType::FP32, {8});
//     Tensor diff(Tensor::DataType::FP32, {8});
//
//     CustomLayer layer(makeNoOpDynamicExpression());
//
//     ASSERT_TRUE(layer.isInitialized());
//     EXPECT_EQ(layer.getConnectionType(x), 0);
//     EXPECT_EQ(layer.getConnectionType(y), 1);
//     EXPECT_EQ(layer.getConnectionType(sum), 0);
//     EXPECT_EQ(layer.getConnectionType(diff), 1);
//
//     EXPECT_TRUE(layer.mustConnectAllInputsToDriveOutput());
//
//     layer.informThatInputConnectionMade(x);
//     EXPECT_TRUE(layer.getOutputsFromInput(x).empty());
//
//     layer.informThatInputConnectionMade(y);
//     std::vector<Tensor> outputs = layer.getOutputsFromInput(y);
//     ASSERT_EQ(outputs.size(), 2u);
//     EXPECT_EQ(outputs[0], sum);
//     EXPECT_EQ(outputs[1], diff);
// }
