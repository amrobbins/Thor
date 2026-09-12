#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Layers/Utility/Transpose.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

template <typename BuilderT, typename EpilogueT>
concept AcceptsEpilogue = requires(BuilderT& builder, const EpilogueT& epilogue) {
    builder.epilogue(epilogue);
};

static_assert(AcceptsEpilogue<Thor::RMSNorm::Builder, ThorImplementation::Expression>);
static_assert(AcceptsEpilogue<Thor::Transpose::Builder, ThorImplementation::Expression>);

// Epilogues are expressions.  Activations may be converted to expressions by the
// caller, but are deliberately not accepted as a second epilogue API surface.
static_assert(!AcceptsEpilogue<Thor::RMSNorm::Builder, Thor::Swish>);
static_assert(!AcceptsEpilogue<Thor::Transpose::Builder, Thor::Swish>);
static_assert(!AcceptsEpilogue<Thor::RMSNorm::Builder, std::shared_ptr<Thor::Swish>>);
static_assert(!AcceptsEpilogue<Thor::Transpose::Builder, std::shared_ptr<Thor::Swish>>);

}  // namespace

TEST(LayerEpilogueApi, EpilogueSurfaceIsExpressionOnly) {
    SUCCEED();
}

TEST(LayerEpilogueApi, ExpressionDefinitionRoundTripUsesGraphScopedPhysicalImport) {
    using ThorImplementation::DataType;
    using ThorImplementation::Expression;

    const Expression input = Thor::LayerEpilogue::input("epilogue_input", DataType::FP32, DataType::FP32);
    const Expression authored = input.sin().exp();
    const ThorImplementation::ExpressionDefinition definition =
        Thor::LayerEpilogue::makeDefinition(authored, "epilogue_input", "epilogue_output", "TestLayer");

    const Expression imported = Thor::LayerEpilogue::expressionFromDefinition(
        definition, "epilogue_input", "epilogue_output", "TestLayer");

    EXPECT_TRUE(Thor::LayerEpilogue::hasSameCanonicalForm(
        authored, imported, "epilogue_input", "epilogue_output", "TestLayer"));
}
