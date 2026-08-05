#include "Utilities/TensorOperations/Einsum/EinsumParser.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

std::vector<int32_t> labels(const std::string& text) {
    std::vector<int32_t> result;
    result.reserve(text.size());
    for (char c : text) {
        result.push_back(EinsumParser::labelId(c));
    }
    return result;
}

}  // namespace

TEST(EinsumParser, ParsesExplicitEquationAndEllipsisPosition) {
    const EinsumEquation equation = EinsumParser::parse("  ...ik, ...kj -> ...ij ");

    ASSERT_EQ(equation.inputs.size(), 2u);
    EXPECT_TRUE(equation.explicit_output);
    EXPECT_EQ(equation.inputs[0].labels, labels("ik"));
    ASSERT_TRUE(equation.inputs[0].ellipsis_position.has_value());
    EXPECT_EQ(*equation.inputs[0].ellipsis_position, 0u);
    EXPECT_EQ(equation.inputs[1].labels, labels("kj"));
    ASSERT_TRUE(equation.output.ellipsis_position.has_value());
    EXPECT_EQ(*equation.output.ellipsis_position, 0u);
    EXPECT_EQ(equation.output.labels, labels("ij"));
}

TEST(EinsumParser, SynthesizesNumpyCompatibleImplicitOutput) {
    const EinsumEquation equation = EinsumParser::parse("ba,ac");

    EXPECT_FALSE(equation.explicit_output);
    EXPECT_FALSE(equation.output.ellipsis_position.has_value());
    EXPECT_EQ(equation.output.labels, labels("bc"));
}

TEST(EinsumParser, ImplicitOutputPlacesEllipsisBeforeSingletonLabels) {
    const EinsumEquation equation = EinsumParser::parse("z...ba,ac");

    ASSERT_TRUE(equation.output.ellipsis_position.has_value());
    EXPECT_EQ(*equation.output.ellipsis_position, 0u);
    EXPECT_EQ(equation.output.labels, labels("bcz"));
}

TEST(EinsumParser, ResolvesMatrixMultiplyAndReductionLabel) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve("ik,kj->ij", {{2, 3}, {3, 4}});

    ASSERT_EQ(equation.inputs.size(), 2u);
    EXPECT_EQ(equation.inputs[0].axis_labels, labels("ik"));
    EXPECT_EQ(equation.inputs[1].axis_labels, labels("kj"));
    EXPECT_EQ(equation.output_labels, labels("ij"));
    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{2, 4}));
    EXPECT_EQ(equation.reduction_labels, labels("k"));
}

TEST(EinsumParser, ResolvesExplicitOutputPermutation) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve("ij->ji", {{2, 3}});
    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{3, 2}));
}

TEST(EinsumParser, ResolvesImplicitOutputInAlphabeticalOrder) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve("ji", {{2, 3}});

    EXPECT_EQ(equation.output_labels, labels("ij"));
    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{3, 2}));
}


TEST(EinsumParser, ImplicitOutputUsesAsciiAlphabeticalLabelOrder) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve("aA", {{2, 3}});

    EXPECT_EQ(equation.output_labels, labels("Aa"));
    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{3, 2}));
}

TEST(EinsumParser, RepeatedInputLabelRepresentsDiagonal) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve("ii->i", {{4, 4}});

    EXPECT_EQ(equation.inputs[0].axis_labels, labels("ii"));
    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{4}));
    EXPECT_TRUE(equation.reduction_labels.empty());
}

TEST(EinsumParser, RepeatedInputLabelCanBeReducedToScalar) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve("ii", {{4, 4}});

    EXPECT_TRUE(equation.output_labels.empty());
    EXPECT_TRUE(equation.output_dimensions.empty());
    EXPECT_EQ(equation.reduction_labels, labels("i"));
}

TEST(EinsumParser, RepeatedInputLabelRequiresEqualDimensionsWithinOperand) {
    EXPECT_THROW((void)EinsumParser::parseAndResolve("ii->i", {{1, 3}}), std::invalid_argument);
}

TEST(EinsumParser, SameLabelBroadcastsAcrossOperands) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve("ij,jk->ik", {{2, 1}, {3, 4}});

    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{2, 4}));
    EXPECT_EQ(equation.label_dimensions.at(EinsumParser::labelId('j')), 3u);
}

TEST(EinsumParser, SameLabelRejectsIncompatibleDimensionsAcrossOperands) {
    EXPECT_THROW((void)EinsumParser::parseAndResolve("ij,jk->ik", {{2, 2}, {3, 4}}), std::invalid_argument);
}

TEST(EinsumParser, ResolvesBatchedEllipsisBroadcast) {
    const ResolvedEinsumEquation equation =
        EinsumParser::parseAndResolve("...ik,...kj->...ij", {{5, 2, 3}, {1, 3, 4}});

    EXPECT_EQ(equation.ellipsis_rank, 1u);
    EXPECT_EQ(equation.inputs[0].ellipsis_rank, 1u);
    EXPECT_EQ(equation.inputs[1].ellipsis_rank, 1u);
    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{5, 2, 4}));
    ASSERT_EQ(equation.output_labels.size(), 3u);
    EXPECT_TRUE(EinsumParser::isEllipsisLabel(equation.output_labels[0]));
}


TEST(EinsumParser, ResolvesEllipsisAtArbitrarySubscriptPosition) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve("i...j->...ij", {{2, 3, 4, 5}});

    EXPECT_EQ(equation.ellipsis_rank, 2u);
    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{3, 4, 2, 5}));
}

TEST(EinsumParser, EllipsesOfDifferentRanksAreRightAligned) {
    const ResolvedEinsumEquation equation =
        EinsumParser::parseAndResolve("...i,...i->...i", {{2, 3, 4}, {3, 4}});

    EXPECT_EQ(equation.ellipsis_rank, 2u);
    EXPECT_EQ(equation.inputs[0].ellipsis_rank, 2u);
    EXPECT_EQ(equation.inputs[1].ellipsis_rank, 1u);
    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{2, 3, 4}));

    ASSERT_EQ(equation.inputs[1].axis_labels.size(), 2u);
    EXPECT_EQ(equation.inputs[1].axis_labels[0], EinsumParser::kEllipsisLabelBase + 1);
    EXPECT_EQ(equation.inputs[1].axis_labels[1], EinsumParser::labelId('i'));
}

TEST(EinsumParser, EllipsisBroadcastRejectsIncompatibleDimensions) {
    EXPECT_THROW((void)EinsumParser::parseAndResolve("...i,...i->...i", {{2, 3, 4}, {5, 3, 4}}), std::invalid_argument);
}

TEST(EinsumParser, ExplicitOutputMustRetainNonEmptyEllipsisForNumpyCompatibility) {
    EXPECT_THROW((void)EinsumParser::parseAndResolve("...i->i", {{2, 3}}), std::invalid_argument);
}

TEST(EinsumParser, ZeroRankOutputEllipsisIsAllowed) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve("ij->...ji", {{2, 3}});

    EXPECT_EQ(equation.ellipsis_rank, 0u);
    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{3, 2}));
}

TEST(EinsumParser, ExplicitReductionCanProduceScalarShape) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve("ij->", {{2, 3}});

    EXPECT_TRUE(equation.output_dimensions.empty());
    EXPECT_EQ(equation.reduction_labels, labels("ij"));
}

TEST(EinsumParser, EmptySubscriptSupportsZeroRankOperandSyntax) {
    const ResolvedEinsumEquation equation = EinsumParser::parseAndResolve(",i->i", {{}, {3}});

    ASSERT_EQ(equation.inputs.size(), 2u);
    EXPECT_TRUE(equation.inputs[0].axis_labels.empty());
    EXPECT_EQ(equation.output_dimensions, (std::vector<uint64_t>{3}));
}

TEST(EinsumParser, RejectsOperandRankMismatchWithoutEllipsis) {
    EXPECT_THROW((void)EinsumParser::parseAndResolve("ij,jk->ik", {{2, 3, 4}, {3, 4}}), std::invalid_argument);
}

TEST(EinsumParser, RejectsOperandRankBelowNamedAxesWithEllipsis) {
    EXPECT_THROW((void)EinsumParser::parseAndResolve("...ij->...ij", {{3}}), std::invalid_argument);
}

TEST(EinsumParser, RejectsOperandCountMismatch) {
    const EinsumEquation equation = EinsumParser::parse("ij,jk->ik");
    EXPECT_THROW((void)EinsumParser::resolve(equation, {{2, 3}}), std::invalid_argument);
}

TEST(EinsumParser, RejectsZeroSizedThorDimension) {
    EXPECT_THROW((void)EinsumParser::parseAndResolve("ij->ij", {{2, 0}}), std::invalid_argument);
}

TEST(EinsumParser, RejectsOutputLabelAbsentFromInputs) {
    EXPECT_THROW((void)EinsumParser::parse("ij->ik"), std::invalid_argument);
}

TEST(EinsumParser, RejectsRepeatedOutputLabel) {
    EXPECT_THROW((void)EinsumParser::parse("ij->ii"), std::invalid_argument);
}

TEST(EinsumParser, RejectsMalformedEquations) {
    const std::vector<std::string> invalid_equations = {
        "i..j->ij",
        "i....j->ij",
        "i...j...->ij",
        "ij->i->j",
        "ij->>ij",
        "ij;j->i",
        "ij->i,j",
        "ij->i1",
    };

    for (const std::string& equation : invalid_equations) {
        EXPECT_THROW((void)EinsumParser::parse(equation), std::invalid_argument) << equation;
    }
}
