#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Layers/Loss/LossShaper.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;

TEST(LossShaperShapeContract, ApiPerOutputReportingPreservesEveryNonBatchDimension) {
    Api::Tensor rawLoss(Api::DataType::FP32, {2, 3, 4});

    Api::LossShaper batch = Api::LossShaper::Builder().lossInput(rawLoss).reportsBatchLoss().construct();
    Api::LossShaper perExample = Api::LossShaper::Builder().lossInput(rawLoss).reportsPerExampleLoss().construct();
    Api::LossShaper perOutput = Api::LossShaper::Builder().lossInput(rawLoss).reportsPerOutputLoss().construct();

    EXPECT_EQ(batch.getLossOutput().getDimensions(), std::vector<uint64_t>({1}));
    EXPECT_EQ(perExample.getLossOutput().getDimensions(), std::vector<uint64_t>({1}));
    EXPECT_EQ(perOutput.getLossOutput().getDimensions(), std::vector<uint64_t>({2, 3, 4}));
}

TEST(LossShaperShapeContract, ImplementationDimensionsMatchReductionSemantics) {
    const std::vector<uint64_t> rawLossDimensions = {5, 2, 3, 4};

    EXPECT_EQ(Impl::LossShaper::getOutputDimensions(rawLossDimensions, Impl::LossShaper::OutputLossType::BATCH),
              std::vector<uint64_t>({1, 1}));
    EXPECT_EQ(Impl::LossShaper::getOutputDimensions(rawLossDimensions, Impl::LossShaper::OutputLossType::PER_EXAMPLE),
              std::vector<uint64_t>({5, 1}));
    EXPECT_EQ(Impl::LossShaper::getOutputDimensions(rawLossDimensions, Impl::LossShaper::OutputLossType::PER_OUTPUT),
              std::vector<uint64_t>({1, 2, 3, 4}));
}

TEST(LossShaperShapeContract, CubReductionPlansUseOriginalTensorAxes) {
    const std::vector<uint64_t> rawLossDimensions = {5, 2, 3, 4};

    EXPECT_EQ(Impl::LossShaper::getReductionAxes(rawLossDimensions, Impl::LossShaper::OutputLossType::BATCH),
              std::vector<uint32_t>({0, 1, 2, 3}));
    EXPECT_EQ(Impl::LossShaper::getReductionAxes(rawLossDimensions, Impl::LossShaper::OutputLossType::PER_EXAMPLE),
              std::vector<uint32_t>({1, 2, 3}));
    EXPECT_EQ(Impl::LossShaper::getReductionAxes(rawLossDimensions, Impl::LossShaper::OutputLossType::PER_OUTPUT),
              std::vector<uint32_t>({0}));

}

TEST(LossShaperShapeContract, SerializesOnlyCanonicalShapeNames) {
    nlohmann::json apiNone = Api::Loss::LossShape::NONE;
    nlohmann::json apiPerExample = Api::Loss::LossShape::PER_EXAMPLE;
    nlohmann::json apiPerOutput = Api::Loss::LossShape::PER_OUTPUT;
    nlohmann::json implPerExample = Impl::LossShaper::OutputLossType::PER_EXAMPLE;
    nlohmann::json implPerOutput = Impl::LossShaper::OutputLossType::PER_OUTPUT;

    EXPECT_EQ(apiNone, "none");
    EXPECT_EQ(apiPerExample, "per_example");
    EXPECT_EQ(apiPerOutput, "per_output");
    EXPECT_EQ(implPerExample, "per_example");
    EXPECT_EQ(implPerOutput, "per_output");

    EXPECT_EQ(apiNone.get<Api::Loss::LossShape>(), Api::Loss::LossShape::NONE);
    EXPECT_EQ(apiPerExample.get<Api::Loss::LossShape>(), Api::Loss::LossShape::PER_EXAMPLE);
    EXPECT_EQ(apiPerOutput.get<Api::Loss::LossShape>(), Api::Loss::LossShape::PER_OUTPUT);
    EXPECT_EQ(implPerExample.get<Impl::LossShaper::OutputLossType>(), Impl::LossShaper::OutputLossType::PER_EXAMPLE);
    EXPECT_EQ(implPerOutput.get<Impl::LossShaper::OutputLossType>(), Impl::LossShaper::OutputLossType::PER_OUTPUT);
}

TEST(LossShaperShapeContract, RejectsRemovedSerializedShapeNames) {
    EXPECT_THROW(nlohmann::json("elementwise").get<Api::Loss::LossShape>(), std::invalid_argument);
    EXPECT_THROW(nlohmann::json("classwise").get<Api::Loss::LossShape>(), std::invalid_argument);
    EXPECT_THROW(nlohmann::json("elementwise").get<Impl::LossShaper::OutputLossType>(), std::invalid_argument);
    EXPECT_THROW(nlohmann::json("classwise").get<Impl::LossShaper::OutputLossType>(), std::invalid_argument);
}

TEST(LossShaperShapeContract, VectorPerOutputReportingRetainsExistingShape) {
    Api::Tensor rawLoss(Api::DataType::FP32, {100});
    Api::LossShaper perOutput = Api::LossShaper::Builder().lossInput(rawLoss).reportsPerOutputLoss().construct();

    EXPECT_EQ(perOutput.getLossOutput().getDimensions(), std::vector<uint64_t>({100}));
}
