#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using std::shared_ptr;

using namespace Thor;

TEST(LossShaper, Builds) {
    srand(time(nullptr));

    for (uint32_t i = 0; i < 10; ++i) {
        Network network;

        vector<uint64_t> dimensions;
        uint32_t numDimensions = 2;
        for (uint32_t j = 0; j < numDimensions; ++j)
            dimensions.push_back(2 + (rand() % 1000));
        Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor lossInput(dataType, dimensions);

        LossShaper::Builder lossShaperBuilder = LossShaper::Builder().network(network).lossInput(lossInput);

        uint32_t shape = rand() % 3;
        ThorImplementation::LossShaper::OutputLossType outputLossType;
        if (shape == 0) {
            outputLossType = ThorImplementation::LossShaper::OutputLossType::BATCH;
            lossShaperBuilder.reportsBatchLoss();
        } else if (shape == 1) {
            outputLossType = ThorImplementation::LossShaper::OutputLossType::CLASSWISE;
            lossShaperBuilder.reportsClasswiseLoss();
        } else if (shape == 2) {
            outputLossType = ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE;
            lossShaperBuilder.reportsElementwiseLoss();
        } else {
            assert(false);
        }
        vector<uint64_t> batchDimensions = {1};
        vector<uint64_t> classwiseDimensions = {dimensions[1]};
        vector<uint64_t> elementwiseDimensions = {dimensions[0]};

        LossShaper lossShaper = lossShaperBuilder.build();

        ASSERT_TRUE(lossShaper.isInitialized());

        Optional<Tensor> actualLossInput = lossShaper.getLossInput();
        ASSERT_TRUE(actualLossInput.isPresent());
        ASSERT_EQ(actualLossInput.get().getDataType(), dataType);
        ASSERT_EQ(actualLossInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualLossOutput = lossShaper.getLossOutput();
        ASSERT_TRUE(actualLossOutput.isPresent());
        ASSERT_EQ(actualLossOutput.get().getDataType(), dataType);
        if (outputLossType == ThorImplementation::LossShaper::OutputLossType::BATCH) {
            ASSERT_EQ(actualLossOutput.get().getDimensions(), batchDimensions);
        } else if (outputLossType == ThorImplementation::LossShaper::OutputLossType::CLASSWISE) {
            ASSERT_EQ(actualLossOutput.get().getDimensions(), classwiseDimensions);
        } else if (outputLossType == ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE) {
            ASSERT_EQ(actualLossOutput.get().getDimensions(), elementwiseDimensions);
        }

        ASSERT_FALSE(actualLossInput.get() == actualLossOutput.get());

        ASSERT_TRUE(lossShaper.getLossInput() == lossShaper.getFeatureInput());
        ASSERT_TRUE(lossShaper.getLossOutput() == lossShaper.getFeatureOutput());

        shared_ptr<Layer> cloneLayer = lossShaper.clone();
        LossShaper *clone = dynamic_cast<LossShaper *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(lossShaper.isInitialized());

        Optional<Tensor> cloneLossInput = clone->getLossInput();
        ASSERT_TRUE(cloneLossInput.isPresent());
        ASSERT_EQ(cloneLossInput.get().getDataType(), dataType);
        ASSERT_EQ(cloneLossInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLossOutput = clone->getLossOutput();
        ASSERT_TRUE(cloneLossOutput.isPresent());
        ASSERT_EQ(cloneLossOutput.get().getDataType(), dataType);
        if (outputLossType == ThorImplementation::LossShaper::OutputLossType::BATCH) {
            ASSERT_EQ(cloneLossOutput.get().getDimensions(), batchDimensions);
        } else if (outputLossType == ThorImplementation::LossShaper::OutputLossType::CLASSWISE) {
            ASSERT_EQ(cloneLossOutput.get().getDimensions(), classwiseDimensions);
        } else if (outputLossType == ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE) {
            ASSERT_EQ(cloneLossOutput.get().getDimensions(), elementwiseDimensions);
        }

        ASSERT_FALSE(cloneLossInput.get() == cloneLossOutput.get());

        ASSERT_TRUE(clone->getLossInput() == clone->getFeatureInput());
        ASSERT_TRUE(clone->getLossOutput() == clone->getFeatureOutput());

        ASSERT_EQ(lossShaper.getId(), clone->getId());
        ASSERT_GT(lossShaper.getId(), 1u);

        ASSERT_TRUE(lossShaper == *clone);
        ASSERT_FALSE(lossShaper != *clone);
        ASSERT_FALSE(lossShaper > *clone);
        ASSERT_FALSE(lossShaper < *clone);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
