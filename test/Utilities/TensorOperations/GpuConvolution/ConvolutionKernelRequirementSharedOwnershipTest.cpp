#include "Utilities/TensorOperations/GpuConvolution/ConvolutionKernelRequirement.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

namespace {

ConvolutionKernelRequirement makeRequirement() {
    return ConvolutionKernelRequirement("shared-ownership-test-gpu",
                                        /*filterWidth=*/3,
                                        /*filterHeight=*/3,
                                        /*filterHorizontalStride=*/1,
                                        /*filterVerticalStride=*/1,
                                        /*leftAndRightPadWidth=*/1,
                                        /*topAndBottomPadHeight=*/1,
                                        /*numInputChannels=*/4,
                                        /*numOutputChannels=*/8,
                                        /*batchSize=*/2,
                                        /*numInputColumns=*/7,
                                        /*numInputRows=*/5);
}

}  // namespace

TEST(ConvolutionKernelRequirementSharedOwnership, CopiesAndMovesShareDescriptorState) {
    ConvolutionKernelRequirement requirement = makeRequirement();

    const cudnnConvolutionDescriptor_t convolutionDescriptor = requirement.getConvolutionDescriptor();
    const cudnnFilterDescriptor_t filterDescriptor = requirement.getWeightsFilterDescriptor();
    const cudnnTensorDescriptor_t inputDescriptor = requirement.getDataInputTensorDescriptor();
    const cudnnTensorDescriptor_t outputDescriptor = requirement.getDataOutputTensorDescriptor();
    const cudnnTensorDescriptor_t biasesDescriptor = requirement.getBiasesTensorDescriptor();
    const size_t hash = std::hash<ConvolutionKernelRequirement>()(requirement);

    ConvolutionKernelRequirement copy = requirement;
    EXPECT_TRUE(copy == requirement);
    EXPECT_EQ(copy.getConvolutionDescriptor(), convolutionDescriptor);
    EXPECT_EQ(copy.getWeightsFilterDescriptor(), filterDescriptor);
    EXPECT_EQ(copy.getDataInputTensorDescriptor(), inputDescriptor);
    EXPECT_EQ(copy.getDataOutputTensorDescriptor(), outputDescriptor);
    EXPECT_EQ(copy.getBiasesTensorDescriptor(), biasesDescriptor);
    EXPECT_EQ(std::hash<ConvolutionKernelRequirement>()(copy), hash);

    ConvolutionKernelRequirement moved = std::move(copy);
    EXPECT_TRUE(moved == requirement);
    EXPECT_EQ(moved.getConvolutionDescriptor(), convolutionDescriptor);
    EXPECT_EQ(moved.getWeightsFilterDescriptor(), filterDescriptor);
    EXPECT_EQ(moved.getDataInputTensorDescriptor(), inputDescriptor);
    EXPECT_EQ(moved.getDataOutputTensorDescriptor(), outputDescriptor);
    EXPECT_EQ(moved.getBiasesTensorDescriptor(), biasesDescriptor);
    EXPECT_EQ(std::hash<ConvolutionKernelRequirement>()(moved), hash);
}

TEST(ConvolutionKernelRequirementSharedOwnership, SharedStateOutlivesOriginalHandle) {
    cudnnConvolutionDescriptor_t convolutionDescriptor = nullptr;
    cudnnFilterDescriptor_t filterDescriptor = nullptr;
    cudnnTensorDescriptor_t outputDescriptor = nullptr;

    ConvolutionKernelRequirement survivor = [&]() {
        ConvolutionKernelRequirement original = makeRequirement();
        convolutionDescriptor = original.getConvolutionDescriptor();
        filterDescriptor = original.getWeightsFilterDescriptor();
        outputDescriptor = original.getDataOutputTensorDescriptor();
        return ConvolutionKernelRequirement(original);
    }();

    EXPECT_EQ(survivor.getConvolutionDescriptor(), convolutionDescriptor);
    EXPECT_EQ(survivor.getWeightsFilterDescriptor(), filterDescriptor);
    EXPECT_EQ(survivor.getDataOutputTensorDescriptor(), outputDescriptor);
    EXPECT_EQ(survivor.getNumOutputColumns(), 7);
    EXPECT_EQ(survivor.getNumOutputRows(), 5);
}

TEST(ConvolutionKernelRequirementSharedOwnership, DistinctHandlesMayCopyAndDestroyConcurrently) {
    constexpr int kNumThreads = 8;
    constexpr int kCopiesPerThread = 10000;

    ConvolutionKernelRequirement root = makeRequirement();
    const cudnnConvolutionDescriptor_t convolutionDescriptor = root.getConvolutionDescriptor();
    const cudnnFilterDescriptor_t filterDescriptor = root.getWeightsFilterDescriptor();
    const cudnnTensorDescriptor_t inputDescriptor = root.getDataInputTensorDescriptor();
    const cudnnTensorDescriptor_t outputDescriptor = root.getDataOutputTensorDescriptor();
    const cudnnTensorDescriptor_t biasesDescriptor = root.getBiasesTensorDescriptor();
    const size_t expectedHash = std::hash<ConvolutionKernelRequirement>()(root);

    std::vector<ConvolutionKernelRequirement> stableSources;
    stableSources.reserve(kNumThreads);
    for (int i = 0; i < kNumThreads; ++i)
        stableSources.push_back(root);

    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (int threadIndex = 0; threadIndex < kNumThreads; ++threadIndex) {
        threads.emplace_back([&, threadIndex]() {
            ConvolutionKernelRequirement &source = stableSources[threadIndex];
            for (int iteration = 0; iteration < kCopiesPerThread; ++iteration) {
                ConvolutionKernelRequirement local = source;
                ConvolutionKernelRequirement moved = std::move(local);

                if (moved.getConvolutionDescriptor() != convolutionDescriptor || moved.getWeightsFilterDescriptor() != filterDescriptor ||
                    moved.getDataInputTensorDescriptor() != inputDescriptor || moved.getDataOutputTensorDescriptor() != outputDescriptor ||
                    moved.getBiasesTensorDescriptor() != biasesDescriptor ||
                    std::hash<ConvolutionKernelRequirement>()(moved) != expectedHash) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }

    for (std::thread &thread : threads)
        thread.join();

    EXPECT_FALSE(failed.load(std::memory_order_relaxed));
    EXPECT_EQ(root.getConvolutionDescriptor(), convolutionDescriptor);
    EXPECT_EQ(root.getDataOutputTensorDescriptor(), outputDescriptor);
}
