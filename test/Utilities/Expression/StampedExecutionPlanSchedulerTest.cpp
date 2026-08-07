#include "Utilities/Expression/StampedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

StampedExecutionStage barrier(std::vector<uint32_t> dependencies = {}, uint32_t gpu_num = 0) {
    return StampedExecutionStage::dependencyBarrier(gpu_num, std::move(dependencies));
}

TEST(StampedExecutionPlanScheduler, LinearChainStaysEntirelyOnCallerLane) {
    std::vector<StampedExecutionStage> stages;
    stages.push_back(barrier());
    stages.push_back(barrier({0}));
    stages.push_back(barrier({1}));
    stages.push_back(barrier({2}));

    const detail::StampedExecutionSchedule schedule = detail::buildStampedExecutionSchedule(stages, 0);

    EXPECT_EQ(schedule.stage_lane_indices, (std::vector<uint32_t>{0, 0, 0, 0}));
    EXPECT_EQ(schedule.lane_gpu_nums, (std::vector<uint32_t>{0}));
    EXPECT_FALSE(schedule.needs_caller_ready_event);
    EXPECT_EQ(schedule.stage_needs_completion_event, (std::vector<bool>{false, false, false, false}));
}

TEST(StampedExecutionPlanScheduler, ForkCreatesOneHelperLaneAndJoinReturnsToCallerLane) {
    std::vector<StampedExecutionStage> stages;
    stages.push_back(barrier());       // 0: fork source
    stages.push_back(barrier({0}));    // 1: caller-stream branch
    stages.push_back(barrier({0}));    // 2: parallel helper branch
    stages.push_back(barrier({1, 2})); // 3: join
    stages.push_back(barrier({3}));    // 4: linear continuation

    const detail::StampedExecutionSchedule schedule = detail::buildStampedExecutionSchedule(stages, 0);

    EXPECT_EQ(schedule.stage_lane_indices, (std::vector<uint32_t>{0, 0, 1, 0, 0}));
    EXPECT_EQ(schedule.lane_gpu_nums, (std::vector<uint32_t>{0, 0}));
    EXPECT_FALSE(schedule.needs_caller_ready_event);
    EXPECT_EQ(schedule.stage_needs_completion_event, (std::vector<bool>{true, false, true, false, false}));
}

TEST(StampedExecutionPlanScheduler, IndependentRootsUseOneCallerReadyEventAndOneCrossStreamJoinEvent) {
    std::vector<StampedExecutionStage> stages;
    stages.push_back(barrier());       // 0: caller-stream root
    stages.push_back(barrier());       // 1: independent helper root
    stages.push_back(barrier({0, 1})); // 2: join onto caller lane
    stages.push_back(barrier({2}));    // 3: linear continuation

    const detail::StampedExecutionSchedule schedule = detail::buildStampedExecutionSchedule(stages, 0);

    EXPECT_EQ(schedule.stage_lane_indices, (std::vector<uint32_t>{0, 1, 0, 0}));
    EXPECT_EQ(schedule.lane_gpu_nums, (std::vector<uint32_t>{0, 0}));
    EXPECT_TRUE(schedule.needs_caller_ready_event);
    EXPECT_EQ(schedule.stage_needs_completion_event, (std::vector<bool>{false, true, false, false}));
}

TEST(StampedExecutionPlanScheduler, NestedForksOnlyCreateLanesForParallelBranches) {
    std::vector<StampedExecutionStage> stages;
    stages.push_back(barrier());       // 0
    stages.push_back(barrier({0}));    // 1: first branch, caller lane
    stages.push_back(barrier({0}));    // 2: second branch, helper lane 1
    stages.push_back(barrier({1}));    // 3: continuation of stage 1
    stages.push_back(barrier({1}));    // 4: fork from stage 1, helper lane 2
    stages.push_back(barrier({2}));    // 5: continuation of stage 2 on helper lane 1
    stages.push_back(barrier({3, 4})); // 6: join first nested fork onto caller lane
    stages.push_back(barrier({5, 6})); // 7: final join onto caller lane

    const detail::StampedExecutionSchedule schedule = detail::buildStampedExecutionSchedule(stages, 0);

    EXPECT_EQ(schedule.stage_lane_indices, (std::vector<uint32_t>{0, 0, 1, 0, 2, 1, 0, 0}));
    EXPECT_EQ(schedule.lane_gpu_nums, (std::vector<uint32_t>{0, 0, 0}));
}

TEST(StampedExecutionPlanScheduler, RejectsNonTopologicalDependencies) {
    std::vector<StampedExecutionStage> stages;
    stages.push_back(barrier({1}));
    stages.push_back(barrier());

    EXPECT_THROW((void)detail::buildStampedExecutionSchedule(stages, 0), std::runtime_error);
}

TEST(StampedExecutionPlanScheduler, RepeatedForkJoinSubmissionReusesEventsWithoutSynchronizingBetweenRuns) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device is required for stamped execution scheduler runtime test.";
    }

    Stream stream(0);
    std::vector<StampedExecutionStage> stages;
    stages.push_back(barrier());
    stages.push_back(barrier({0}));
    stages.push_back(barrier({0}));
    stages.push_back(barrier({1, 2}));

    StampedExecutionPlan plan(std::move(stages), {}, stream);
    for (uint32_t iteration = 0; iteration < 128; ++iteration) {
        plan.runOn(stream);
    }
    stream.synchronize();
}

}  // namespace
