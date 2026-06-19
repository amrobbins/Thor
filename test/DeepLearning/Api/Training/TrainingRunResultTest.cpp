#include "DeepLearning/Api/Training/Results/TrainingRunResult.h"

#include "gtest/gtest.h"

#include <exception>
#include <new>
#include <stdexcept>

using namespace Thor;

namespace {

template <typename ExceptionT>
std::exception_ptr makeExceptionPtr(ExceptionT exception) {
    try {
        throw exception;
    } catch (...) {
        return std::current_exception();
    }
}

}  // namespace

TEST(TrainingRunStatus, HasStableUserFacingNames) {
    EXPECT_STREQ(trainingRunStatusName(TrainingRunStatus::NOT_STARTED), "not_started");
    EXPECT_STREQ(trainingRunStatusName(TrainingRunStatus::RUNNING), "running");
    EXPECT_STREQ(trainingRunStatusName(TrainingRunStatus::COMPLETED), "completed");
    EXPECT_STREQ(trainingRunStatusName(TrainingRunStatus::FAILED), "failed");
    EXPECT_STREQ(trainingRunStatusName(TrainingRunStatus::CANCELLED), "cancelled");
    EXPECT_STREQ(trainingRunStatusName(TrainingRunStatus::INTERRUPTED), "interrupted");
    EXPECT_STREQ(trainingRunStatusName(TrainingRunStatus::OUT_OF_MEMORY), "oom");
}

TEST(TrainingRunResult, CompletedResultCarriesRunNameAndFinalStats) {
    TrainingStatsSnapshot train;
    train.phase = TrainingEventPhase::TRAIN;
    train.loss = 1.25;
    TrainingStatsSnapshot validate;
    validate.phase = TrainingEventPhase::VALIDATE;
    validate.loss = 1.5;

    TrainingRunResult result = TrainingRunResult::completedResult("fold_0", train, validate);

    EXPECT_EQ(result.runName, "fold_0");
    EXPECT_EQ(result.status, TrainingRunStatus::COMPLETED);
    EXPECT_TRUE(result.completed());
    ASSERT_TRUE(result.finalTrainingStats.has_value());
    ASSERT_TRUE(result.finalValidationStats.has_value());
    ASSERT_TRUE(result.finalTrainingStats->loss.has_value());
    ASSERT_TRUE(result.finalValidationStats->loss.has_value());
    EXPECT_DOUBLE_EQ(*result.finalTrainingStats->loss, 1.25);
    EXPECT_DOUBLE_EQ(*result.finalValidationStats->loss, 1.5);
}

TEST(TrainingRunResult, ClassifiesCancellationAndInterruptExceptions) {
    TrainingRunResult cancelled = TrainingRunResult::fromException("fold_1", makeExceptionPtr(TrainingCancelled("cancelled by sibling")));
    TrainingRunResult interrupted = TrainingRunResult::fromException("fold_2", makeExceptionPtr(TrainingInterrupted("ctrl-c")));

    EXPECT_EQ(cancelled.status, TrainingRunStatus::CANCELLED);
    EXPECT_TRUE(cancelled.cancelled());
    EXPECT_EQ(cancelled.exception.type, "TrainingCancelled");
    EXPECT_EQ(cancelled.exception.message, "cancelled by sibling");

    EXPECT_EQ(interrupted.status, TrainingRunStatus::INTERRUPTED);
    EXPECT_TRUE(interrupted.failed());
    EXPECT_EQ(interrupted.exception.type, "TrainingInterrupted");
    EXPECT_EQ(interrupted.exception.message, "ctrl-c");
}

TEST(TrainingRunResult, ClassifiesOutOfMemoryExceptions) {
    TrainingRunResult badAlloc = TrainingRunResult::fromException("fold_3", makeExceptionPtr(std::bad_alloc{}));
    TrainingRunResult cudaMessage =
        TrainingRunResult::fromException("fold_4", makeExceptionPtr(std::runtime_error("CUDA_ERROR_OUT_OF_MEMORY during placement")));
    TrainingRunResult networkMessage =
        TrainingRunResult::fromException("fold_5", makeExceptionPtr(std::logic_error("Error when stamping network, error: GPU OUT OF MEMORY")));

    EXPECT_EQ(badAlloc.status, TrainingRunStatus::OUT_OF_MEMORY);
    EXPECT_EQ(cudaMessage.status, TrainingRunStatus::OUT_OF_MEMORY);
    EXPECT_EQ(networkMessage.status, TrainingRunStatus::OUT_OF_MEMORY);
}

TEST(TrainingRunResult, ClassifiesOrdinaryExceptionAsFailed) {
    TrainingRunResult result = TrainingRunResult::fromException("arch_a", makeExceptionPtr(std::runtime_error("loader failed")));

    EXPECT_EQ(result.runName, "arch_a");
    EXPECT_EQ(result.status, TrainingRunStatus::FAILED);
    EXPECT_TRUE(result.failed());
    EXPECT_EQ(result.exception.message, "loader failed");
}
