#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"

#include "gtest/gtest.h"

using namespace Thor;

TEST(TrainingCancellationToken, DefaultTokenIsNotCancelled) {
    TrainingCancellationToken token;

    EXPECT_FALSE(token.isCancellationRequested());
    EXPECT_NO_THROW(token.throwIfCancellationRequested());
}

TEST(TrainingCancellationSource, CancelsSharedTokens) {
    TrainingCancellationSource source;
    TrainingCancellationToken token = source.token();
    TrainingCancellationToken copiedToken = token;

    EXPECT_FALSE(source.isCancellationRequested());
    EXPECT_FALSE(token.isCancellationRequested());
    EXPECT_FALSE(copiedToken.isCancellationRequested());

    source.requestCancellation();

    EXPECT_TRUE(source.isCancellationRequested());
    EXPECT_TRUE(token.isCancellationRequested());
    EXPECT_TRUE(copiedToken.isCancellationRequested());
    EXPECT_THROW(token.throwIfCancellationRequested("stop fold"), TrainingCancelled);
}
