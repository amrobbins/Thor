// The batched submission queue of the audits and sweeps (benchmarks/Fp8OracleBatchQueue.h): one warmup and sync,
// then per batch every pass of every cell in order, exactly one measured sync, and each cell collected once.
// Host-only.

#include "benchmarks/Fp8OracleBatchQueue.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

using ThorImplementation::ReductionBenchmarking::Fp8OracleBatchQueue;

TEST(Fp8OracleBatchQueue, BatchesSubmitEveryPassThenSyncOnceThenCollect) {
    enum Type { Warmup, Submit, Sync, Collect };
    struct Action {
        Type type;
        int cell;
        int pass;
    };
    std::vector<Action> log;
    Fp8OracleBatchQueue queue(128, 5, [&] { log.push_back({Sync, -1, -1}); });
    for (int id = 0; id < 257; ++id) {
        queue.add({
            [&, id] { log.push_back({Warmup, id, -1}); },
            [&, id](int pass) { log.push_back({Submit, id, pass}); },
            [&, id] { log.push_back({Collect, id, -1}); },
        });
    }
    EXPECT_EQ(queue.batches(), 2u);
    EXPECT_EQ(queue.pending(), 1u);
    queue.flush();
    EXPECT_EQ(queue.batches(), 3u);
    EXPECT_EQ(queue.completedCells(), 257u);
    EXPECT_EQ(queue.pending(), 0u);
    const size_t logged = log.size();
    queue.flush();  // nothing pending: no work
    EXPECT_EQ(log.size(), logged);

    size_t cursor = 0;
    ASSERT_GE(log.size(), 2u);
    EXPECT_EQ(log[cursor++].type, Warmup);  // one warmup, then a sync
    EXPECT_EQ(log[cursor++].type, Sync);
    for (int batch = 0; batch < 3; ++batch) {
        const int first = batch * 128;
        const int count = batch == 2 ? 1 : 128;
        for (int pass = 0; pass < 5; ++pass) {
            for (int id = first; id < first + count; ++id) {
                ASSERT_LT(cursor, log.size());
                const Action action = log[cursor++];
                EXPECT_EQ(action.type, Submit);
                EXPECT_EQ(action.cell, id);
                EXPECT_EQ(action.pass, pass);
            }
        }
        ASSERT_LT(cursor, log.size());
        EXPECT_EQ(log[cursor++].type, Sync);  // exactly one measured sync per batch
        for (int id = first; id < first + count; ++id) {
            ASSERT_LT(cursor, log.size());
            const Action action = log[cursor++];
            EXPECT_EQ(action.type, Collect);
            EXPECT_EQ(action.cell, id);
        }
    }
    EXPECT_EQ(cursor, log.size());
    EXPECT_THROW(Fp8OracleBatchQueue invalid(0, 5, [] {}), std::invalid_argument);
}
