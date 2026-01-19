// #include "Utilities/TarFile/ArchiveShardWriterWorker.h"
//
// using namespace std;
//
// // writeShards will write one or more archive shards, depending on the number of shards there are and the desired level of parallelism,
// // this is controlled by the contents of shardPlan.
// // All files for any single archive shard should be contained within a single shardPlan, so that one writeShards invocation will create
// // the whole shard. That way there will be some number of threads dedicated to some number of shards.
// void writeShards(deque<ArchiveFileWriteParams>&& shardPlan, uint32_t threadPoolSize) {
//     AsyncQueue shardWriterQueue(std::move(shardPlan));
//     shardWriterQueue.open();
//     ArchiveShardWriterWorker archiveShardWriterWorker(shardWriterQueue);
//     ThreadPool threadPool(archiveShardWriterWorker, threadPoolSize);
//     shardWriterQueue.waitForEmpty();
//     shardWriterQueue.close();
//     threadPool.stop();
// }
