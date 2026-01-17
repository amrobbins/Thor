#include "Utilities/TarFile/ArchiveShardWriter.h"

using namespace std;

void writeShard(deque<ArchiveFileWriteParams> shardPlan) {
    AsyncQueue shardWriterQueue(std::move(shardPlan));
    ArchiveShardWriter archiveShardWriter(shardWriterQueue);
    ThreadPool threadPool(archiveShardWriter, 2);
    shardWriterQueue.waitForEmpty();
    shardWriterQueue.close();
}
