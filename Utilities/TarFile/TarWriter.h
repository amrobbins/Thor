#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TarFile/TarArchive.h"
#include "Utilities/ThreadPool/ThreadPool.h"

namespace thor_file {

struct ArchiveCreationPlanEntry {
    ThorImplementation::Tensor deviceTensor;

    uint64_t offsetBytes = 0;
    uint64_t numBytes = 0;
    std::string path_in_tar;
};

struct ArchiveShardCreationPlan {
    explicit ArchiveShardCreationPlan(const std::string &path) : archiveShardPath(path) {}

    std::string archiveShardPath;
    uint32_t shardNumber;
    uint64_t totalBytes = 0;
    std::vector<ArchiveCreationPlanEntry> entries;
};

struct WorkerJobContext {
    WorkerJobContext(std::unordered_map<std::string, EntryInfo> &archiveIndex,
                     std::mutex &archiveIndexMutex,
                     std::filesystem::path archiveDirectory)
        : archiveIndex(archiveIndex), archiveIndexMutex(archiveIndexMutex), archiveDirectory(archiveDirectory) {}

    std::unordered_map<std::string, EntryInfo> &archiveIndex;
    std::mutex &archiveIndexMutex;
    std::filesystem::path archiveDirectory;
};

class TarWriter {
   public:
    explicit TarWriter(std::string archiveName, uint64_t shard_payload_limit_bytes = 5e9);
    TarWriter(const TarWriter &) = delete;
    TarWriter &operator=(const TarWriter &) = delete;
    virtual ~TarWriter() = default;

    // Called once per file that will be added to the archive
    void addArchiveFile(const std::string &pathInTar, ThorImplementation::Tensor &tensor);

    // Called after all files have been added to the archive. Does the actual writing of the data to disk.
    std::string createArchive(std::filesystem::path archiveDirectory, bool overwriteIfExists);

   private:
    // e.g. "MyModel"
    std::string archiveName;
    const uint64_t SHARD_PAYLOAD_LIMIT;

    std::vector<ArchiveShardCreationPlan> archiveShardCreationPlan;

    std::mutex archiveIndexMutex;
};

}  // namespace thor_file
