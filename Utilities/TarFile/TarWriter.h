#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TarFile/TarArchive.h"
#include "Utilities/ThreadPool/ThreadPool.h"

namespace thor_file {

class TarWriter {
   public:
    explicit TarWriter(const std::string &archiveName,
                       uint64_t archiveShardSizeLimitBytes = 5'010'000'000,
                       uint64_t fileShardSizeLimitBytes = 500'000'000);
    TarWriter(const TarWriter &) = delete;
    TarWriter &operator=(const TarWriter &) = delete;
    virtual ~TarWriter() = default;

    // Called once per file that will be added to the archive
    void addArchiveFile(const std::string &pathInTar, ThorImplementation::Tensor &tensor);

    // Called after all files have been added to the archive. Does the actual writing of the data to disk.
    std::string createArchive(std::filesystem::path archiveDirectory, bool overwriteIfExists);

   private:
    const uint64_t ARCHIVE_SHARD_PAYLOAD_LIMIT;
    const uint64_t MAX_FILE_SHARD_BYTES;

    // e.g. "MyModel"
    const std::string archiveName;

    std::vector<ArchiveShardPlan> archiveShardCreationPlan;

    std::mutex archiveIndexMutex;
};

}  // namespace thor_file
