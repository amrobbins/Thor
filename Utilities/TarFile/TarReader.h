#pragma once

#include "Crc32.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TarFile/TarArchive.h"
#include "Utilities/ThreadPool/ThreadPool.h"

namespace thor_file {

class TarReader {
   public:
    explicit TarReader(std::string archiveName, std::filesystem::path archiveDirectory);
    TarReader(const TarReader &) = delete;
    TarReader &operator=(const TarReader &) = delete;
    virtual ~TarReader() = default;

    void registerReadRequest(const std::string &pathInTar, ThorImplementation::Tensor &destTensor);
    void executeReadRequests();

    bool containsFile(const std::string &pathInTar) const;
    uint64_t getFileSize(const std::string &pathInTar) const;
    const std::vector<EntryInfo> &getFileShards(std::string pathInTar) const;
    std::unordered_map<std::string, std::vector<EntryInfo>> getArchiveEntries() const;

   private:
    void scan();

    std::filesystem::path archiveDirectory;
    std::string archiveName;
    std::string archiveId;  // 32-char hex-ish
    uint32_t numShards;
    std::unordered_map<std::string, std::vector<EntryInfo>> archiveIndex;
    std::vector<std::string> shardFilenames;

    std::unordered_map<uint32_t, uint32_t> shardPlanArrayIndex;
    std::vector<ArchiveShardPlan> readPlan;

    std::string errorMessage;
    std::mutex mtx;
};

}  // namespace thor_file
