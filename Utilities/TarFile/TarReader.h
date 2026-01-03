#pragma once

#include "Crc32.h"
#include "Utilities/TarFile/TarArchive.h"

namespace thor_file {

// -------------------- TarReader (scan-on-open, index offsets) --------------------
//
// Uses libarchive to parse PAX correctly (x/g headers, long names, large sizes).
// Assumes uncompressed tar file on disk so offsets are meaningful and seekable.

struct ShardFd {
    std::string path;
    int fd = -1;
    uint64_t size = 0;
};

struct FileSliceFd {
    int fd = -1;  // owned by TarReader; valid while TarReader lives
    std::string shard_path;
    uint32_t shard_index = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint32_t crc = 0;
};

class TarReader {
   public:
    explicit TarReader(std::string tarPath);
    virtual ~TarReader();
    const std::unordered_map<std::string, EntryInfo>& entries() const;
    bool contains(std::string pathInTar) const;
    const EntryInfo& info(std::string pathInTar) const;
    uint64_t dataOffset(std::string pathInTar) const;
    // Random-access read of the whole file (works because archive is uncompressed and contiguous).
    void readFile(std::string pathInTar, void* mem, uint64_t fileSize) const;
    FileSliceFd getFileSliceFd(std::string pathInTar) const;

    // Verifies every entry's payload CRC. Throws on first failure.
    void verifyAll() const;

   private:
    void scan();

    // std::string tarPath;
    // std::unordered_map<std::string, EntryInfo> entries_;

    std::string prefix_;      // e.g. "path/to/MyModel"
    std::string archive_id_;  // 32-char hex-ish
    uint32_t num_shards_;
    std::unordered_map<std::string, EntryInfo> index_;
    std::vector<std::string> shard_paths_;
    std::vector<ShardFd> shard_fds_;
};

}  // namespace thor_file
