#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/ArchiveShardReaderWorker.h"

using namespace std;
using json = nlohmann::json;

namespace thor_file {
static uint64_t read_u64_le(const uint8_t b[8]) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | static_cast<uint64_t>(b[i]);
    }
    return v;
}

static uint32_t read_u32_le(const uint8_t b[4]) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static string read_footer_json(const string& path) {
    ifstream in(path, ios::binary);
    if (!in)
        throw runtime_error("TarReader::scan: failed to open shard: " + path);

    in.seekg(0, ios::end);
    const streamoff file_size_off = in.tellg();
    if (file_size_off < 0)
        throw runtime_error("TarReader::scan: tellg failed: " + path);

    const uint64_t file_size = static_cast<uint64_t>(file_size_off);
    if (file_size < kFooterSize) {
        throw runtime_error("TarReader::scan: file too small to contain footer: " + path);
    }

    // Read footer
    in.seekg(static_cast<streamoff>(file_size - kFooterSize), ios::beg);

    char magic[8];
    uint8_t len_bytes[8];
    uint8_t crc_bytes[4];
    uint8_t reserved_bytes[4];

    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(len_bytes), 8);
    in.read(reinterpret_cast<char*>(crc_bytes), 4);
    in.read(reinterpret_cast<char*>(reserved_bytes), 4);
    if (!in)
        throw runtime_error("TarReader::scan: failed to read footer: " + path);

    if (memcmp(magic, kFooterMagic, 8) != 0) {
        throw runtime_error("TarReader::scan: footer magic mismatch (no index?): " + path);
    }

    const uint64_t json_len = read_u64_le(len_bytes);
    if (json_len == 0) {
        throw runtime_error("TarReader::scan: json_len is 0: " + path);
    }
    if (json_len > file_size - kFooterSize) {
        throw runtime_error("TarReader::scan: json_len exceeds file size: " + path);
    }

    const uint32_t index_crc_expected = read_u32_le(crc_bytes);

    const uint64_t json_start = file_size - kFooterSize - json_len;

    in.seekg(static_cast<streamoff>(json_start), ios::beg);
    string json_str(json_len, '\0');
    in.read(json_str.data(), static_cast<streamsize>(json_len));
    if (!in)
        throw runtime_error("TarReader::scan: failed to read json blob: " + path);

    // Compute CRC over JSON bytes exactly as written
    const uint32_t index_crc_computed = crc32_ieee(0xFFFFFFFF, (uint8_t*)json_str.data(), json_str.size());
    if (index_crc_computed != index_crc_expected) {
        throw runtime_error("read_footer_json: index CRC mismatch in " + path + " expected=" + to_string(index_crc_expected) +
                            " got=" + to_string(index_crc_computed));
    }

    return json_str;
}

static void require(bool cond, const string& msg) {
    if (!cond)
        throw runtime_error("TarReader::scan: " + msg);
}

static bool is_sha32(string_view s) {
    if (s.size() != 32)
        return false;
    for (char c : s) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        if (!ok)
            return false;
    }
    return true;
}

unordered_map<string, vector<EntryInfo>> TarReader::getArchiveEntries() const { return archiveIndex; }

bool TarReader::containsFile(const string& pathInTar) const {
    string cleanedPathInTar = cleanTarPath(pathInTar);
    return archiveIndex.find(cleanedPathInTar) != archiveIndex.end();
}

const vector<EntryInfo>& TarReader::getFileShards(string pathInTar) const {
    pathInTar = cleanTarPath(pathInTar);
    auto it = archiveIndex.find(pathInTar);
    if (it == archiveIndex.end())
        throw runtime_error("tar entry not found: " + pathInTar);
    return it->second;
}

auto to_abs_norm = [](const string& p) -> string {
    filesystem::path ap = filesystem::absolute(filesystem::path(p));
    ap = ap.lexically_normal();
    return ap.string();
};

static string shard0PathAny(const string& prefix) {
    string numbered0 = shard_filename(prefix, 0);  // prefix + ".000000.thor.tar"
    string single0 = prefix + ".thor.tar";

    bool numberedExists = filesystem::exists(numbered0);
    bool singleExists = filesystem::exists(single0);

    if (numberedExists && singleExists)
        throw runtime_error("TarReader::scan: found both (" + numbered0 + " and " + single0 +
                            "), only one of those should be present, cannot open archive.");

    if (numberedExists)
        return numbered0;
    if (filesystem::exists(single0))
        return single0;

    throw runtime_error("TarReader::scan: missing shard0 (" + numbered0 + " or " + single0 + ")");
}

static string getArchiveShardPath(const string& prefix, const uint32_t shardNum, const uint32_t numShards) {
    if (shardNum == 0 && numShards == 1)
        return prefix + ".thor.tar";
    return shard_filename(prefix, shardNum);
}

void TarReader::scan() {
    namespace fs = filesystem;

    archiveIndex.clear();
    shardFilenames.clear();
    archiveId.clear();
    numShards = 0;

    // Step 1: Read index JSON from shard0 (either prefix.000000.thor.tar or prefix.thor.tar)
    const string first_path = shard0PathAny(archiveDirectory / archiveName);

    const string referenceIndexJsonStr = read_footer_json(first_path);

    json shard0IndexJson;
    try {
        shard0IndexJson = json::parse(referenceIndexJsonStr);
    } catch (const exception& e) {
        throw runtime_error(string("TarReader::scan: failed parsing JSON index: ") + e.what());
    }

    require(shard0IndexJson.contains("shard_index"), "index JSON missing 'shard_index'");
    const uint32_t shard0Index = shard0IndexJson.at("shard_index").get<uint32_t>();
    require(shard0Index == 0, "shard_index in shard 0 index must be 0");

    require(shard0IndexJson.contains("format_version"), "index JSON missing 'format_version'");
    const uint32_t version = shard0IndexJson.at("format_version").get<uint32_t>();
    require(version == 1, "unsupported format_version: " + to_string(version));

    require(shard0IndexJson.contains("checksum_alg"), "index JSON missing 'checksum_alg'");
    const string crcAlgo = shard0IndexJson.at("checksum_alg").get<string>();
    require(crcAlgo == "crc32_ieee", "unsupported checksum_alg: " + crcAlgo);

    // Compare later ignoring shard_index
    json shard0IndexCompareJson = shard0IndexJson;
    shard0IndexCompareJson.erase("shard_index");

    require(shard0IndexCompareJson.is_object(), "index JSON must be an object");
    require(shard0IndexCompareJson.contains("archive_id"), "index JSON missing 'archive_id'");
    require(shard0IndexCompareJson.contains("num_shards"), "index JSON missing 'num_shards'");
    require(shard0IndexCompareJson.contains("files"), "index JSON missing 'files'");

    const string shard0ArchiveId = shard0IndexCompareJson.at("archive_id").get<string>();
    require(is_sha32(shard0ArchiveId), "archive_id must look like 32 hex chars");

    const uint32_t shard0NumShards = shard0IndexCompareJson.at("num_shards").get<uint32_t>();
    require(shard0NumShards >= 1, "num_shards must be >= 1");
    readPlan.reserve(shard0NumShards);
    shardPlanArrayIndex.reserve(shard0NumShards);

    archiveId = shard0ArchiveId;
    numShards = shard0NumShards;

    // Step 2: Determine expected shard paths
    shardFilenames.reserve(numShards);

    if (numShards == 1) {
        // single-shard convention is prefix.thor.tar
        const string shardFilename = archiveName + ".thor.tar";
        require(fs::exists(archiveDirectory / shardFilename), "missing single shard: " + (archiveDirectory / shardFilename).string());
        shardFilenames.push_back(shardFilename);
    } else {
        // multi-shard convention is numbered
        for (uint32_t i = 0; i < numShards; ++i) {
            const string shardFilename = shard_filename(archiveName, i);
            require(fs::exists(archiveDirectory / shardFilename),
                    "missing shard " + to_string(i) + ": " + (archiveDirectory / shardFilename).string());
            shardFilenames.push_back(shardFilename);
        }
    }

    // Step 3: Ensure each shard has identical index except shard_index
    for (uint32_t shardNum = 0; shardNum < numShards; ++shardNum) {
        const string shardPath = archiveDirectory / shardFilenames[shardNum];
        const string shardIndexJsonStr = read_footer_json(shardPath);

        json shardIndexJson = json::parse(shardIndexJsonStr);

        require(shardIndexJson.contains("archive_id"), "index JSON missing 'archive_id' in shard " + to_string(shardNum));
        string shardArchiveId = shardIndexJson.at("archive_id").get<string>();

        require(archiveId == shardArchiveId,
                "Archive ids mismatch between shards: shard0=" + archiveId + " shard" + to_string(shardNum) + "=" + shardArchiveId);

        require(shardIndexJson.contains("shard_index"), "index JSON missing 'shard_index' in: " + shardPath);
        const uint32_t shardIndex = shardIndexJson.at("shard_index").get<uint32_t>();
        require(shardIndex == shardNum, "shard_index mismatch in: " + shardPath);

        shardIndexJson.erase("shard_index");
        require(shard0IndexCompareJson == shardIndexJson,
                "index JSON differs across shards (expected identical except shard_index): " + shardPath);
    }

    // Step 4: Load entries
    const auto& files = shard0IndexCompareJson.at("files");
    require(files.is_object(), "'files' must be an object mapping path->info");

    for (const auto& [pathInArchive, shardList] : files.items()) {
        if (!shardList.is_array()) {
            throw std::runtime_error("files[" + pathInArchive + "] is not an array");
        }

        for (const json& entryInfoJ : shardList) {
            require(entryInfoJ.is_object(), "file info must be an object for: " + pathInArchive);
            require(entryInfoJ.contains("archive_shard"), "missing 'shard' for: " + pathInArchive);
            require(entryInfoJ.contains("file_data_offset"), "missing 'file_data_offset' for: " + pathInArchive);
            require(entryInfoJ.contains("size"), "missing 'size' for: " + pathInArchive);
            require(entryInfoJ.contains("crc"), "missing 'crc' for: " + pathInArchive);

            EntryInfo entryInfo{};
            entryInfo.archiveShard = entryInfoJ.at("archive_shard").get<uint32_t>();
            entryInfo.fileDataOffset = entryInfoJ.at("file_data_offset").get<uint64_t>();
            entryInfo.tensorDataOffset = entryInfoJ.at("tensor_data_offset").get<uint64_t>();
            entryInfo.size = entryInfoJ.at("size").get<uint64_t>();
            entryInfo.crc = entryInfoJ.at("crc").get<uint32_t>();

            require(entryInfo.archiveShard < numShards, "entry refers to invalid shard for: " + pathInArchive);

            const string& shardPath = archiveDirectory / shardFilenames[entryInfo.archiveShard];
            const uint64_t shardSize = static_cast<uint64_t>(fs::file_size(shardPath));

            require(entryInfo.fileDataOffset < shardSize, "data_offset beyond shard size for: " + pathInArchive);
            require(entryInfo.fileDataOffset + entryInfo.size <= shardSize, "data read would exceed shard size for: " + pathInArchive);

            require(!archiveIndex.contains(pathInArchive), "file path duplicated in archive: " + pathInArchive);
            archiveIndex[pathInArchive].emplace_back(entryInfo);
        }
    }
}

TarReader::TarReader(string archiveName, filesystem::path archiveDirectory) {
    this->archiveName = archiveName;
    this->archiveDirectory = archiveDirectory;
    scan();
}

uint64_t TarReader::getFileSize(const std::string& pathInTar) const {
    string cleanedPathInTar = cleanTarPath(pathInTar);
    require(containsFile(cleanedPathInTar),
            "Archive " + archiveName + " at " + archiveDirectory.string() + " does not contain file " + pathInTar +
                " but a read was requested of it.");
    vector<EntryInfo> fileEnties = archiveIndex.find(cleanedPathInTar)->second;
    uint64_t numBytes = 0;
    for (EntryInfo entryInfo : fileEnties) {
        numBytes += entryInfo.size;
    }
    return numBytes;
}

void TarReader::registerReadRequest(const string& pathInTar, ThorImplementation::Tensor& destTensor) {
    string cleanedPathInTar = cleanTarPath(pathInTar);
    auto it = archiveIndex.find(cleanedPathInTar);
    if (it == archiveIndex.end()) {
        if (pathInTar != cleanedPathInTar)
            throw runtime_error("tar entry not found: " + pathInTar + " cleaned as " + cleanedPathInTar);
        else
            throw runtime_error("tar entry not found: " + cleanedPathInTar);
    }

    // Iterate over all file shards and register a read request to the archive shard containing each file shard
    const vector<EntryInfo>& shardEntries = it->second;

    for (const EntryInfo& entry : shardEntries) {
        // // Get or create if does not already exist:
        // auto [fileIt, inserted] = readPlan.try_emplace(entry.archiveShard, ArchiveShardPlan(shardPaths[entry.archiveShard]));
        // ArchiveShardPlan& shardPlan = fileIt->second;

        uint32_t planArrayIndex;
        const auto mapIt = shardPlanArrayIndex.find(entry.archiveShard);
        if (mapIt == shardPlanArrayIndex.end()) {
            planArrayIndex = readPlan.size();

            shardPlanArrayIndex.emplace(entry.archiveShard, planArrayIndex);

            ArchiveShardPlan shardPlan(shardFilenames[entry.archiveShard]);
            shardPlan.shardNumber = entry.archiveShard;
            readPlan.emplace_back(shardPlan);
        } else {
            planArrayIndex = mapIt->second;
        }

        ArchivePlanEntry planEntry;
        planEntry.tensor = destTensor;
        planEntry.pathInTar = cleanedPathInTar;
        planEntry.fileOffsetBytes = entry.fileDataOffset;
        planEntry.tensorOffsetBytes = entry.tensorDataOffset;
        planEntry.numBytes = entry.size;
        planEntry.expectedCrc = entry.crc;
        readPlan[planArrayIndex].entries.push_back(planEntry);
    }
}

void TarReader::executeReadRequests() {
    if (readPlan.empty())
        return;

    ArchiveReaderContext context(archiveDirectory, errorMessage, mtx);

    ThreadPool<ArchiveShardReaderWorker, ArchiveShardPlan, ArchiveReaderContext> archiveReaderThreadPool(readPlan, context, 3);
    archiveReaderThreadPool.wait();

    // Check for any errors, throw runtime error when present.
    for (ArchiveShardPlan shardPlan : readPlan) {
        require(errorMessage.empty(), errorMessage);
    }

    readPlan.clear();
    shardPlanArrayIndex.clear();
}

}  // namespace thor_file
