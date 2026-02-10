#include "Utilities/TarFile/TarWriter.h"

#include "Utilities/TarFile/ArchiveShardWriterWorker.h"

#include "Crc32.h"
#include "Crc32c.h"

using namespace std;

namespace thor_file {

static string make_archive_id_sha32() {
    uint8_t buf[16];

    // XOR's a time based seed random with random_device random, so get entropy based random when available,
    // when not available still different run to run.
    random_device rd;

    const uint64_t t = static_cast<uint64_t>(chrono::high_resolution_clock::now().time_since_epoch().count());
    const uint64_t tid = static_cast<uint64_t>(hash<thread::id>{}(this_thread::get_id()));
    const uint64_t addr = reinterpret_cast<uint64_t>(&buf);

    seed_seq seed{static_cast<uint32_t>(t),
                  static_cast<uint32_t>(t >> 32),
                  static_cast<uint32_t>(tid),
                  static_cast<uint32_t>(tid >> 32),
                  static_cast<uint32_t>(addr),
                  static_cast<uint32_t>(addr >> 32)};
    mt19937_64 gen(seed);

    for (int i = 0; i < 16; i += 4) {
        uint32_t x = rd() ^ static_cast<uint32_t>(gen());
        buf[i + 0] = static_cast<uint8_t>((x >> 0) & 0xFF);
        buf[i + 1] = static_cast<uint8_t>((x >> 8) & 0xFF);
        buf[i + 2] = static_cast<uint8_t>((x >> 16) & 0xFF);
        buf[i + 3] = static_cast<uint8_t>((x >> 24) & 0xFF);
    }

    static const char* hexd = "0123456789abcdef";
    string out(32, '\0');
    for (int i = 0; i < 16; ++i) {
        out[2 * i + 0] = hexd[(buf[i] >> 4) & 0xF];
        out[2 * i + 1] = hexd[(buf[i] >> 0) & 0xF];
    }
    return out;
}

static bool is_valid_shard_filename(const string& base, const string& fname) {
    // pattern: base.thor.tar
    if (fname == base + ".thor.tar")
        return true;

    // pattern: base + "." + 6 digits + ".thor.tar"
    const string suffix = ".thor.tar";
    const size_t base_len = base.size();
    const size_t want_len = base_len + 1 + 6 + suffix.size();
    if (fname.size() != want_len)
        return false;
    if (fname.compare(0, base_len, base) != 0)
        return false;
    if (fname[base_len] != '.')
        return false;
    if (fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;

    // check 6 digits
    const size_t digits_off = base_len + 1;
    for (size_t i = 0; i < 6; ++i) {
        unsigned char c = static_cast<unsigned char>(fname[digits_off + i]);
        if (!isdigit(c))
            return false;
    }
    return true;
}

static void write_u64_le(std::ostream& out, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) {
        b[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
    }
    out.write(reinterpret_cast<const char*>(b), 8);
}

static void write_u32_le(std::ostream& out, uint32_t v) {
    uint8_t b[4];
    b[0] = static_cast<uint8_t>((v >> 0) & 0xFF);
    b[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    b[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    b[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    out.write(reinterpret_cast<const char*>(b), 4);
}

static uint64_t round_up_512(uint64_t n) { return (n + 511ull) & ~511ull; }

static string archiveShardTempPath(const string& prefix, uint32_t shard_idx) {
    char buf[64];
    snprintf(buf, sizeof(buf), ".%06u.thor.tar.incomplete", shard_idx);
    return prefix + string(buf);
}

static std::string strip_suffix_or_throw(const std::string& s, const std::string& suffix, uint32_t numShards) {
    if (s.size() < suffix.size() || s.compare(s.size() - suffix.size(), suffix.size(), suffix) != 0) {
        throw std::runtime_error("expected suffix '" + suffix + "' on: " + s);
    }
    string permanent = s.substr(0, s.size() - suffix.size());

    if (numShards == 1) {
        // We expect shard 0 naming: "<prefix>.000000.thor.tar"
        // Strip the ".000000" so it becomes "<prefix>.thor.tar".
        constexpr const char* kShard0 = ".000000";
        constexpr const char* kTar = ".thor.tar";

        const std::string expected_tail = std::string(kShard0) + kTar;
        if (permanent.size() < expected_tail.size() ||
            permanent.compare(permanent.size() - expected_tail.size(), expected_tail.size(), expected_tail) != 0) {
            throw std::runtime_error("expected single-shard name to end with '" + expected_tail + "' but got: " + permanent);
        }

        // Remove ".000000" immediately before ".thor.tar"
        permanent.erase(permanent.size() - expected_tail.size(), std::strlen(kShard0));
    }

    return permanent;
}

static uint32_t usualFileSize(uint64_t fileSize) {
    // Tar header + round to even 512 byte boundary
    return 512ull + round_up_512(fileSize);
}

TarWriter::TarWriter(const string& archiveName, uint64_t archiveShardSizeLimitBytes, uint64_t fileShardSizeLimitBytes)
    : ARCHIVE_SHARD_PAYLOAD_LIMIT(archiveShardSizeLimitBytes), MAX_FILE_SHARD_BYTES(fileShardSizeLimitBytes), archiveName(archiveName) {}

/**
 * Plan the archive shards adding 1 file at a time
 */
void TarWriter::addArchiveFile(const string& pathInTar, ThorImplementation::Tensor& tensor) {
    string cleanedPathInTar = cleanTarPath(pathInTar);

    size_t fileSize = tensor.getArraySizeInBytes();

    if (fileSize <= MAX_FILE_SHARD_BYTES) {
        // The file is small enough that it will not be sharded

        // Start a new shard when needed
        uint64_t wholeFileBytes = usualFileSize(fileSize);
        if (archiveShardCreationPlan.empty() || archiveShardCreationPlan.back().totalBytes + wholeFileBytes > ARCHIVE_SHARD_PAYLOAD_LIMIT) {
            uint32_t nextShardIndex = archiveShardCreationPlan.size();
            string shardPath = archiveShardTempPath(archiveName, nextShardIndex);
            archiveShardCreationPlan.emplace_back(shardPath);
        }

        // Place the file in the archive shard's plan
        ArchivePlanEntry shardPlan = {.tensor = tensor, .tensorOffsetBytes = 0UL, .numBytes = fileSize, .pathInTar = cleanedPathInTar};
        archiveShardCreationPlan.back().entries.push_back(shardPlan);
        archiveShardCreationPlan.back().totalBytes += wholeFileBytes;
        archiveShardCreationPlan.back().shardNumber = archiveShardCreationPlan.size() - 1;

        return;
    }

    // The file will be sharded in the archive, so its path will be taken to be a directory containing numbered shards in the archive.
    // Ensure the directory prefix ends with '/'
    std::string dir = cleanedPathInTar;
    if (!dir.empty() && dir.back() != '/')
        dir.push_back('/');

    size_t shardOffset = 0;
    size_t fileShardNum = 0;
    while (shardOffset < fileSize) {
        const size_t remainingBytes = fileSize - shardOffset;
        const size_t thisShardNumBytes = usualFileSize((remainingBytes > MAX_FILE_SHARD_BYTES) ? MAX_FILE_SHARD_BYTES : remainingBytes);

        // Start a new archive shard when necessary
        if (archiveShardCreationPlan.empty() ||
            archiveShardCreationPlan.back().totalBytes + thisShardNumBytes > ARCHIVE_SHARD_PAYLOAD_LIMIT) {
            uint32_t nextShardIndex = archiveShardCreationPlan.size();
            string archiveShardPath = archiveShardTempPath(archiveName, nextShardIndex);
            archiveShardCreationPlan.emplace_back(archiveShardPath);
        }

        // Place the file shard in the archive shard's plan
        // FIXME: I need file shard tests, it is not right yet
        std::string fileShardPath = dir + "shard" + std::to_string(fileShardNum);
        ArchivePlanEntry shardPlan = {
            .tensor = tensor, .tensorOffsetBytes = shardOffset, .numBytes = thisShardNumBytes, .pathInTar = fileShardPath};
        archiveShardCreationPlan.back().entries.push_back(shardPlan);
        archiveShardCreationPlan.back().totalBytes += thisShardNumBytes;
        archiveShardCreationPlan.back().shardNumber = archiveShardCreationPlan.size() - 1;

        shardOffset += thisShardNumBytes;
        ++fileShardNum;
    }
}

string TarWriter::createArchive(filesystem::path archiveDirectory, bool overwriteIfExists) {
    if (archiveShardCreationPlan.empty())
        throw runtime_error("Error: Archive creation request received, but no content has been registered to archive " +
                            archiveDirectory.string() + "/" + archiveName);

    // Create archive id for this run (32 chars, sha-like)
    string archiveId = make_archive_id_sha32();
    if (archiveDirectory.empty())
        archiveDirectory = filesystem::path(".");
    if (!filesystem::exists(archiveDirectory)) {
        filesystem::create_directories(archiveDirectory);
    }

    // Scan for any existing shard *paths* matching this prefix (any type: file/dir/symlink/etc.)
    vector<filesystem::path> matches;
    for (const auto& dir_entry : filesystem::directory_iterator(archiveDirectory)) {
        const string fname = dir_entry.path().filename().string();
        if (!is_valid_shard_filename(archiveName, fname))
            continue;

        // Anything that exists here is a conflict.
        matches.push_back(dir_entry.path());
    }

    if (!matches.empty()) {
        if (!overwriteIfExists) {
            // Mention one example match for debugging
            throw runtime_error("TarWriter: archive already exists for prefix '" + archiveName +
                                "' (found shard file: " + matches.front().string() +
                                "). Set overwriteIfExists=true to overwrite or move it out of the way or choose a different archive name.");
        }

        // Overwrite mode: remove all existing shards for this prefix to avoid stale leftovers.
        for (const auto& p : matches) {
            error_code errorCode;
            auto status = filesystem::symlink_status(p, errorCode);
            if (errorCode)
                throw runtime_error("TarWriter: symlink_status failed for: " + p.string());

            if (filesystem::is_regular_file(status) || filesystem::is_symlink(status)) {
                filesystem::remove(p, errorCode);
                if (errorCode)
                    throw runtime_error("TarWriter: failed to remove existing shard: " + p.string());
            } else {
                throw runtime_error("TarWriter: cannot overwrite non-file shard path: " + p.string() +
                                    " move it out of the way or choose a different archive name");
            }
        }
    }

    // global index (same across shards except shard_index in the JSON)
    // FIXME: changed to vector<EntryInfo>
    std::unordered_map<std::string, vector<EntryInfo>> archiveIndex;

    // Start the thread pool with writer workers, wait for it to finish running.
    ArchiveWorkerJobContext workerContext(archiveIndex, archiveIndexMutex, archiveDirectory);
    ThreadPool<ArchiveShardWriterWorker, ArchiveShardPlan, ArchiveWorkerJobContext> archiveWriterThreadPool(
        archiveShardCreationPlan, workerContext, 3);
    archiveWriterThreadPool.wait();

    uint32_t num_shards = archiveShardCreationPlan.size();

    // Build base JSON (same across shards except shard_index)
    // Use ordered_json to keep key ordering stable.
    nlohmann::ordered_json base;
    base["format_version"] = 1;
    base["checksum_alg"] = "crc32_ieee";
    base["archive_id"] = archiveId;  // 32-char “sha-like”
    base["num_shards"] = num_shards;

    nlohmann::ordered_json files = nlohmann::ordered_json::object();
    for (const auto& kv : archiveIndex) {
        const string& path = kv.first;
        const vector<EntryInfo>& entries = kv.second;

        nlohmann::ordered_json fileEntries = nlohmann::ordered_json::array();
        for (const EntryInfo& fileEntry : entries) {
            fileEntries.push_back({
                {"archive_shard", fileEntry.archiveShard},
                {"file_data_offset", fileEntry.fileDataOffset},
                {"tensor_data_offset", fileEntry.tensorDataOffset},
                {"size", fileEntry.size},
                {"crc", fileEntry.crc},
            });
        }
        files[path] = std::move(fileEntries);
    }
    base["files"] = std::move(files);

    // Append per-shard JSON + footer
    for (uint32_t shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
        nlohmann::ordered_json indexJ = base;
        indexJ["shard_index"] = shard_idx;  // per-shard field differs

        const string json_str = indexJ.dump();  // UTF-8
        uint32_t index_crc = crc32_ieee(0xFFFFFFFF, (uint8_t*)json_str.c_str(), json_str.size());

        // string archiveShardPath = strip_suffix_or_throw(archiveShardCreationPlan[shard_idx].archiveShardPath, ".incomplete", num_shards);
        string archiveShardPath = (archiveDirectory / archiveShardCreationPlan[shard_idx].archiveShardPath).string();  // + ".incomplete";
        ofstream out(archiveShardPath, ios::binary | ios::app);
        if (!out)
            throw runtime_error("createArchive: failed to reopen shard for append: " + archiveShardPath);

        // Write JSON bytes
        out.write(json_str.data(), static_cast<streamsize>(json_str.size()));
        if (!out)
            throw runtime_error("createArchive: failed writing JSON to: " + archiveShardPath);

        // Write footer: magic + json_len (LE)
        out.write(kFooterMagic, 8);
        write_u64_le(out, static_cast<uint64_t>(json_str.size()));
        write_u32_le(out, index_crc);
        write_u32_le(out, 0u);  // reserved for future (version/flags/etc)

        if (!out)
            throw runtime_error("createArchive: failed writing footer to: " + archiveShardPath);

        out.flush();
        if (!out)
            throw runtime_error("createArchive: flush failed: " + archiveShardPath);
    }

    // Ensure no files in the way, right before attempting move
    for (uint32_t shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
        string temp_path = archiveShardCreationPlan[shard_idx].archiveShardPath;
        string permanent_path = strip_suffix_or_throw(temp_path, ".incomplete", num_shards);

        std::error_code ec;
        if (filesystem::exists(permanent_path)) {
            filesystem::remove(permanent_path, ec);
            if (ec)
                throw runtime_error("createArchive: failed to rename file " + temp_path + " to: " + permanent_path);
        }
    }

    // Move files to their permanent names
    for (uint32_t shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
        string temp_path = (archiveDirectory / archiveShardCreationPlan[shard_idx].archiveShardPath).string();
        string permanent_path = strip_suffix_or_throw(temp_path, ".incomplete", num_shards);

        std::error_code ec;
        filesystem::rename(temp_path, permanent_path, ec);
        if (ec) {
            throw std::runtime_error("createArchive: rename failed: " + temp_path + " -> " + permanent_path + " (" + ec.message() + ")");
        }
    }

    return archiveId;
}

}  // namespace thor_file
