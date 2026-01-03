#include "Utilities/TarFile/TarReader.h"

using namespace std;
using json = nlohmann::json;

namespace thor_file {

TarReader::TarReader(string tarPath) : prefix_(std::move(tarPath)) { scan(); }

TarReader::~TarReader() {
    for (ShardFd& s : shard_fds_) {
        if (s.fd >= 0) {
            close(s.fd);
            s.fd = -1;
        }
    }
}

const unordered_map<string, EntryInfo>& TarReader::entries() const { return index_; }

bool TarReader::contains(string pathInTar) const {
    pathInTar = cleanTarPath(pathInTar);
    return index_.find(pathInTar) != index_.end();
}

const EntryInfo& TarReader::info(string pathInTar) const {
    pathInTar = cleanTarPath(pathInTar);
    auto it = index_.find(pathInTar);
    if (it == index_.end())
        throw runtime_error("tar entry not found: " + pathInTar);
    return it->second;
}

uint64_t TarReader::dataOffset(string pathInTar) const { return info(std::move(pathInTar)).data_offset; }

static void pread_full(int fd, const std::string& path, void* dst, uint64_t bytes, uint64_t offset) {
    uint8_t* out = static_cast<uint8_t*>(dst);
    uint64_t remaining = bytes;

    while (remaining > 0) {
        const size_t chunk =
            (remaining > static_cast<uint64_t>(SSIZE_MAX)) ? static_cast<size_t>(SSIZE_MAX) : static_cast<size_t>(remaining);

        errno = 0;
        const ssize_t n = ::pread(fd, out, chunk, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error("pread failed on " + path + ": " + std::strerror(errno));
        }
        if (n == 0) {
            throw std::runtime_error("pread hit EOF unexpectedly on " + path);
        }

        out += static_cast<size_t>(n);
        offset += static_cast<uint64_t>(n);
        remaining -= static_cast<uint64_t>(n);
    }
}

void TarReader::readFile(std::string pathInTar, void* mem, uint64_t fileSize) const {
    pathInTar = cleanTarPath(pathInTar);
    const EntryInfo& e = info(pathInTar);

    if (fileSize != e.size) {
        throw std::runtime_error("Trying to read '" + pathInTar + "' of size " + std::to_string(e.size) + " bytes, but caller provided " +
                                 std::to_string(fileSize) + " bytes.");
    }

    if (e.shard >= shard_fds_.size()) {
        throw std::runtime_error("Invalid shard index in entry for '" + pathInTar + "'");
    }

    const ShardFd& s = shard_fds_[e.shard];
    if (s.fd < 0) {
        throw std::runtime_error("Shard fd is not open for shard " + std::to_string(e.shard));
    }

    // Extra sanity: make sure read stays inside the shard file
    if (e.data_offset + e.size > s.size) {
        throw std::runtime_error("Read would exceed shard size for '" + pathInTar + "'");
    }

    pread_full(s.fd, s.path, mem, e.size, e.data_offset);

    // Validate the CRC on data usage
    const uint32_t computed_crc = crc32_ieee(0, (uint8_t*)mem, (size_t)e.size);
    if (computed_crc != e.crc_ieee) {
        throw std::runtime_error("CRC mismatch for '" + pathInTar + "' expected=" + std::to_string(e.crc_ieee) +
                                 " got=" + std::to_string(computed_crc));
    }
}

auto to_abs_norm = [](const std::string& p) -> std::string {
    filesystem::path ap = filesystem::absolute(filesystem::path(p));
    ap = ap.lexically_normal();
    return ap.string();
};

FileSliceFd TarReader::getFileSliceFd(std::string pathInTar) const {
    pathInTar = cleanTarPath(pathInTar);
    const EntryInfo& e = info(pathInTar);
    if (e.shard >= shard_fds_.size())
        throw std::runtime_error("invalid shard for: " + pathInTar);

    FileSliceFd s;
    s.fd = shard_fds_[e.shard].fd;
    s.shard_path = to_abs_norm(shard_paths_[e.shard]);
    s.shard_index = e.shard;
    s.offset = e.data_offset;
    s.size = e.size;
    s.crc_ieee = e.crc_ieee;
    return s;
}

static int open_readonly_fd(const std::string& path, uint64_t& size_out) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("open failed: " + path + ": " + std::strerror(errno));
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error("fstat failed: " + path + ": " + std::strerror(e));
    }

    size_out = static_cast<uint64_t>(st.st_size);
    return fd;
}

static std::string shard0_path_any(const std::string& prefix) {
    const std::string numbered0 = shard_filename(prefix, 0);  // prefix + ".000000.thor"
    const std::string single0 = prefix + ".thor";

    if (std::filesystem::exists(numbered0))
        return numbered0;
    if (std::filesystem::exists(single0))
        return single0;

    throw std::runtime_error("TarReader::scan: missing shard0 (" + numbered0 + " or " + single0 + ")");
}

void TarReader::scan() {
    namespace fs = std::filesystem;

    index_.clear();
    shard_paths_.clear();
    shard_fds_.clear();
    archive_id_.clear();
    num_shards_ = 0;

    // Step 1: Read index JSON from shard0 (either prefix.000000.thor or prefix.thor)
    const std::string first_path = shard0_path_any(prefix_);

    const std::string ref_json_str = read_footer_json(first_path);

    json j0;
    try {
        j0 = json::parse(ref_json_str);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("TarReader::scan: failed parsing JSON index: ") + e.what());
    }

    require(j0.contains("shard_index"), "index JSON missing 'shard_index'");
    const uint32_t shard_index0 = j0.at("shard_index").get<uint32_t>();
    require(shard_index0 == 0, "shard_index in shard 0 index must be 0");

    require(j0.contains("format_version"), "index JSON missing 'format_version'");
    const uint32_t ver = j0.at("format_version").get<uint32_t>();
    require(ver == 1, "unsupported format_version: " + std::to_string(ver));

    require(j0.contains("checksum_alg"), "index JSON missing 'checksum_alg'");
    const std::string alg = j0.at("checksum_alg").get<std::string>();
    require(alg == "crc32_ieee", "unsupported checksum_alg: " + alg);

    // Compare later ignoring shard_index
    json j0_cmp = j0;
    j0_cmp.erase("shard_index");

    require(j0_cmp.is_object(), "index JSON must be an object");
    require(j0_cmp.contains("archive_id"), "index JSON missing 'archive_id'");
    require(j0_cmp.contains("num_shards"), "index JSON missing 'num_shards'");
    require(j0_cmp.contains("files"), "index JSON missing 'files'");

    const std::string archive_id = j0_cmp.at("archive_id").get<std::string>();
    require(is_sha32(archive_id), "archive_id must look like 32 hex chars");

    const uint32_t num_shards = j0_cmp.at("num_shards").get<uint32_t>();
    require(num_shards >= 1, "num_shards must be >= 1");

    archive_id_ = archive_id;
    num_shards_ = num_shards;

    // Step 2: Determine expected shard paths
    shard_paths_.reserve(num_shards_);

    if (num_shards_ == 1) {
        // single-shard convention is prefix.thor
        const std::string p0 = prefix_ + ".thor";
        require(fs::exists(p0), "missing single shard: " + p0);
        shard_paths_.push_back(p0);

        // also ensure the numbered form isn't present (optional strictness)
        // require(!fs::exists(shard_filename(prefix_, 0)), "unexpected numbered shard exists for single-shard archive");
    } else {
        // multi-shard convention is numbered
        for (uint32_t i = 0; i < num_shards_; ++i) {
            const std::string p = shard_filename(prefix_, i);
            require(fs::exists(p), "missing shard " + std::to_string(i) + ": " + p);
            shard_paths_.push_back(p);
        }
    }

    // Step 3: Ensure each shard has identical index except shard_index
    for (uint32_t i = 0; i < num_shards_; ++i) {
        const std::string p = shard_paths_[i];
        const std::string i_json_str = read_footer_json(p);

        json ji = json::parse(i_json_str);

        require(ji.contains("archive_id"), "index JSON missing 'archive_id' in shard " + std::to_string(i));
        const std::string jiArchiveId = ji.at("archive_id").get<std::string>();

        require(archive_id == jiArchiveId,
                "Archive ids mismatch between shards: shard0=" + archive_id + " shard" + std::to_string(i) + "=" + jiArchiveId);

        require(ji.contains("shard_index"), "index JSON missing 'shard_index' in: " + p);
        const uint32_t shard_idx = ji.at("shard_index").get<uint32_t>();
        require(shard_idx == i, "shard_index mismatch in: " + p);

        ji.erase("shard_index");
        require(j0_cmp == ji, "index JSON differs across shards (expected identical except shard_index): " + p);
    }

    // Step 4: Load entries
    const auto& files = j0_cmp.at("files");
    require(files.is_object(), "'files' must be an object mapping path->info");

    for (auto it = files.begin(); it != files.end(); ++it) {
        const std::string path_in_archive = it.key();
        const json& info = it.value();

        require(info.is_object(), "file info must be an object for: " + path_in_archive);
        require(info.contains("shard"), "missing 'shard' for: " + path_in_archive);
        require(info.contains("data_offset"), "missing 'data_offset' for: " + path_in_archive);
        require(info.contains("size"), "missing 'size' for: " + path_in_archive);
        require(info.contains("crc_ieee"), "missing 'size' for: " + path_in_archive);

        EntryInfo e{};
        e.shard = info.at("shard").get<uint32_t>();
        e.data_offset = info.at("data_offset").get<uint64_t>();
        e.size = info.at("size").get<uint64_t>();
        e.crc_ieee = info.at("crc_ieee").get<uint32_t>();

        require(e.shard < num_shards_, "entry refers to invalid shard for: " + path_in_archive);

        const std::string& shard_path = shard_paths_[e.shard];
        const uint64_t shard_size = static_cast<uint64_t>(fs::file_size(shard_path));

        require(e.data_offset < shard_size, "data_offset beyond shard size for: " + path_in_archive);
        require(e.data_offset + e.size <= shard_size, "data read would exceed shard size for: " + path_in_archive);

        require(!index_.contains(path_in_archive), "file path duplicated in archive: " + path_in_archive);
        index_[path_in_archive] = e;
    }

    // Step 5: Open all shards once and keep fds for fast pread()
    shard_fds_.reserve(num_shards_);
    for (uint32_t i = 0; i < num_shards_; ++i) {
        ShardFd s;
        s.path = shard_paths_[i];
        s.fd = open_readonly_fd(s.path, s.size);

        shard_fds_.push_back(std::move(s));
    }
}

void TarReader::verifyAll() const {
    if (index_.empty())
        return;

    // Collect entries and sort by (shard, data_offset) to improve sequentiality.
    struct Item {
        std::string path;
        EntryInfo info;
    };
    std::vector<Item> items;
    items.reserve(index_.size());
    for (const auto& kv : index_) {
        items.push_back(Item{kv.first, kv.second});
    }

    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.info.shard != b.info.shard)
            return a.info.shard < b.info.shard;
        return a.info.data_offset < b.info.data_offset;
    });

    // Reuse a buffer sized to the largest file we see.
    uint64_t max_size = 0;
    for (const auto& it : items) {
        max_size = std::max(max_size, it.info.size);
    }
    std::vector<uint8_t> buf;
    buf.resize(static_cast<size_t>(max_size));

    // Verify each entry. This leverages your existing validation code path.
    for (const auto& it : items) {
        if (it.info.size == 0) {
            // Still validate the index expectation is "empty" CRC if you care.
            // Typically crc32("") == 0, which matches your writer usage.
            continue;
        }

        // readFile will throw on I/O problems or CRC mismatch
        readFile(it.path, buf.data(), it.info.size);
    }
}

}  // namespace thor_file
