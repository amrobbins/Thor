#include "Utilities/TarFile/TarWriter.h"

#include "Crc32.h"
#include "Crc32c.h"

using namespace std;

namespace thor_file {

// -------------------- TarWriter (PAX, uncompressed) --------------------

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
    // patter base.thor.tar
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

TarWriter::TarWriter(string archiveName, filesystem::path archiveDirectory, bool overwriteIfExists, uint64_t shard_payload_limit_bytes) {
    shard_payload_limit_ = shard_payload_limit_bytes;
    finished_ = false;
    cur_ = 0;
    index_.clear();
    shards_.clear();

    if (archiveDirectory.empty())
        archiveDirectory = filesystem::path(".");
    if (!filesystem::exists(archiveDirectory)) {
        filesystem::create_directories(archiveDirectory);
    }

    prefix_ = (archiveDirectory / archiveName).string();

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
            throw runtime_error("TarWriter: archive already exists for prefix '" + prefix_ +
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

    // Create archive id for this run (32 chars, sha-like)
    archive_id_ = make_archive_id_sha32();

    // Open first shard and initialize bookkeeping
    openShard_(0);
    cur_ = 0;
}

TarWriter::~TarWriter() {
    try {
        if (!finished_ && !shards_.empty())
            finishArchive();
    } catch (...) {
        // swallow (or log)
    }
}

static uint64_t round_up_512(uint64_t n) { return (n + 511ull) & ~511ull; }

static string shard_temp_path(const string& prefix, uint32_t shard_idx) {
    char buf[64];
    snprintf(buf, sizeof(buf), ".%06u.thor.tar.incomplete", shard_idx);
    return prefix + string(buf);
}

la_ssize_t TarWriter::write_cb(archive*, void* client_data, const void* buffer, size_t length) {
    auto* s = static_cast<ShardState*>(client_data);
    s->out.write(static_cast<const char*>(buffer), static_cast<streamsize>(length));
    if (!s->out)
        return -1;
    s->pos += static_cast<uint64_t>(length);
    return static_cast<la_ssize_t>(length);
}

int TarWriter::close_cb(archive*, void* client_data) {
    auto* s = static_cast<ShardState*>(client_data);
    s->out.flush();
    return s->out ? ARCHIVE_OK : ARCHIVE_FATAL;
}

void TarWriter::openShard_(uint32_t shard_index) {
    if (shard_index != shards_.size())
        throw runtime_error("openShard_: shard_index out of sequence");

    shards_.emplace_back();
    ShardState& st = shards_.back();

    if (shard_index == 0) {
        st.path = prefix_ + ".thor.tar.incomplete";
    } else {
        if (shard_index == 1) {
            // Now that there will be multiple shards, rename the first one to be a member of the group
            const filesystem::path old_path = filesystem::path(prefix_ + ".thor.tar.incomplete");
            const filesystem::path new_path = filesystem::path(prefix_ + ".000000.thor.tar.incomplete");
            std::error_code ec;
            shards_[0].path = new_path.string();
            filesystem::rename(old_path, new_path, ec);
            if (ec) {
                throw std::runtime_error("promoteSingleShardToNumbered_: rename failed: " + old_path.string() + " -> " + new_path.string() +
                                         " (" + ec.message() + ")");
            }
        }

        st.path = shard_temp_path(prefix_, shard_index);
    }
    st.out.open(st.path, ios::binary | ios::trunc);
    if (!st.out)
        throw runtime_error("failed to open shard for write: " + st.path);
    // 16 MB IO buffer to support NVMe speeds.
    st.io_buf.resize(16 * 1024 * 1024);
    st.out.rdbuf()->pubsetbuf(st.io_buf.data(), static_cast<std::streamsize>(st.io_buf.size()));

    st.a = archive_write_new();
    if (!st.a)
        throw runtime_error("archive_write_new failed");

    archive_write_set_format_pax_restricted(st.a);
    archive_write_set_bytes_per_block(st.a, 0);      // no internal blocking/buffering
    archive_write_set_bytes_in_last_block(st.a, 1);  // don't pad the final block weirdly

    if (archive_write_open(st.a, &st, /*open*/ nullptr, &TarWriter::write_cb, &TarWriter::close_cb) != ARCHIVE_OK) {
        const char* err = archive_error_string(st.a);
        archive_write_free(st.a);
        st.a = nullptr;
        throw runtime_error(string("archive_write_open failed: ") + (err ? err : "(unknown)"));
    }
}

void TarWriter::closeShard_(uint32_t shard_index) {
    auto& s = shards_.at(shard_index);
    if (s.closed)
        return;

    if (archive_write_close(s.a) != ARCHIVE_OK) {
        const char* err = archive_error_string(s.a);
        archive_write_free(s.a);
        s.a = nullptr;
        throw runtime_error(string("archive_write_close failed: ") + (err ? err : "(unknown)"));
    }
    archive_write_free(s.a);
    s.a = nullptr;

    s.out.close();
    if (!s.out)
        throw runtime_error("failed closing shard output: " + s.path);
    s.io_buf.clear();

    s.closed = true;
}

void TarWriter::addArchiveFile(string path_in_tar, const void* data, size_t size, int permissions, time_t mtime) {
    if (finished_)
        throw runtime_error("TarWriter::add_bytes: archive already finished");

    path_in_tar = cleanTarPath(path_in_tar);

    uint32_t data_crc = Crc32c::compute((uint8_t*)data, size);

    // Shard rollover policy:
    // With PAX, header overhead varies (extra 512B blocks for extended headers), so we estimate
    // conservatively. Offsets are still exact because we track actual bytes written.
    auto estimated_entry_bytes = [&](uint64_t file_size) -> uint64_t {
        // 1 tar header block + payload padded to 512
        uint64_t base = 512ull + round_up_512(file_size);

        // PAX slack: worst-case can be more than 1 block if you have many attrs / long paths.
        // Adjust if you want tighter. This only affects when we roll shards, not correctness.
        uint64_t pax_slack = 4096ull;  // 8 blocks of slack

        return base + pax_slack;
    };

    // Reserve space for end-of-archive marker (2x 512 blocks)
    const uint64_t end_marker = 1024ull;

    ShardState& s = shards_[cur_];

    if (shard_payload_limit_ > 0 && s.pos > 0) {
        const uint64_t est = estimated_entry_bytes(static_cast<uint64_t>(size));

        // Note: finishArchive() will append JSON+footer after we close tar;
        // that will exceed shard_payload_limit_ by (json+footer). If you need strict
        // on-disk size limits, subtract a reserved trailer budget here.
        if (s.pos + est + end_marker > shard_payload_limit_) {
            closeShard_(cur_);
            openShard_(cur_ + 1);
            ++cur_;
        }
    }

    ShardState& s2 = shards_[cur_];

    archive_entry* entry = archive_entry_new();
    if (!entry)
        throw runtime_error("archive_entry_new failed");

    archive_entry_set_pathname(entry, path_in_tar.c_str());
    archive_entry_set_filetype(entry, AE_IFREG);
    archive_entry_set_perm(entry, permissions);
    archive_entry_set_size(entry, static_cast<la_int64_t>(size));
    archive_entry_set_mtime(entry, static_cast<time_t>(mtime), 0);

    if (archive_write_header(s2.a, entry) != ARCHIVE_OK) {
        const char* err = archive_error_string(s2.a);
        archive_entry_free(entry);
        throw runtime_error(string("archive_write_header failed: ") + (err ? err : "(unknown)"));
    }

    const uint64_t data_off_after_header = s2.pos;

    // Write payload
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t remaining = size;
    while (remaining > 0) {
        la_ssize_t wrote = archive_write_data(s2.a, p, remaining);
        if (wrote < 0) {
            const char* err = archive_error_string(s2.a);
            archive_entry_free(entry);
            throw runtime_error(string("archive_write_data failed: ") + (err ? err : "(unknown)"));
        }
        p += static_cast<size_t>(wrote);
        remaining -= static_cast<size_t>(wrote);
    }

    if (archive_write_finish_entry(s2.a) != ARCHIVE_OK) {
        const char* err = archive_error_string(s2.a);
        archive_entry_free(entry);
        throw runtime_error(string("archive_write_finish_entry failed: ") + (err ? err : "(unknown)"));
    }

    archive_entry_free(entry);

    // Record in global index. This (shard, data_offset, size) is what your reader needs.
    EntryInfo info;
    info.shard = cur_;
    info.data_offset = data_off_after_header;
    info.size = static_cast<uint64_t>(size);
    info.crc = data_crc;

    index_[path_in_tar] = info;
}

static std::string strip_suffix_or_throw(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size() || s.compare(s.size() - suffix.size(), suffix.size(), suffix) != 0) {
        throw std::runtime_error("expected suffix '" + suffix + "' on: " + s);
    }
    return s.substr(0, s.size() - suffix.size());
}

void TarWriter::finishArchive() {
    if (finished_)
        return;
    if (shards_.empty())
        throw runtime_error("TarWriter::finishArchive: no shards to finish");

    // Close all shards (writes tar end markers).
    for (uint32_t i = 0; i < shards_.size(); ++i) {
        closeShard_(i);
    }

    const uint32_t num_shards = static_cast<uint32_t>(shards_.size());

    // Build base JSON (same across shards except shard_index)
    // Use ordered_json to keep key ordering stable if you ever compare strings.
    nlohmann::ordered_json base;
    base["format_version"] = 1;
    base["checksum_alg"] = "crc32c";
    base["archive_id"] = archive_id_;  // 32-char “sha-like”
    base["num_shards"] = num_shards;

    nlohmann::ordered_json files = nlohmann::ordered_json::object();
    for (const auto& kv : index_) {
        const string& path = kv.first;
        const EntryInfo& e = kv.second;
        files[path] = {{"shard", e.shard}, {"data_offset", e.data_offset}, {"size", e.size}, {"crc", e.crc}};
    }
    base["files"] = std::move(files);

    // Append per-shard JSON + footer
    for (uint32_t shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
        nlohmann::ordered_json j = base;
        j["shard_index"] = shard_idx;  // per-shard field differs

        const string json_str = j.dump();  // UTF-8
        uint32_t index_crc = Crc32c::compute((uint8_t*)json_str.c_str(), json_str.size());

        ofstream out(shards_[shard_idx].path, ios::binary | ios::app);
        if (!out)
            throw runtime_error("finishArchive: failed to reopen shard for append: " + shards_[shard_idx].path);

        // Write JSON bytes
        out.write(json_str.data(), static_cast<streamsize>(json_str.size()));
        if (!out)
            throw runtime_error("finishArchive: failed writing JSON to: " + shards_[shard_idx].path);

        // Write footer: magic + json_len (LE)
        out.write(kFooterMagic, 8);
        write_u64_le(out, static_cast<uint64_t>(json_str.size()));
        write_u32_le(out, index_crc);
        write_u32_le(out, 0u);  // reserved for future (version/flags/etc)

        if (!out)
            throw runtime_error("finishArchive: failed writing footer to: " + shards_[shard_idx].path);

        out.flush();
        if (!out)
            throw runtime_error("finishArchive: flush failed: " + shards_[shard_idx].path);
    }

    // Ensure no files in the way, right before attempting move
    for (uint32_t shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
        string temp_path = shards_[shard_idx].path;
        string permanent_path = strip_suffix_or_throw(temp_path, ".incomplete");

        std::error_code ec;
        if (filesystem::exists(permanent_path)) {
            filesystem::remove(permanent_path, ec);
            if (ec)
                throw runtime_error("finishArchive: failed to rename file " + temp_path + " to: " + permanent_path);
        }
    }

    // Move files to their permanent names
    for (uint32_t shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
        string temp_path = shards_[shard_idx].path;
        string permanent_path = strip_suffix_or_throw(temp_path, ".incomplete");

        std::error_code ec;
        filesystem::rename(temp_path, permanent_path, ec);
        if (ec) {
            throw std::runtime_error("finishArchive: rename failed: " + temp_path + " -> " + permanent_path + " (" + ec.message() + ")");
        }
    }

    shards_.clear();
    finished_ = true;
}

}  // namespace thor_file
