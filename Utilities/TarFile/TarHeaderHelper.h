#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// --- TAR helpers ---

static constexpr uint32_t TAR_BLOCK = 512;

// octal field writer (null-terminated or space-terminated depending on field)
static void tar_write_octal(char* dst, size_t width, uint64_t value) {
    // TAR fields are typically width bytes including trailing NUL or space.
    // We'll format as octal with leading zeros and trailing NUL.
    // If value doesn't fit, caller should use PAX.
    // width includes the trailing '\0' (common for these fields).
    if (width < 2)
        throw std::runtime_error("tar_write_octal: width too small");

    // max digits we can store before terminating NUL
    size_t digits = width - 1;
    // produce octal string right-aligned
    for (size_t i = 0; i < digits; ++i)
        dst[i] = '0';
    dst[digits] = '\0';

    size_t pos = digits;
    uint64_t v = value;
    while (v != 0 && pos > 0) {
        --pos;
        dst[pos] = char('0' + (v & 7));
        v >>= 3;
    }
    if (v != 0) {
        // didn't fit
        throw std::runtime_error("tar_write_octal: value does not fit field");
    }
}

// basic ustar header (512 bytes)
struct TarHeader {
    char name[100];      // 0
    char mode[8];        // 100
    char uid[8];         // 108
    char gid[8];         // 116
    char size[12];       // 124
    char mtime[12];      // 136
    char chksum[8];      // 148
    char typeflag;       // 156
    char linkname[100];  // 157
    char magic[6];       // 257
    char version[2];     // 263
    char uname[32];      // 265
    char gname[32];      // 297
    char devmajor[8];    // 329
    char devminor[8];    // 337
    char prefix[155];    // 345
    char pad[12];        // 500
};
static_assert(sizeof(TarHeader) == 512, "TarHeader must be 512 bytes");

// checksum: set chksum field to spaces, sum of all bytes of header as unsigned char
static uint32_t tar_checksum(const TarHeader& h) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&h);
    uint32_t sum = 0;
    for (size_t i = 0; i < 512; ++i)
        sum += p[i];
    return sum;
}

static void tar_finalize_checksum(TarHeader& h) {
    // chksum field is 8 bytes. Convention: 6 octal digits, NUL, space.
    std::memset(h.chksum, ' ', sizeof(h.chksum));
    uint32_t sum = tar_checksum(h);

    // write 6 digits octal
    char tmp[8]{};
    // tmp: 7 bytes including NUL
    tar_write_octal(tmp, 7, sum);
    // copy first 6 digits
    std::memcpy(h.chksum, tmp, 6);
    h.chksum[6] = '\0';
    h.chksum[7] = ' ';
}

// Splits a path into ustar name/prefix if possible.
// Returns true if it fits; false if not.
static bool tar_split_ustar_name_prefix(const std::string& path, std::array<char, 100>& nameOut, std::array<char, 155>& prefixOut) {
    nameOut.fill('\0');
    prefixOut.fill('\0');

    if (path.size() <= 100) {
        std::memcpy(nameOut.data(), path.data(), path.size());
        return true;
    }
    // Try ustar prefix+name split at a '/'
    // prefix <= 155, name <= 100
    // Choose the last '/' that makes it fit.
    size_t bestSlash = std::string::npos;
    for (size_t i = path.size(); i-- > 0;) {
        if (path[i] == '/') {
            size_t prefixLen = i;  // [0..i-1]
            size_t nameLen = path.size() - (i + 1);
            if (prefixLen <= 155 && nameLen <= 100) {
                bestSlash = i;
                break;
            }
        }
    }
    if (bestSlash == std::string::npos)
        return false;

    size_t prefixLen = bestSlash;
    size_t nameLen = path.size() - (bestSlash + 1);
    std::memcpy(prefixOut.data(), path.data(), prefixLen);
    std::memcpy(nameOut.data(), path.data() + bestSlash + 1, nameLen);
    return true;
}

// PAX record line: "<len> <key>=<value>\n" where len includes the digits, space, content, newline.
static std::string pax_record(const std::string& key, const std::string& value) {
    std::string body = key + "=" + value + "\n";
    // compute length with digit count iteration
    // len = digits(len) + 1 + body.size()
    size_t len = 0;
    for (;;) {
        size_t digits = std::to_string(len ? len : (body.size() + 3)).size();  // seed guess
        size_t newLen = digits + 1 + body.size();
        if (newLen == len)
            break;
        len = newLen;
    }
    return std::to_string(len) + " " + body;
}

struct TarSeparatorBuildResult {
    uint32_t totalBytesAppended;  // bytes added into out buffer
    uint32_t paxBytes;            // bytes for pax header+payload+pad (0 if none)
    uint32_t fileHeaderBytes;     // 512 always
    uint32_t payloadPadBytes;     // pad after file payload to 512 boundary
};

// Builds (optional PAX) + file header for the *next* file,
// AND also includes the *payload padding* for the *previous* file.
// You pass in prevFileSize (payload bytes of previous file) so it can add TAR payload padding.
// You pass in tailBytesIn so it can append into tailAndTarSeparator[tailBytesIn..].
// Returns new tail byte count (tailBytesIn + appended).
//
// Usage in your flow after dumping prev file payload:
//   tailBytes = createTarSeparator(nextPath, nextFileSize, prevFileSize, tailBytes, scratch, scratchCap);
//   memcpy(nextBounceBuf, scratch, tailBytes);
//   prefetch payload after tailBytes...
//
static uint32_t createTarSeparator(const std::string& nextPathInTar,
                                   uint64_t nextFileSize,
                                   uint64_t prevFileSize,
                                   uint32_t tailBytesIn,
                                   uint8_t* tailAndTarSeparator,
                                   uint32_t scratchCap,
                                   TarSeparatorBuildResult* dbg = nullptr) {
    if (tailBytesIn > scratchCap)
        throw std::runtime_error("createTarSeparator: tailBytesIn > scratchCap");

    uint8_t* out = tailAndTarSeparator + tailBytesIn;
    uint32_t remainingCap = scratchCap - tailBytesIn;

    auto appendBytes = [&](const void* p, uint32_t n) {
        if (n > remainingCap)
            throw std::runtime_error("createTarSeparator: scratch buffer too small");
        std::memcpy(out, p, n);
        out += n;
        remainingCap -= n;
    };
    auto appendZero = [&](uint32_t n) {
        if (n > remainingCap)
            throw std::runtime_error("createTarSeparator: scratch buffer too small");
        std::memset(out, 0, n);
        out += n;
        remainingCap -= n;
    };

    TarSeparatorBuildResult local{};
    local.totalBytesAppended = 0;

    // 1) Add padding after *previous* file payload to 512-byte boundary.
    uint32_t payloadPad = uint32_t((TAR_BLOCK - (prevFileSize % TAR_BLOCK)) % TAR_BLOCK);
    if (payloadPad) {
        appendZero(payloadPad);
        local.payloadPadBytes = payloadPad;
        local.totalBytesAppended += payloadPad;
    }

    // 2) Decide if we need PAX for the NEXT file.
    // We can fit name/prefix into ustar without PAX sometimes. If not, use PAX path.
    // Also if size can't fit 12-octal field, we should use PAX "size".
    std::array<char, 100> name{};
    std::array<char, 155> prefix{};
    bool fitsUstarName = tar_split_ustar_name_prefix(nextPathInTar, name, prefix);

    bool sizeFitsOctal = true;
    try {
        char tmp[12];
        tar_write_octal(tmp, sizeof(tmp), nextFileSize);
    } catch (...) {
        sizeFitsOctal = false;
    }

    bool needPax = (!fitsUstarName) || (!sizeFitsOctal);

    std::vector<uint8_t> paxPayload;
    if (needPax) {
        std::string paxText;
        paxText += pax_record("path", nextPathInTar);
        if (!sizeFitsOctal)
            paxText += pax_record("size", std::to_string(nextFileSize));

        paxPayload.assign(paxText.begin(), paxText.end());

        // PAX extended header itself is a normal tar header with typeflag 'x'
        TarHeader paxHdr{};
        std::memset(&paxHdr, 0, sizeof(paxHdr));

        // name: something short; doesn't matter much
        const std::string paxName = "PaxHeader";
        std::memcpy(paxHdr.name, paxName.data(), std::min<size_t>(paxName.size(), sizeof(paxHdr.name)));

        std::memcpy(paxHdr.magic, "ustar", 5);
        paxHdr.magic[5] = '\0';
        paxHdr.version[0] = '0';
        paxHdr.version[1] = '0';
        paxHdr.typeflag = 'x';

        // mode/uid/gid/mtime can be zeros
        tar_write_octal(paxHdr.mode, sizeof(paxHdr.mode), 0644);
        tar_write_octal(paxHdr.uid, sizeof(paxHdr.uid), 0);
        tar_write_octal(paxHdr.gid, sizeof(paxHdr.gid), 0);
        tar_write_octal(paxHdr.mtime, sizeof(paxHdr.mtime), 0);

        // pax payload size
        tar_write_octal(paxHdr.size, sizeof(paxHdr.size), paxPayload.size());

        tar_finalize_checksum(paxHdr);

        appendBytes(&paxHdr, 512);
        local.paxBytes += 512;
        local.totalBytesAppended += 512;

        // payload then pad to 512
        appendBytes(paxPayload.data(), static_cast<uint32_t>(paxPayload.size()));
        local.paxBytes += static_cast<uint32_t>(paxPayload.size());
        local.totalBytesAppended += static_cast<uint32_t>(paxPayload.size());

        uint32_t paxPad = uint32_t((TAR_BLOCK - (paxPayload.size() % TAR_BLOCK)) % TAR_BLOCK);
        if (paxPad) {
            appendZero(paxPad);
            local.paxBytes += paxPad;
            local.totalBytesAppended += paxPad;
        }
    }

    // 3) File header for NEXT file (ustar)
    TarHeader fileHdr{};
    std::memset(&fileHdr, 0, sizeof(fileHdr));

    // If we used PAX for path, we can put something truncated/empty here; tar readers use pax "path".
    // But it's still good to put a reasonable fallback name.
    if (fitsUstarName) {
        std::memcpy(fileHdr.name, name.data(), sizeof(fileHdr.name));
        std::memcpy(fileHdr.prefix, prefix.data(), sizeof(fileHdr.prefix));
    } else {
        // fallback: last 100 bytes of path
        const std::string tail = (nextPathInTar.size() <= 100) ? nextPathInTar : nextPathInTar.substr(nextPathInTar.size() - 100);
        std::memcpy(fileHdr.name, tail.data(), tail.size());
    }

    std::memcpy(fileHdr.magic, "ustar", 5);
    fileHdr.magic[5] = '\0';
    fileHdr.version[0] = '0';
    fileHdr.version[1] = '0';

    fileHdr.typeflag = '0';  // regular file

    tar_write_octal(fileHdr.mode, sizeof(fileHdr.mode), 0644);
    tar_write_octal(fileHdr.uid, sizeof(fileHdr.uid), 0);
    tar_write_octal(fileHdr.gid, sizeof(fileHdr.gid), 0);
    tar_write_octal(fileHdr.mtime, sizeof(fileHdr.mtime), 0);

    // If size doesn't fit octal, we relied on PAX size; tar field can be zeros or best-effort.
    if (sizeFitsOctal) {
        tar_write_octal(fileHdr.size, sizeof(fileHdr.size), nextFileSize);
    } else {
        tar_write_octal(fileHdr.size, sizeof(fileHdr.size), 0);
    }

    tar_finalize_checksum(fileHdr);

    appendBytes(&fileHdr, 512);
    local.fileHeaderBytes = 512;
    local.totalBytesAppended += 512;

    if (dbg)
        *dbg = local;

    return tailBytesIn + local.totalBytesAppended;
}

static uint32_t appendTarEndOfArchive(uint64_t lastFileSize, uint32_t tailBytesIn, uint8_t* tailAndTarSeparator, uint32_t scratchCap) {
    // TAR finalize bytes are:
    //   A) pad last file payload to 512 boundary
    //   B) two 512-byte zero blocks (1024 bytes)
    // For O_DIRECT convenience, we additionally pad so the appended region ends on a 4KB boundary.
    //
    // Inputs:
    // - lastFileSize: payload byte count of the LAST file in the archive (not including its 512 header)
    // - tailBytesIn: bytes already present in tailAndTarSeparator (carry/tar pieces)
    //
    // Returns:
    // - new tail byte count (tailBytesIn + appended)
    if (tailBytesIn > scratchCap)
        throw std::runtime_error("appendTarEndOfArchive: tailBytesIn > scratchCap");

    // A) pad last payload to 512
    uint32_t pad512 = static_cast<uint32_t>((512 - (lastFileSize % 512)) % 512);

    // B) end-of-archive markers
    uint32_t eoa = 2 * 512;  // 1024

    // Minimum bytes we must append:
    uint32_t minAppend = pad512 + eoa;

    // Now choose appendBytes >= minAppend so that (tailBytesIn + appendBytes) % 4096 == 0.
    // (Assume 4096 is the O_DIRECT alignment you are enforcing.)
    uint32_t mod = tailBytesIn & (4096 - 1);
    uint32_t want = (4096 - mod) & (4096 - 1);  // bytes to reach next 4KB boundary (0 if already aligned)

    // If already aligned, we still need minAppend; make want be 4096 so we can add full blocks cleanly.
    if (want == 0)
        want = 4096;

    uint32_t appendBytes = want;
    while (appendBytes < minAppend)
        appendBytes += 4096;

    if (tailBytesIn + appendBytes > scratchCap) {
        throw std::runtime_error("appendTarEndOfArchive: scratch buffer too small");
    }

    std::memset(tailAndTarSeparator + tailBytesIn, 0, appendBytes);
    return tailBytesIn + appendBytes;
}
