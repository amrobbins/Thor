#include "DeepLearning/Api/Data/DatasetId.h"

#include <openssl/evp.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace Thor {
namespace {

bool isCanonicalUuid(std::string_view value) {
    if (value.size() != 36) {
        return false;
    }
    for (size_t i = 0; i < value.size(); ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (value[i] != '-') {
                return false;
            }
            continue;
        }
        const unsigned char ch = static_cast<unsigned char>(value[i]);
        if (!std::isdigit(ch) && !(ch >= 'a' && ch <= 'f')) {
            return false;
        }
    }
    return true;
}

std::string formatUuid(const std::array<uint8_t, 16> &bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            out << '-';
        }
        out << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return out.str();
}

std::array<uint8_t, 32> sha256(std::string_view material) {
    std::array<uint8_t, 32> digest{};
    unsigned int digestLength = 0;
    using ContextPtr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
    ContextPtr context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
    if (!context) {
        throw std::runtime_error("DatasetId failed to allocate SHA-256 context.");
    }
    if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(context.get(), material.data(), material.size()) != 1 ||
        EVP_DigestFinal_ex(context.get(), digest.data(), &digestLength) != 1 || digestLength != digest.size()) {
        throw std::runtime_error("DatasetId failed to compute SHA-256 digest.");
    }
    return digest;
}

}  // namespace

DatasetId::DatasetId(std::string value) : value(std::move(value)) {
    if (!isCanonicalUuid(this->value)) {
        throw std::runtime_error("DatasetId must be a canonical lower-case UUID string.");
    }
}

DatasetId DatasetId::generate() {
    std::array<uint8_t, 16> bytes{};
    std::random_device random;
    for (uint8_t &byte : bytes) {
        byte = static_cast<uint8_t>(random());
    }
    bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0fU) | 0x40U);
    bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3fU) | 0x80U);
    return DatasetId(formatUuid(bytes));
}

DatasetId DatasetId::fromStableMaterial(std::string_view material) {
    const std::array<uint8_t, 32> digest = sha256(material);
    std::array<uint8_t, 16> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = digest[i];
    }
    bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0fU) | 0x80U);
    bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3fU) | 0x80U);
    return DatasetId(formatUuid(bytes));
}

}  // namespace Thor
