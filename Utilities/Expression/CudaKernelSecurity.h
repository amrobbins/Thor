#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace ThorImplementation {

struct CudaKernelSignatureVerificationResult {
    bool verified = false;
    std::string message;
};

struct CudaKernelOutOfBandKeys {
    std::string signing_public_key;
    std::string source_decryption_key;
};

struct CudaKernelSourceInspection {
    std::string name;
    std::string entrypoint;
    std::string source;
    std::string compiled_source;
    std::string compiled_source_hash;
    bool loaded_source_compilation_allowed = false;
    bool source_encrypted = false;
    std::string source_encryption_algorithm;
    std::string source_decryption_key_fingerprint;
    std::string signature_algorithm;
    std::string signing_public_key_fingerprint;
    std::string signature;
};

[[nodiscard]] nlohmann::json cudaKernelSourceInspectionToJson(const CudaKernelSourceInspection& info);
[[nodiscard]] nlohmann::json cudaKernelSourceInspectionListToJson(const std::vector<CudaKernelSourceInspection>& infos);

[[nodiscard]] nlohmann::json cudaKernelManifestFromExpressionJson(const nlohmann::json& expression_json);
[[nodiscard]] std::string cudaKernelManifestCanonicalBytes(const nlohmann::json& manifest);
[[nodiscard]] CudaKernelOutOfBandKeys cudaKernelGenerateAndAttachManifestSignature(nlohmann::json& expression_json);
[[nodiscard]] std::vector<CudaKernelOutOfBandKeys> cudaKernelGenerateAndAttachManifestSignatures(nlohmann::json& root_json);
[[nodiscard]] std::vector<CudaKernelOutOfBandKeys> collectCudaKernelOutOfBandKeys(const nlohmann::json& j);
[[nodiscard]] bool cudaKernelExpressionJsonContainsEncryptedSources(const nlohmann::json& expression_json);
[[nodiscard]] bool cudaKernelExpressionJsonContainsPlaintextSources(const nlohmann::json& expression_json);
[[nodiscard]] nlohmann::json cudaKernelDecryptSerializedCudaSources(
    const nlohmann::json& expression_json,
    const std::string& trusted_source_decryption_key);
[[nodiscard]] CudaKernelSignatureVerificationResult cudaKernelVerifyManifestSignature(
    const nlohmann::json& expression_json,
    const std::string& trusted_public_key);
[[nodiscard]] std::vector<std::string> collectCudaKernelSigningPublicKeys(const nlohmann::json& j);
[[nodiscard]] std::vector<CudaKernelSourceInspection> collectCudaKernelSourceInfo(const nlohmann::json& j);
[[nodiscard]] nlohmann::json collectCudaKernelSourceInfoJson(const nlohmann::json& j);
[[nodiscard]] std::string cudaKernelLoadedModelSafetyDisclaimer();

}  // namespace ThorImplementation
