#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace ThorImplementation {

struct CudaKernelSignatureVerificationResult {
    bool verified = false;
    std::string message;
};

struct CudaKernelSourceInspection {
    std::string name;
    std::string entrypoint;
    std::string source;
    std::string compiled_source;
    std::string compiled_source_hash;
    bool loaded_source_compilation_allowed = false;
    std::string signature_algorithm;
    std::string signing_public_key_fingerprint;
    std::string signature;
};

[[nodiscard]] nlohmann::json cudaKernelSourceInspectionToJson(const CudaKernelSourceInspection& info);
[[nodiscard]] nlohmann::json cudaKernelSourceInspectionListToJson(const std::vector<CudaKernelSourceInspection>& infos);

[[nodiscard]] nlohmann::json cudaKernelManifestFromExpressionJson(const nlohmann::json& expression_json);
[[nodiscard]] std::string cudaKernelManifestCanonicalBytes(const nlohmann::json& manifest);
[[nodiscard]] std::string cudaKernelGenerateAndAttachManifestSignature(nlohmann::json& expression_json);
[[nodiscard]] CudaKernelSignatureVerificationResult cudaKernelVerifyManifestSignature(
    const nlohmann::json& expression_json,
    const std::string& trusted_public_key);
[[nodiscard]] std::vector<std::string> collectCudaKernelSigningPublicKeys(const nlohmann::json& j);
[[nodiscard]] std::vector<CudaKernelSourceInspection> collectCudaKernelSourceInfo(const nlohmann::json& j);
[[nodiscard]] nlohmann::json collectCudaKernelSourceInfoJson(const nlohmann::json& j);
[[nodiscard]] std::string cudaKernelLoadedModelSafetyDisclaimer();

}  // namespace ThorImplementation
