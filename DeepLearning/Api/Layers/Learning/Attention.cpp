
#include "DeepLearning/Api/Layers/Learning/Attention.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

using DataType = ThorImplementation::TensorDescriptor::DataType;
using json = nlohmann::json;

namespace {

std::string dtypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return "fp16";
        case DataType::BF16:
            return "bf16";
        case DataType::FP32:
            return "fp32";
        case DataType::FP8_E4M3:
            return "fp8_e4m3";
        case DataType::FP8_E5M2:
            return "fp8_e5m2";
        default:
            return "dtype(" + std::to_string(static_cast<int>(dtype)) + ")";
    }
}

bool isStorageDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::BF16; }
bool isComputeDType(DataType dtype) { return dtype == DataType::FP32; }

uint64_t checkedMul(uint64_t a, uint64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::invalid_argument(std::string("Attention ") + what + " overflows uint64_t.");
    }
    return a * b;
}

void requireRank2FeatureInput(const Thor::Tensor& tensor) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("Attention feature input tensor is not initialized.");
    }
    const std::vector<uint64_t> dims = tensor.getDimensions();
    if (dims.size() != 2) {
        throw std::invalid_argument("Attention feature input must have rank 2 [sequence, features] at the API level.");
    }
    if (dims[0] == 0 || dims[1] == 0) {
        throw std::invalid_argument("Attention feature input dimensions must be non-zero.");
    }
}

std::shared_ptr<Thor::ParameterSpecification> makeParameter(const std::string& name,
                                                            const std::vector<uint64_t>& shape,
                                                            DataType dtype,
                                                            const std::shared_ptr<Thor::Initializer>& initializer,
                                                            const std::shared_ptr<Thor::Optimizer>& optimizer) {
    Thor::ParameterSpecification::Builder builder;
    builder.name(name).shape(shape).dtype(dtype).initializer(initializer->clone()).trainable(true);
    if (optimizer != nullptr) {
        std::shared_ptr<Thor::Optimizer> optimizer_copy = optimizer;
        builder.optimizer(optimizer_copy);
    }
    return std::make_shared<Thor::ParameterSpecification>(builder.build());
}

constexpr bool kUsePackedQkvProjection = Thor::Attention::USE_PACKED_QKV_PROJECTION;

bool usePackedQkvProjectionForLayer(bool useRope) {
    // PackedQkvProjection is not being supported anymore as it was shown to be slower.
    // It's being left here as an orphaned reference if there is some future opportunity to gain performance using a packed QKV.
    if constexpr (!kUsePackedQkvProjection) {
        return false;
    } else {
        // RoPE is still a generic expression op and must not consume non-dense Q/K views sliced out of packed QKV.
        // Keep RoPE layers on the legacy split projection path until a layout-aware RoPE materialization path lands.
        return !useRope;
    }
}


std::string attentionMaskKindToString(ThorImplementation::AttentionMaskKind value) {
    switch (value) {
        case ThorImplementation::AttentionMaskKind::None:
            return "none";
        case ThorImplementation::AttentionMaskKind::CausalTopLeft:
            return "causal_top_left";
        case ThorImplementation::AttentionMaskKind::CausalBottomRight:
            return "causal_bottom_right";
        case ThorImplementation::AttentionMaskKind::SlidingWindowTopLeft:
            return "sliding_window_top_left";
        case ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight:
            return "sliding_window_bottom_right";
    }
    throw std::runtime_error("Unknown AttentionMaskKind value.");
}

ThorImplementation::AttentionMaskKind attentionMaskKindFromString(const std::string& value) {
    if (value == "none")
        return ThorImplementation::AttentionMaskKind::None;
    if (value == "causal_top_left")
        return ThorImplementation::AttentionMaskKind::CausalTopLeft;
    if (value == "causal_bottom_right")
        return ThorImplementation::AttentionMaskKind::CausalBottomRight;
    if (value == "sliding_window_top_left")
        return ThorImplementation::AttentionMaskKind::SlidingWindowTopLeft;
    if (value == "sliding_window_bottom_right")
        return ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight;
    throw std::runtime_error("Unknown Attention mask kind: " + value);
}

std::string rotaryScalingKindToString(ThorImplementation::RotaryScalingKind value) {
    switch (value) {
        case ThorImplementation::RotaryScalingKind::None:
            return "none";
        case ThorImplementation::RotaryScalingKind::Linear:
            return "linear";
        case ThorImplementation::RotaryScalingKind::DynamicNTK:
            return "dynamic_ntk";
        case ThorImplementation::RotaryScalingKind::Yarn:
            return "yarn";
        case ThorImplementation::RotaryScalingKind::LongRope:
            return "longrope";
        case ThorImplementation::RotaryScalingKind::Llama3:
            return "llama3";
    }
    throw std::runtime_error("Unknown RotaryScalingKind value.");
}

ThorImplementation::RotaryScalingKind rotaryScalingKindFromString(const std::string& value) {
    if (value == "none")
        return ThorImplementation::RotaryScalingKind::None;
    if (value == "linear")
        return ThorImplementation::RotaryScalingKind::Linear;
    if (value == "dynamic_ntk")
        return ThorImplementation::RotaryScalingKind::DynamicNTK;
    if (value == "yarn")
        return ThorImplementation::RotaryScalingKind::Yarn;
    if (value == "longrope")
        return ThorImplementation::RotaryScalingKind::LongRope;
    if (value == "llama3")
        return ThorImplementation::RotaryScalingKind::Llama3;
    throw std::runtime_error("Unknown Attention RoPE scaling kind: " + value);
}

json optionalDataTypeToJson(std::optional<DataType> value) {
    if (!value.has_value())
        return nullptr;
    return value.value();
}

std::optional<DataType> optionalDataTypeFromJson(const json& j) {
    if (j.is_null())
        return std::nullopt;
    return j.get<DataType>();
}

json ropeOptionsToJson(const ThorImplementation::RotaryPositionEmbeddingOptions& opts) {
    json j;
    j["sequence_axis"] = opts.sequence_axis;
    j["head_dim_axis"] = opts.head_dim_axis;
    j["rotary_dim"] = opts.rotary_dim;
    j["base"] = opts.base;
    j["position_offset"] = opts.position_offset;
    j["interleaved"] = opts.interleaved;
    j["inverse"] = opts.inverse;
    j["scaling_kind"] = rotaryScalingKindToString(opts.scaling_kind);
    j["scaling_factor"] = opts.scaling_factor;
    j["original_max_position_embeddings"] = opts.original_max_position_embeddings;
    j["attention_factor"] = opts.attention_factor.has_value() ? json(opts.attention_factor.value()) : json(nullptr);
    j["yarn_beta_fast"] = opts.yarn_beta_fast;
    j["yarn_beta_slow"] = opts.yarn_beta_slow;
    j["llama3_low_freq_factor"] = opts.llama3_low_freq_factor;
    j["llama3_high_freq_factor"] = opts.llama3_high_freq_factor;
    j["long_rope_short_factors"] = opts.long_rope_short_factors;
    j["long_rope_long_factors"] = opts.long_rope_long_factors;
    j["output_dtype"] = optionalDataTypeToJson(opts.output_dtype);
    j["compute_dtype"] = optionalDataTypeToJson(opts.compute_dtype);
    j["allow_in_place_materialization"] = opts.allow_in_place_materialization;
    return j;
}

ThorImplementation::RotaryPositionEmbeddingOptions ropeOptionsFromJson(const json& j) {
    ThorImplementation::RotaryPositionEmbeddingOptions opts;
    opts.sequence_axis = j.value("sequence_axis", opts.sequence_axis);
    opts.head_dim_axis = j.value("head_dim_axis", opts.head_dim_axis);
    opts.rotary_dim = j.value("rotary_dim", opts.rotary_dim);
    opts.base = j.value("base", opts.base);
    opts.position_offset = j.value("position_offset", opts.position_offset);
    opts.interleaved = j.value("interleaved", opts.interleaved);
    opts.inverse = j.value("inverse", opts.inverse);
    opts.scaling_kind = rotaryScalingKindFromString(j.value("scaling_kind", std::string("none")));
    opts.scaling_factor = j.value("scaling_factor", opts.scaling_factor);
    opts.original_max_position_embeddings = j.value("original_max_position_embeddings", opts.original_max_position_embeddings);
    if (j.contains("attention_factor") && !j.at("attention_factor").is_null()) {
        opts.attention_factor = j.at("attention_factor").get<double>();
    }
    opts.yarn_beta_fast = j.value("yarn_beta_fast", opts.yarn_beta_fast);
    opts.yarn_beta_slow = j.value("yarn_beta_slow", opts.yarn_beta_slow);
    opts.llama3_low_freq_factor = j.value("llama3_low_freq_factor", opts.llama3_low_freq_factor);
    opts.llama3_high_freq_factor = j.value("llama3_high_freq_factor", opts.llama3_high_freq_factor);
    opts.long_rope_short_factors = j.value("long_rope_short_factors", std::vector<double>{});
    opts.long_rope_long_factors = j.value("long_rope_long_factors", std::vector<double>{});
    if (j.contains("output_dtype")) {
        opts.output_dtype = optionalDataTypeFromJson(j.at("output_dtype"));
    }
    if (j.contains("compute_dtype")) {
        opts.compute_dtype = optionalDataTypeFromJson(j.at("compute_dtype"));
    }
    opts.allow_in_place_materialization = j.value("allow_in_place_materialization", opts.allow_in_place_materialization);
    return opts;
}

ThorImplementation::DynamicExpression makeAttentionExpression(uint64_t sequenceLength,
                                                              uint64_t inputFeatures,
                                                              uint64_t outputFeatures,
                                                              uint32_t numHeads,
                                                              uint32_t numKeyValueHeads,
                                                              uint32_t headDim,
                                                              uint32_t valueDim,
                                                              bool hasBias,
                                                              bool useRope,
                                                              bool ropeInPlace,
                                                              ThorImplementation::RotaryPositionEmbeddingOptions ropeOptions,
                                                              ThorImplementation::AttentionMaskKind maskKind,
                                                              int64_t diagonalLeftBound,
                                                              int64_t diagonalRightBound,
                                                              bool useAlibiMask,
                                                              std::optional<double> attentionScale,
                                                              DataType inputDType,
                                                              DataType weightsDType,
                                                              DataType computeDType,
                                                              DataType outputDType) {
    using ThorImplementation::AttentionOptions;
    using ThorImplementation::AttentionTensorLayout;
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    const bool usePackedQkvProjection = usePackedQkvProjectionForLayer(useRope);
    std::vector<std::string> expectedInputs;
    if (usePackedQkvProjection) {
        expectedInputs = {"feature_input", "qkv_weights", "output_weights"};
        if (hasBias) {
            expectedInputs.push_back("qkv_bias");
            expectedInputs.push_back("output_bias");
        }
    } else {
        expectedInputs = {"feature_input", "query_weights", "key_weights", "value_weights", "output_weights"};
        if (hasBias) {
            expectedInputs.push_back("query_bias");
            expectedInputs.push_back("key_bias");
            expectedInputs.push_back("value_bias");
            expectedInputs.push_back("output_bias");
        }
    }

    return DynamicExpression(
        expectedInputs,
        {"feature_output"},
        [sequenceLength,
         inputFeatures,
         outputFeatures,
         usePackedQkvProjection,
         numHeads,
         numKeyValueHeads,
         headDim,
         valueDim,
         hasBias,
         useRope,
         ropeInPlace,
         ropeOptions,
         maskKind,
         diagonalLeftBound,
         diagonalRightBound,
         useAlibiMask,
         attentionScale,
         inputDType,
         weightsDType,
         computeDType,
         outputDType](const DynamicExpression::TensorMap& inputs,
                      const DynamicExpression::TensorMap& outputs,
                      Stream& stream) -> DynamicExpressionBuild {
            (void)stream;

            Tensor featureInput = inputs.at("feature_input");
            const auto inputDims = featureInput.getDimensions();
            if (inputDims.size() != 3) {
                throw std::runtime_error("Attention runtime feature input must be [batch, sequence, features].");
            }
            const uint64_t batch = inputDims[0];
            if (batch == 0 || inputDims[1] != sequenceLength || inputDims[2] != inputFeatures) {
                throw std::runtime_error("Attention runtime feature input shape does not match the API shape.");
            }

            const uint64_t queryWidth = checkedMul(numHeads, headDim, "query projection width");
            const uint64_t keyWidth = checkedMul(numKeyValueHeads, headDim, "key projection width");
            const uint64_t valueWidth = checkedMul(numKeyValueHeads, valueDim, "value projection width");
            const uint64_t qkvWidth = queryWidth + keyWidth + valueWidth;

            auto validateWeight = [&](const char* name, uint64_t in, uint64_t out) {
                const Tensor& w = inputs.at(name);
                if (w.getDimensions() != std::vector<uint64_t>{in, out}) {
                    throw std::runtime_error(std::string("Attention ") + name + " shape mismatch.");
                }
                if (w.getDataType() != weightsDType) {
                    throw std::runtime_error(std::string("Attention ") + name + " dtype mismatch.");
                }
            };
            if (usePackedQkvProjection) {
                validateWeight("qkv_weights", inputFeatures, qkvWidth);
            } else {
                validateWeight("query_weights", inputFeatures, queryWidth);
                validateWeight("key_weights", inputFeatures, keyWidth);
                validateWeight("value_weights", inputFeatures, valueWidth);
            }
            validateWeight("output_weights", checkedMul(numHeads, valueDim, "output projection input width"), outputFeatures);

            if (hasBias) {
                auto validateBias = [&](const char* name, uint64_t width) {
                    const Tensor& b = inputs.at(name);
                    if (b.getDimensions() != std::vector<uint64_t>{width}) {
                        throw std::runtime_error(std::string("Attention ") + name + " shape mismatch.");
                    }
                    if (b.getDataType() != weightsDType) {
                        throw std::runtime_error(std::string("Attention ") + name + " dtype mismatch.");
                    }
                };
                if (usePackedQkvProjection) {
                    validateBias("qkv_bias", qkvWidth);
                } else {
                    validateBias("query_bias", queryWidth);
                    validateBias("key_bias", keyWidth);
                    validateBias("value_bias", valueWidth);
                }
                validateBias("output_bias", outputFeatures);
            }

            featureInput.reshape({batch * sequenceLength, inputFeatures});

            Expression x = Expression::input("feature_input", inputDType, inputDType);
            Expression ow = Expression::input("output_weights", weightsDType, weightsDType);

            struct ProjectedQkv {
                Expression q;
                Expression k;
                Expression v;
            };

            auto buildSplitProjection = [&]() -> ProjectedQkv {
                Expression qw = Expression::input("query_weights", weightsDType, weightsDType);
                Expression kw = Expression::input("key_weights", weightsDType, weightsDType);
                Expression vw = Expression::input("value_weights", weightsDType, weightsDType);

                Expression q = Expression::matmul(x, qw, false, false, computeDType, outputDType);
                Expression k = Expression::matmul(x, kw, false, false, computeDType, outputDType);
                Expression v = Expression::matmul(x, vw, false, false, computeDType, outputDType);
                if (hasBias) {
                    q = q + Expression::input("query_bias", weightsDType, weightsDType);
                    k = k + Expression::input("key_bias", weightsDType, weightsDType);
                    v = v + Expression::input("value_bias", weightsDType, weightsDType);
                }

                // Keep Thor's high-level attention tensors dense and logical [B,S,H,D].  AttentionTensorLayout::BSHD
                // tells the cuDNN adapter how to reinterpret that token-major storage for cuDNN's [B,H,S,D]
                // descriptor contract; generic Thor expression ops never see a fake [B,H,S,D] reshape.
                q = q.reshape({batch, sequenceLength, numHeads, headDim}).withOutputDType(outputDType);
                k = k.reshape({batch, sequenceLength, numKeyValueHeads, headDim}).withOutputDType(outputDType);
                v = v.reshape({batch, sequenceLength, numKeyValueHeads, valueDim}).withOutputDType(outputDType);
                return ProjectedQkv{std::move(q), std::move(k), std::move(v)};
            };

            auto buildPackedProjection = [&]() -> ProjectedQkv {
                Expression qkvWeights = Expression::input("qkv_weights", weightsDType, weightsDType);
                Expression qkv = Expression::matmul(x, qkvWeights, false, false, computeDType, outputDType);
                if (hasBias) {
                    qkv = qkv + Expression::input("qkv_bias", weightsDType, weightsDType);
                }

                // Packed QKV produces one token-major [B*S, Q+K+V] buffer.  Q/K/V are zero-copy strided views into
                // that buffer, with row stride equal to the full packed width.  This is the final no-split form needed
                // for packed-QKV forward and for packed-QKV training once view backward accumulates dQ/dK/dV to dQKV.
                const uint64_t batchStride = sequenceLength * qkvWidth;
                Expression q = qkv.stridedView({batch, sequenceLength, numHeads, headDim}, {batchStride, qkvWidth, headDim, 1}, 0)
                                   .withOutputDType(outputDType);
                Expression k =
                    qkv.stridedView({batch, sequenceLength, numKeyValueHeads, headDim}, {batchStride, qkvWidth, headDim, 1}, queryWidth)
                        .withOutputDType(outputDType);
                Expression v =
                    qkv.stridedView(
                           {batch, sequenceLength, numKeyValueHeads, valueDim}, {batchStride, qkvWidth, valueDim, 1}, queryWidth + keyWidth)
                        .withOutputDType(outputDType);
                return ProjectedQkv{std::move(q), std::move(k), std::move(v)};
            };

            ProjectedQkv projected = [&]() -> ProjectedQkv {
                if constexpr (kUsePackedQkvProjection) {
                    if (usePackedQkvProjection) {
                        return buildPackedProjection();
                    }
                }
                return buildSplitProjection();
            }();

            Expression q = std::move(projected.q);
            Expression k = std::move(projected.k);
            Expression v = std::move(projected.v);

            if (useRope) {
                ThorImplementation::RotaryPositionEmbeddingOptions opts = ropeOptions;
                opts.sequence_axis = 1;
                opts.head_dim_axis = 3;
                opts.allow_in_place_materialization = ropeInPlace;
                if (!opts.compute_dtype.has_value()) {
                    opts.compute_dtype = computeDType;
                }
                if (!opts.output_dtype.has_value()) {
                    opts.output_dtype = outputDType;
                }

                q = q.rotaryPositionEmbedding(opts);
                k = k.rotaryPositionEmbedding(opts);
            }

            AttentionOptions options;
            options.q_layout = AttentionTensorLayout::BSHD;
            options.k_layout = AttentionTensorLayout::BSHD;
            options.v_layout = AttentionTensorLayout::BSHD;
            options.o_layout = AttentionTensorLayout::BSHD;
            options.mask_kind = maskKind;
            options.diagonal_left_bound = diagonalLeftBound;
            options.diagonal_right_bound = diagonalRightBound;
            options.use_alibi_mask = useAlibiMask;
            options.compute_dtype = computeDType;
            options.output_dtype = outputDType;
            if (attentionScale.has_value()) {
                options.attention_scale = static_cast<float>(attentionScale.value());
            }

            Expression attn = Expression::scaledDotProductAttention(q, k, v, options).withOutputDType(outputDType);
            Expression merged = attn.reshape({batch * sequenceLength, checkedMul(numHeads, valueDim, "merged head width")});
            Expression out = Expression::matmul(merged, ow, false, false, computeDType, outputDType);
            if (hasBias) {
                out = out + Expression::input("output_bias", weightsDType, weightsDType);
            }
            out = out.reshape({batch, sequenceLength, outputFeatures}).withOutputDType(outputDType);

            auto expressionOutputs = Expression::outputs({{"feature_output", out}});

            DynamicExpression::TensorMap stampInputs = inputs;
            stampInputs["feature_input"] = featureInput;

            return DynamicExpressionBuild{
                std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), stream.getGpuNum())),
                stampInputs,
                {},
                outputs,
                {},
            };
        });
}

}  // namespace

namespace Thor {

void Attention::Builder::verifyConfig() const {
    if (!_network.has_value()) {
        throw std::invalid_argument("Attention::Builder requires network().");
    }
    if (!_featureInput.has_value()) {
        throw std::invalid_argument("Attention::Builder requires featureInput().");
    }
    if (!_numHeads.has_value()) {
        throw std::invalid_argument("Attention::Builder requires numHeads().");
    }
    requireRank2FeatureInput(_featureInput.value());
    if (_numHeads.value() == 0) {
        throw std::invalid_argument("Attention numHeads must be non-zero.");
    }
    if (_numKeyValueHeads.value_or(_numHeads.value()) == 0) {
        throw std::invalid_argument("Attention numKeyValueHeads must be non-zero.");
    }
    if (_numHeads.value() % _numKeyValueHeads.value_or(_numHeads.value()) != 0) {
        throw std::invalid_argument("Attention numHeads must be a multiple of numKeyValueHeads for MHA/GQA/MQA.");
    }
    if (_headDim.value_or(1) == 0 || _valueDim.value_or(1) == 0) {
        throw std::invalid_argument("Attention headDim/valueDim must be non-zero.");
    }

    const DataType inputDType = _featureInput->getDataType();
    const DataType weightsDType = _weightsDataType.value_or(inputDType);
    const DataType outputDType = _outputDataType.value_or(inputDType);
    const DataType computeDType = _computeDataType.value_or(DataType::FP32);
    if (!isStorageDType(inputDType)) {
        throw std::invalid_argument("Attention feature input dtype must be fp16 or bf16 for the current training path. Got " +
                                    dtypeName(inputDType) + ".");
    }
    if (!isStorageDType(weightsDType)) {
        throw std::invalid_argument("Attention weightsDataType must be fp16 or bf16 for the current training path. Got " +
                                    dtypeName(weightsDType) + ".");
    }
    if (!isStorageDType(outputDType)) {
        throw std::invalid_argument("Attention outputDataType must be fp16 or bf16 for the current training path. Got " +
                                    dtypeName(outputDType) + ".");
    }
    if (!isComputeDType(computeDType)) {
        throw std::invalid_argument("Attention computeDataType must currently be fp32 for cuDNN SDPA training. Got " +
                                    dtypeName(computeDType) + ".");
    }

    const auto maskKind = _maskKind.value_or(ThorImplementation::AttentionMaskKind::None);
    const bool useAlibi = _useAlibiMask.value_or(false);
    const int64_t rightBound = _diagonalRightBound.value_or(0);
    if (useAlibi && maskKind != ThorImplementation::AttentionMaskKind::CausalTopLeft &&
        maskKind != ThorImplementation::AttentionMaskKind::CausalBottomRight &&
        maskKind != ThorImplementation::AttentionMaskKind::SlidingWindowTopLeft &&
        maskKind != ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight) {
        throw std::invalid_argument("Attention ALiBi requires a causal/sliding-window diagonal mask.");
    }
    if (useAlibi && rightBound != 0) {
        throw std::invalid_argument("Attention ALiBi requires diagonalRightBound == 0.");
    }
    if (useAlibi && (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
                     maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight)) {
        throw std::invalid_argument("Attention ALiBi cannot currently be combined with bottom-right/decode masks in cuDNN SDPA.");
    }
    if (_ropeInPlace.value_or(false) && !_useRope.value_or(false)) {
        throw std::invalid_argument("Attention ropeInPlace requires useRope to be enabled.");
    }
    if (_attentionScale.has_value() && (!std::isfinite(_attentionScale.value()) || _attentionScale.value() <= 0.0)) {
        throw std::invalid_argument("Attention attentionScale must be finite and positive.");
    }
}

Attention Attention::Builder::build() {
    if (!_numKeyValueHeads.has_value() && _numHeads.has_value()) {
        _numKeyValueHeads = _numHeads.value();
    }
    if (!_headDim.has_value()) {
        if (!_featureInput.has_value() || !_numHeads.has_value()) {
            throw std::invalid_argument("Attention headDim default requires featureInput and numHeads.");
        }
        const uint64_t inputFeatures = _featureInput->getDimensions().at(1);
        if (inputFeatures % _numHeads.value() != 0) {
            throw std::invalid_argument("Attention default headDim requires input features divisible by numHeads.");
        }
        _headDim = static_cast<uint32_t>(inputFeatures / _numHeads.value());
    }
    if (!_valueDim.has_value()) {
        _valueDim = _headDim.value();
    }
    if (!_outputFeatures.has_value() && _featureInput.has_value()) {
        _outputFeatures = static_cast<uint32_t>(_featureInput->getDimensions().at(1));
    }
    if (!_hasBias.has_value()) {
        _hasBias = false;
    }
    if (!_maskKind.has_value()) {
        _maskKind = ThorImplementation::AttentionMaskKind::None;
    }
    if (!_diagonalLeftBound.has_value()) {
        _diagonalLeftBound = 0;
    }
    if (!_diagonalRightBound.has_value()) {
        _diagonalRightBound = 0;
    }
    if (!_useAlibiMask.has_value()) {
        _useAlibiMask = false;
    }
    if (!_useRope.has_value()) {
        _useRope = false;
    }
    if (!_ropeInPlace.has_value()) {
        _ropeInPlace = false;
    }
    if (!_ropeOptions.has_value()) {
        _ropeOptions = ThorImplementation::RotaryPositionEmbeddingOptions{};
    }
    if (!_weightsDataType.has_value() && _featureInput.has_value()) {
        _weightsDataType = _featureInput->getDataType();
    }
    if (!_computeDataType.has_value()) {
        _computeDataType = DataType::FP32;
    }
    if (!_outputDataType.has_value() && _featureInput.has_value()) {
        _outputDataType = _featureInput->getDataType();
    }
    if (_weightsInitializer == nullptr) {
        _weightsInitializer = Glorot::Builder().build();
    }
    if (_biasInitializer == nullptr) {
        _biasInitializer = Glorot::Builder().build();
    }

    verifyConfig();

    const auto inputDims = _featureInput->getDimensions();
    const uint64_t sequenceLength = inputDims.at(0);
    const uint64_t inputFeatures = inputDims.at(1);
    const uint64_t qWidth = checkedMul(_numHeads.value(), _headDim.value(), "query projection width");
    const uint64_t kvKeyWidth = checkedMul(_numKeyValueHeads.value(), _headDim.value(), "key projection width");
    const uint64_t kvValueWidth = checkedMul(_numKeyValueHeads.value(), _valueDim.value(), "value projection width");
    const uint64_t mergedWidth = checkedMul(_numHeads.value(), _valueDim.value(), "merged head width");

    const uint64_t qkvWidth = qWidth + kvKeyWidth + kvValueWidth;
    const bool usePackedQkvProjection = usePackedQkvProjectionForLayer(_useRope.value());

    std::vector<std::shared_ptr<ParameterSpecification>> parameters;
    if (usePackedQkvProjection) {
        parameters.push_back(
            makeParameter("qkv_weights", {inputFeatures, qkvWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
    } else {
        parameters.push_back(
            makeParameter("query_weights", {inputFeatures, qWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
        parameters.push_back(
            makeParameter("key_weights", {inputFeatures, kvKeyWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
        parameters.push_back(
            makeParameter("value_weights", {inputFeatures, kvValueWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
    }
    parameters.push_back(
        makeParameter("output_weights", {mergedWidth, _outputFeatures.value()}, _weightsDataType.value(), _weightsInitializer, _optimizer));
    if (_hasBias.value()) {
        if (usePackedQkvProjection) {
            parameters.push_back(makeParameter("qkv_bias", {qkvWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
        } else {
            parameters.push_back(makeParameter("query_bias", {qWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
            parameters.push_back(makeParameter("key_bias", {kvKeyWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
            parameters.push_back(makeParameter("value_bias", {kvValueWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
        }
        parameters.push_back(
            makeParameter("output_bias", {_outputFeatures.value()}, _weightsDataType.value(), _biasInitializer, _optimizer));
    }

    Tensor output(_outputDataType.value(), {sequenceLength, _outputFeatures.value()});
    Attention layer(makeAttentionExpression(sequenceLength,
                                            inputFeatures,
                                            _outputFeatures.value(),
                                            _numHeads.value(),
                                            _numKeyValueHeads.value(),
                                            _headDim.value(),
                                            _valueDim.value(),
                                            _hasBias.value(),
                                            _useRope.value(),
                                            _ropeInPlace.value(),
                                            _ropeOptions.value(),
                                            _maskKind.value(),
                                            _diagonalLeftBound.value(),
                                            _diagonalRightBound.value(),
                                            _useAlibiMask.value(),
                                            _attentionScale,
                                            _featureInput->getDataType(),
                                            _weightsDataType.value(),
                                            _computeDataType.value(),
                                            _outputDataType.value()),
                    {{{"feature_input", _featureInput.value()}}},
                    {{{"feature_output", output}}},
                    std::move(parameters),
                    _numHeads.value(),
                    _numKeyValueHeads.value(),
                    _headDim.value(),
                    _valueDim.value(),
                    _outputFeatures.value(),
                    _hasBias.value(),
                    _useRope.value(),
                    _ropeInPlace.value(),
                    _ropeOptions.value(),
                    _maskKind.value(),
                    _diagonalLeftBound.value(),
                    _diagonalRightBound.value(),
                    _useAlibiMask.value(),
                    _attentionScale,
                    _weightsDataType.value(),
                    _computeDataType.value(),
                    _outputDataType.value());

    layer.addToNetwork(_network.value());
    return layer;
}


json Attention::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = "1.0.0";
    j["layer_type"] = "attention";
    j["layer_name"] = std::string("layer") + std::to_string(getId());

    j["num_heads"] = numHeads;
    j["num_key_value_heads"] = numKeyValueHeads;
    j["head_dim"] = headDim;
    j["value_dim"] = valueDim;
    j["output_features"] = outputFeatures;
    j["has_bias"] = hasBias;
    j["use_rope"] = useRope;
    j["rope_in_place"] = ropeInPlace;
    j["rope_options"] = ropeOptionsToJson(ropeOptions);
    j["mask_kind"] = attentionMaskKindToString(maskKind);
    j["diagonal_left_bound"] = diagonalLeftBound;
    j["diagonal_right_bound"] = diagonalRightBound;
    j["use_alibi_mask"] = useAlibiMask;
    j["attention_scale"] = attentionScale.has_value() ? json(attentionScale.value()) : json(nullptr);
    j["weights_data_type"] = weightsDataType;
    j["compute_data_type"] = computeDataType;
    j["output_data_type"] = outputDataType;

    const std::optional<Tensor> input = getFeatureInput();
    const std::optional<Tensor> output = getFeatureOutput();
    if (!input.has_value() || !output.has_value()) {
        throw std::runtime_error("Attention serialization requires one feature input and one feature output.");
    }
    j["feature_input"] = input.value().architectureJson();
    j["feature_output"] = output.value().architectureJson();
    j["parameters"] = getParametersArchitectureJson()["parameters"];
    return j;
}

json Attention::serialize(thor_file::TarWriter& archiveWriter,
                          Stream stream,
                          bool saveOptimizerState,
                          ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, "layer" + std::to_string(getId()));
    return j;
}

void Attention::deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in Attention::deserialize: " + j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "attention") {
        throw std::runtime_error("Layer type mismatch in Attention::deserialize: " + j.at("layer_type").get<std::string>());
    }

    const uint64_t inputOriginalId = j.at("feature_input").at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(inputOriginalId);
    Tensor featureOutput = Tensor::deserialize(j.at("feature_output"), archiveReader.get());

    const std::vector<uint64_t> inputDims = featureInput.getDimensions();
    if (inputDims.size() != 2) {
        throw std::runtime_error("Attention deserialize expected rank-2 feature_input.");
    }
    const uint64_t sequenceLength = inputDims.at(0);
    const uint64_t inputFeatures = inputDims.at(1);

    const uint32_t numHeads = j.at("num_heads").get<uint32_t>();
    const uint32_t numKeyValueHeads = j.at("num_key_value_heads").get<uint32_t>();
    const uint32_t headDim = j.at("head_dim").get<uint32_t>();
    const uint32_t valueDim = j.at("value_dim").get<uint32_t>();
    const uint32_t outputFeatures = j.at("output_features").get<uint32_t>();
    const bool hasBias = j.at("has_bias").get<bool>();
    const bool useRope = j.at("use_rope").get<bool>();
    const bool ropeInPlace = j.value("rope_in_place", false);
    ThorImplementation::RotaryPositionEmbeddingOptions ropeOptions = ropeOptionsFromJson(j.at("rope_options"));
    const ThorImplementation::AttentionMaskKind maskKind = attentionMaskKindFromString(j.value("mask_kind", std::string("none")));
    const int64_t diagonalLeftBound = j.value("diagonal_left_bound", int64_t{0});
    const int64_t diagonalRightBound = j.value("diagonal_right_bound", int64_t{0});
    const bool useAlibiMask = j.value("use_alibi_mask", false);
    std::optional<double> attentionScale = std::nullopt;
    if (j.contains("attention_scale") && !j.at("attention_scale").is_null()) {
        attentionScale = j.at("attention_scale").get<double>();
    }
    const DataType weightsDataType = j.at("weights_data_type").get<DataType>();
    const DataType computeDataType = j.at("compute_data_type").get<DataType>();
    const DataType outputDataType = j.at("output_data_type").get<DataType>();

    std::vector<std::shared_ptr<ParameterSpecification>> parameters;
    if (j.contains("parameters")) {
        const json& parametersJson = j.at("parameters");
        if (!parametersJson.is_object()) {
            throw std::runtime_error("Attention parameters must be an object keyed by parameter name.");
        }
        for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
            parameters.push_back(std::make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }

    std::vector<std::string> requiredParameterNames;
    if (usePackedQkvProjectionForLayer(useRope)) {
        requiredParameterNames = {"qkv_weights", "output_weights"};
        if (hasBias) {
            requiredParameterNames.push_back("qkv_bias");
            requiredParameterNames.push_back("output_bias");
        }
    } else {
        requiredParameterNames = {"query_weights", "key_weights", "value_weights", "output_weights"};
        if (hasBias) {
            requiredParameterNames.push_back("query_bias");
            requiredParameterNames.push_back("key_bias");
            requiredParameterNames.push_back("value_bias");
            requiredParameterNames.push_back("output_bias");
        }
    }
    for (const std::string& requiredName : requiredParameterNames) {
        bool found = false;
        for (const auto& parameter : parameters) {
            if (parameter != nullptr && parameter->getName() == requiredName) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Attention deserialize did not find required parameter '" + requiredName + "'.");
        }
    }

    Attention layer(makeAttentionExpression(sequenceLength,
                                            inputFeatures,
                                            outputFeatures,
                                            numHeads,
                                            numKeyValueHeads,
                                            headDim,
                                            valueDim,
                                            hasBias,
                                            useRope,
                                            ropeInPlace,
                                            ropeOptions,
                                            maskKind,
                                            diagonalLeftBound,
                                            diagonalRightBound,
                                            useAlibiMask,
                                            attentionScale,
                                            featureInput.getDataType(),
                                            weightsDataType,
                                            computeDataType,
                                            outputDataType),
                    {{{"feature_input", featureInput}}},
                    {{{"feature_output", featureOutput}}},
                    std::move(parameters),
                    numHeads,
                    numKeyValueHeads,
                    headDim,
                    valueDim,
                    outputFeatures,
                    hasBias,
                    useRope,
                    ropeInPlace,
                    std::move(ropeOptions),
                    maskKind,
                    diagonalLeftBound,
                    diagonalRightBound,
                    useAlibiMask,
                    attentionScale,
                    weightsDataType,
                    computeDataType,
                    outputDataType);
    layer.addToNetwork(network);
}

}  // namespace Thor


namespace {
static const bool registeredAttention = [] {
    Thor::TrainableLayer::register_layer("attention", &Thor::Attention::deserialize);
    return true;
}();
}  // namespace
