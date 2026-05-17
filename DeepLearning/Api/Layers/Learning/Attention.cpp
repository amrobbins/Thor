
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
    // It is being left here as an orphaned reference if there is some future opportunity to gain performance using a packed QKV.
    if constexpr (!kUsePackedQkvProjection) {
        return false;
    } else {
        // RoPE is still a generic expression op and must not consume non-dense Q/K views sliced out of packed QKV.
        // Keep RoPE layers on the legacy split projection path until a layout-aware RoPE materialization path lands.
        return !useRope;
    }
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

}  // namespace Thor
