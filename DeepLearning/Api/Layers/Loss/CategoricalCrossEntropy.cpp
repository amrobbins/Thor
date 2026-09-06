#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kLossName = "loss";
constexpr const char* kGradientName = "predictions_grad";

void validateDenseTargetRaggedDType(const char* what, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported ragged CategoricalCrossEntropy ") + what + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::DynamicExpression makeRaggedCategoricalCrossEntropyLossExpression(DataType lossDataType) {
    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss = (-(labels * logits.logSoftmax())).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeRaggedCategoricalCrossEntropyGradientExpression(DataType predictionsDataType) {
    validateDenseTargetRaggedDType("predictions", predictionsDataType);
    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression gradient =
        ((logits.softmax() - labels) * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void CategoricalCrossEntropy::buildSupportLayersAndAddToNetwork() {
    if (isRagged()) {
        if (labelType == LabelType::SPARSE) {
            THOR_THROW_IF_FALSE(!logitsNativeLossAddedToNetwork);
            SparseCategoricalCrossEntropy::Builder sparseBuilder;
            sparseBuilder.network(*network)
                .predictions(raggedPredictionsTensor.value())
                .labels(raggedLabelsTensor.value())
                .numClasses(numClasses)
                .logitsNativeLossAddedToNetwork()
                .reportsRawLoss()
                .lossDataType(lossDataType)
                .lossWeight(lossWeight.value_or(1.0f));
            if (ignoreIndex.has_value())
                sparseBuilder.ignoreIndex(ignoreIndex.value());
            if (raggedMaskTensor.has_value())
                sparseBuilder.mask(raggedMaskTensor.value());

            SparseCategoricalCrossEntropy rawSparseCrossEntropy;
            sparseBuilder.populateAndAdd(rawSparseCrossEntropy, LabelType::SPARSE, std::optional<uint32_t>(numClasses));
            raggedRawLossTensor = rawSparseCrossEntropy.getRaggedRawLoss();
            lossShaperInput = raggedRawLossTensor->getValues();

            if (lossShape == LossShape::NONE) {
                lossTensor = lossShaperInput;
                Stub::Builder().network(*network).inputTensor(lossShaperInput).build();
            } else if (lossShape == LossShape::RAW) {
                lossTensor = lossShaperInput;
            } else if (lossShape == LossShape::PER_EXAMPLE) {
                lossTensor = RaggedLossShaper::Builder()
                                 .network(*network)
                                 .lossInput(raggedRawLossTensor.value())
                                 .reportsPerExampleLoss()
                                 .build()
                                 .getLossOutput();
            } else if (lossShape == LossShape::BATCH) {
                lossTensor = RaggedLossShaper::Builder()
                                 .network(*network)
                                 .lossInput(raggedRawLossTensor.value())
                                 .reportsBatchLoss()
                                 .build()
                                 .getLossOutput();
            } else {
                THOR_UNREACHABLE();
            }
            return;
        }

        THOR_THROW_IF_FALSE(labelType == LabelType::DENSE);
        validateDenseTargetRaggedDType("predictions", predictionsTensor.getDataType());
        validateDenseTargetRaggedDType("labels", labelsTensor.getDataType());

        RaggedCustomLoss rawCrossEntropy =
            RaggedCustomLoss::Builder()
                .network(*network)
                .lossExpression(makeRaggedCategoricalCrossEntropyLossExpression(lossDataType))
                .gradientExpression(makeRaggedCategoricalCrossEntropyGradientExpression(predictionsTensor.getDataType()))
                .predictions(raggedPredictionsTensor.value())
                .labels(raggedLabelsTensor.value())
                .predictionsName(kPredictionsName)
                .labelsName(kLabelsName)
                .lossName(kLossName)
                .gradientName(kGradientName)
                .lossDataType(lossDataType)
                .lossWeight(lossWeight.value_or(1.0f))
                .build();
        raggedRawLossTensor = rawCrossEntropy.getRaggedRawLoss();
        lossShaperInput = raggedRawLossTensor->getValues();
        if (lossShape == LossShape::NONE) {
            lossTensor = lossShaperInput;
            Stub::Builder().network(*network).inputTensor(lossShaperInput).build();
        } else if (lossShape == LossShape::RAW) {
            lossTensor = lossShaperInput;
        } else if (lossShape == LossShape::PER_EXAMPLE) {
            lossTensor = RaggedLossShaper::Builder()
                             .network(*network)
                             .lossInput(raggedRawLossTensor.value())
                             .reportsPerExampleLoss()
                             .build()
                             .getLossOutput();
        } else if (lossShape == LossShape::BATCH) {
            lossTensor = RaggedLossShaper::Builder()
                             .network(*network)
                             .lossInput(raggedRawLossTensor.value())
                             .reportsBatchLoss()
                             .build()
                             .getLossOutput();
        } else {
            THOR_UNREACHABLE();
        }
        return;
    }

    if (labelType == LabelType::SPARSE) {
        THOR_THROW_IF_FALSE(!logitsNativeLossAddedToNetwork);
        SparseCategoricalCrossEntropy::Builder sparseBuilder;
        sparseBuilder.network(*network)
            .predictions(predictionsTensor)
            .labels(labelsTensor)
            .numClasses(numClasses)
            .logitsNativeLossAddedToNetwork()
            .reportsRawLoss()
            .lossDataType(lossDataType);
        sparseBuilder.lossWeight(lossWeight.value_or(1.0f));
        if (ignoreIndex.has_value())
            sparseBuilder.ignoreIndex(ignoreIndex.value());
        if (maskTensor.has_value())
            sparseBuilder.mask(maskTensor.value());

        SparseCategoricalCrossEntropy rawSparseCrossEntropy;
        sparseBuilder.populateAndAdd(rawSparseCrossEntropy, LabelType::SPARSE, std::optional<uint32_t>(numClasses));
        lossShaperInput = rawSparseCrossEntropy.getLoss();

        finalizeLossReporting();
        return;
    }

    THOR_THROW_IF_FALSE(!softmaxAddedToNetwork);
    shared_ptr<Activation> softmax = Softmax::Builder().backwardComputedExternally().build();
    softmaxOutput = softmax->addToNetwork(predictionsTensor, network);

    CategoricalCrossEntropy::Builder categoricalCrossEntropyBuilder = CategoricalCrossEntropy::Builder()
                                                                          .network(*network)
                                                                          .predictions(softmaxOutput)
                                                                          .labels(labelsTensor)
                                                                          .softmaxAddedToNetwork()
                                                                          .reportsRawLoss()
                                                                          .lossDataType(lossDataType)
                                                                          .lossWeight(lossWeight.value_or(1.0f));

    CategoricalCrossEntropy crossEntropy;
    categoricalCrossEntropyBuilder.populateAndAdd(
        crossEntropy, labelType, labelType == LabelType::SPARSE ? std::optional<uint32_t>(numClasses) : std::nullopt);
    lossShaperInput = crossEntropy.getLoss();

    finalizeLossReporting();
}

json CategoricalCrossEntropy::architectureJson() const {
    if (isRagged()) {
        json j = Loss::architectureJson();
        j["layer_type"] = labelType == LabelType::SPARSE ? "sparse_categorical_cross_entropy" : "categorical_cross_entropy";
        j["loss_shape"] = lossShape;
        j["ragged_predictions"] = raggedPredictionsTensor->architectureJson();
        j["ragged_labels"] = raggedLabelsTensor->architectureJson();
        if (raggedMaskTensor.has_value()) j["ragged_mask"] = raggedMaskTensor->architectureJson();
        if (raggedRawLossTensor.has_value()) j["ragged_raw_loss"] = raggedRawLossTensor->architectureJson();
        j["num_classes"] = numClasses;
        j["logits_native"] = logitsNativeLossAddedToNetwork;
        if (ignoreIndex.has_value()) j["ignore_index"] = ignoreIndex.value();
        return j;
    }

    json j;
    j["factory"] = Layer::Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = labelType == LabelType::SPARSE ? "sparse_categorical_cross_entropy" : "categorical_cross_entropy";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["loss_shape"] = LossShape::RAW;
    j["loss_data_type"] = lossDataType;
    j["labels_tensor"] = labelsTensor.architectureJson();
    j["predictions_tensor"] = predictionsTensor.architectureJson();
    j["num_classes"] = numClasses;
    j["logits_native"] = logitsNativeLossAddedToNetwork;
    if (ignoreIndex.has_value())
        j["ignore_index"] = ignoreIndex.value();
    if (maskTensor.has_value())
        j["mask_tensor"] = maskTensor.value().architectureJson();
    j["softmax_output_tensor"] = softmaxOutput.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();

    ThorImplementation::addLossWeightToJson(j, lossWeight);
    return j;
}

void CategoricalCrossEntropy::deserializeInto(const json &j,
                                             Network *network,
                                             CategoricalCrossEntropy &categoricalCrossEntropy,
                                             LabelType labelType,
                                             const string &expectedLayerType) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in CategoricalCrossEntropy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != expectedLayerType)
        throw runtime_error("Layer type mismatch in CategoricalCrossEntropy::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "CategoricalCrossEntropy");
        RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "CategoricalCrossEntropy");
        const LossShape serializedShape = j.at("loss_shape").get<LossShape>();
        if (serializedShape == LossShape::PER_OUTPUT)
            throw runtime_error("Serialized ragged CategoricalCrossEntropy cannot use LossShape::PER_OUTPUT.");

        if (labelType == LabelType::SPARSE) {
            const vector<uint64_t> predictionTrailing = predictions.getTrailingDimensions();
            if (predictionTrailing.size() != 1 || predictionTrailing.back() <= 1)
                throw runtime_error(
                    "Serialized ragged SparseCategoricalCrossEntropy predictions must have trailing shape [C] with C > 1.");
            if (!CategoricalCrossEntropy::Builder::sparseRaggedValueIsScalar(labels.getTrailingDimensions()))
                throw runtime_error(
                    "Serialized ragged SparseCategoricalCrossEntropy labels must have scalar trailing shape [] or [1].");
            if (!predictions.sharesPartitionWith(labels) || predictions.getBatchSize() != labels.getBatchSize() ||
                predictions.getMaxTotalValues() != labels.getMaxTotalValues())
                throw runtime_error(
                    "Serialized ragged SparseCategoricalCrossEntropy predictions and labels must share one row partition and capacity.");
            const DataType labelsDType = labels.getValuesDataType();
            if (labelsDType != DataType::UINT8 && labelsDType != DataType::UINT16 && labelsDType != DataType::UINT32)
                throw runtime_error(
                    "Serialized ragged SparseCategoricalCrossEntropy labels must use UINT8, UINT16, or UINT32.");
            const uint32_t serializedNumClasses = j.at("num_classes").get<uint32_t>();
            if (serializedNumClasses <= 1 || predictionTrailing.back() != serializedNumClasses)
                throw runtime_error(
                    "Serialized ragged SparseCategoricalCrossEntropy num_classes must match the prediction class dimension.");

            const bool logitsNative = j.value("logits_native", false);
            if (logitsNative) {
                if (serializedShape != LossShape::RAW)
                    throw runtime_error("Serialized ragged logits-native SparseCategoricalCrossEntropy must use LossShape::RAW.");
                const DataType serializedLossDType = j.at("loss_data_type").get<DataType>();
                if (serializedLossDType != DataType::FP16 && serializedLossDType != DataType::FP32)
                    throw runtime_error("Serialized ragged SparseCategoricalCrossEntropy loss dtype must be FP16 or FP32.");
                const json& rawJson = j.at("ragged_raw_loss");
                SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
                    rawJson, j.at("ragged_predictions"), predictions, "SparseCategoricalCrossEntropy");
                Tensor rawValues = Tensor::deserialize(rawJson.at("values"));
                if (rawValues.getDimensions() != vector<uint64_t>{predictions.getMaxTotalValues()} ||
                    rawValues.getDataType() != serializedLossDType) {
                    throw runtime_error(
                        "Serialized ragged SparseCategoricalCrossEntropy raw loss must have packed scalar shape [max_total_values].");
                }

                categoricalCrossEntropy.labelType = LabelType::SPARSE;
                categoricalCrossEntropy.numClasses = j.at("num_classes").get<uint32_t>();
                categoricalCrossEntropy.logitsNativeLossAddedToNetwork = true;
                categoricalCrossEntropy.lossShape = LossShape::RAW;
                categoricalCrossEntropy.lossDataType = serializedLossDType;
                categoricalCrossEntropy.lossWeight = ThorImplementation::lossWeightFromJson(j);
                if (j.contains("ignore_index"))
                    categoricalCrossEntropy.ignoreIndex = j.at("ignore_index").get<uint32_t>();
                categoricalCrossEntropy.predictionsTensor = predictions.getValues();
                categoricalCrossEntropy.labelsTensor = labels.getValues();
                categoricalCrossEntropy.softmaxOutput = predictions.getValues();
                categoricalCrossEntropy.raggedPredictionsTensor = predictions;
                categoricalCrossEntropy.raggedLabelsTensor = labels;
                if (j.contains("ragged_mask")) {
                    RaggedTensor mask = SegmentedPrimitiveDetail::reconstructInput(
                        j.at("ragged_mask"), network, "SparseCategoricalCrossEntropy");
                    if (!CategoricalCrossEntropy::Builder::sparseRaggedValueIsScalar(mask.getTrailingDimensions()) ||
                        !predictions.sharesPartitionWith(mask) ||
                        predictions.getBatchSize() != mask.getBatchSize() ||
                        predictions.getMaxTotalValues() != mask.getMaxTotalValues())
                        throw runtime_error(
                            "Serialized ragged SparseCategoricalCrossEntropy mask must be scalar per token on the prediction partition.");
                    const DataType maskDType = mask.getValuesDataType();
                    if (maskDType != DataType::BOOLEAN && maskDType != DataType::UINT8 && maskDType != DataType::FP16 &&
                        maskDType != DataType::FP32)
                        throw runtime_error(
                            "Serialized ragged SparseCategoricalCrossEntropy mask dtype is unsupported.");
                    categoricalCrossEntropy.raggedMaskTensor = mask;
                    categoricalCrossEntropy.maskTensor = mask.getValues();
                }
                categoricalCrossEntropy.lossShaperInput = rawValues;
                categoricalCrossEntropy.lossTensor = rawValues;
                categoricalCrossEntropy.raggedRawLossTensor = predictions.withValues(rawValues);
                categoricalCrossEntropy.network = network;
                categoricalCrossEntropy.initialized = true;
                categoricalCrossEntropy.addToNetwork(network);
                return;
            }

            SparseCategoricalCrossEntropy::Builder builder;
            builder.network(*network)
                .predictions(predictions)
                .labels(labels)
                .numClasses(serializedNumClasses)
                .lossDataType(j.at("loss_data_type").get<DataType>())
                .lossWeight(ThorImplementation::lossWeightFromJson(j).value_or(1.0f));
            if (j.contains("ignore_index")) builder.ignoreIndex(j.at("ignore_index").get<uint32_t>());
            if (j.contains("ragged_mask")) {
                RaggedTensor mask = SegmentedPrimitiveDetail::reconstructInput(
                    j.at("ragged_mask"), network, "SparseCategoricalCrossEntropy");
                builder.mask(mask);
            }
            switch (serializedShape) {
                case LossShape::NONE: builder.reportsNoLoss(); break;
                case LossShape::BATCH: builder.reportsBatchLoss(); break;
                case LossShape::PER_EXAMPLE: builder.reportsPerExampleLoss(); break;
                case LossShape::RAW: builder.reportsRawLoss(); break;
                case LossShape::PER_OUTPUT: THOR_UNREACHABLE();
            }
            (void)builder.build();
            return;
        }

        CategoricalCrossEntropy::Builder builder;
        builder.network(*network)
            .predictions(predictions)
            .labels(labels)
            .lossDataType(j.at("loss_data_type").get<DataType>())
            .lossWeight(ThorImplementation::lossWeightFromJson(j).value_or(1.0f));
        switch (serializedShape) {
            case LossShape::NONE: builder.reportsNoLoss(); break;
            case LossShape::BATCH: builder.reportsBatchLoss(); break;
            case LossShape::PER_EXAMPLE: builder.reportsPerExampleLoss(); break;
            case LossShape::RAW: builder.reportsRawLoss(); break;
            case LossShape::PER_OUTPUT: THOR_UNREACHABLE();
        }
        (void)builder.build();
        return;
    }

    categoricalCrossEntropy.labelType = labelType;
    categoricalCrossEntropy.lossShape = j.at("loss_shape").get<Loss::LossShape>();
    categoricalCrossEntropy.lossDataType = j.at("loss_data_type").get<DataType>();
    categoricalCrossEntropy.numClasses = j.value("num_classes", 0u);
    categoricalCrossEntropy.logitsNativeLossAddedToNetwork = j.value("logits_native", false);
    if (j.contains("ignore_index"))
        categoricalCrossEntropy.ignoreIndex = j.at("ignore_index").get<uint32_t>();

    categoricalCrossEntropy.lossWeight = ThorImplementation::lossWeightFromJson(j);

    uint64_t originalTensorId;
    if (categoricalCrossEntropy.logitsNativeLossAddedToNetwork) {
        originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
        categoricalCrossEntropy.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
        categoricalCrossEntropy.softmaxOutput = categoricalCrossEntropy.predictionsTensor;
    } else {
        originalTensorId = j["softmax_output_tensor"].at("id").get<uint64_t>();
        categoricalCrossEntropy.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
        categoricalCrossEntropy.softmaxAddedToNetwork = true;
        categoricalCrossEntropy.softmaxOutput = categoricalCrossEntropy.predictionsTensor;
    }

    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    categoricalCrossEntropy.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    if (j.contains("mask_tensor")) {
        originalTensorId = j["mask_tensor"].at("id").get<uint64_t>();
        categoricalCrossEntropy.maskTensor = network->getApiTensorByOriginalId(originalTensorId);
    }

    categoricalCrossEntropy.lossTensor = Tensor::deserialize(j["loss_shaper_input_tensor"]);
    categoricalCrossEntropy.lossShaperInput = categoricalCrossEntropy.lossTensor;

    categoricalCrossEntropy.initialized = true;
    categoricalCrossEntropy.addToNetwork(network);
}

void CategoricalCrossEntropy::deserialize(const json &j, Network *network) {
    CategoricalCrossEntropy categoricalCrossEntropy;
    deserializeInto(j, network, categoricalCrossEntropy, LabelType::DENSE, "categorical_cross_entropy");
}

void SparseCategoricalCrossEntropy::deserialize(const json &j, Network *network) {
    SparseCategoricalCrossEntropy sparseCategoricalCrossEntropy;
    deserializeInto(j, network, sparseCategoricalCrossEntropy, LabelType::SPARSE, "sparse_categorical_cross_entropy");
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("categorical_cross_entropy", &Thor::CategoricalCrossEntropy::deserialize);
    Thor::Loss::register_layer("sparse_categorical_cross_entropy", &Thor::SparseCategoricalCrossEntropy::deserialize);
    return true;
}();
}  // namespace
