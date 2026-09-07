#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"
#include <optional>

namespace Thor {

class CategoricalCrossEntropy;

class Softmax : public Activation {
   public:
    class Builder;
    Softmax() : backwardComputedExternally(false) {}

    ~Softmax() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Softmax> myClone = std::make_shared<Softmax>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    static void validateFeatureInputDataType(DataType dataType) {
        switch (dataType) {
            case DataType::FP16:
            case DataType::BF16:
            case DataType::FP32:
                return;
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
                throw std::invalid_argument(
                    "Softmax does not accept FP8 input tensors. Cast to FP16, BF16, or FP32 before Softmax.");
            default:
                throw std::invalid_argument("Softmax supports FP16, BF16, and FP32 input tensors.");
        }
    }

    bool supportsRaggedStandalone() const override { return true; }
    bool supportsRaggedLearningLayerFusion() const override { return false; }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.softmax();
    }

    ThorImplementation::RaggedExpression toRaggedExpression(
        const ThorImplementation::RaggedExpression& input) const override {
        // Ordinary ragged Softmax is tokenwise: normalize independently over the
        // final trailing/channel dimension for every active packed value.  Do not
        // route this through the default mapValues(toExpression) hook: R11A's
        // RaggedExpression::softmax() carries the exact active-prefix metadata and
        // lowers to the dedicated ragged softmax forward/backward boundary.
        return input.softmax();
    }

    std::string getLayerType() const override { return "Softmax"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Softmax::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "softmax")
            throw std::runtime_error("Layer type mismatch in Softmax::deserialize: " + j.at("layer_type").get<std::string>());

        if (j.value("use_ragged", false)) {
            Softmax softmax;
            softmax.initialized = true;
            softmax.deserializeStandaloneFields(j, network);
            THOR_THROW_IF_FALSE(softmax.getFeatureInput().has_value());
            validateFeatureInputDataType(softmax.getFeatureInput().value().getDataType());
            softmax.addToNetwork(network);
            return;
        }

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
        validateFeatureInputDataType(featureInput.getDataType());

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Softmax softmax;
        softmax.featureInput = featureInput;
        softmax.featureOutput = featureOutput;
        softmax.initialized = true;
        softmax.addToNetwork(network);
    }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)drivingLayer;
        (void)drivingApiLayer;
        THOR_THROW_IF_FALSE(initialized);
        if (featureInput.has_value() && connectingApiTensor == featureInput.value()) {
            validateFeatureInputDataType(connectingApiTensor.getDataType());
        }

        if (backwardComputedExternally) {
            // Loss-owned softmax keeps the legacy dense external-backward physical-layer contract.
            // CategoricalCrossEntropy is still dense-only in R11B, so this path has no
            // row-partition input and must be driven by the values tensor itself.
            THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());
            return std::make_shared<ThorImplementation::Softmax>(true);
        }

        return stampExpressionBackedActivation(placement, connectingApiTensor, inferenceOnly);
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)tensorPlacement;
        if (!backwardComputedExternally) return getExpressionBackedActivationMemRequirementInBytes(batchSize);
        // Loss-owned dense softmax keeps the legacy physical-layer accounting.
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }

    bool backwardComputedExternally;
};

class Softmax::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Softmax> softmax = std::make_shared<Softmax>();
        softmax->backwardComputedExternally = _backwardComputedExternally.value_or(false);
        if (_featureInput.has_value()) {
            // Standalone layer support.  Use the common initializer so a RaggedTensor
            // remains ragged and preserves its logical row partition exactly.
            THOR_THROW_IF_FALSE(_network.has_value());
            applyStandaloneConfiguration(*softmax);
            softmax->initialized = true;
            softmax->addToNetwork(_network.value());
        } else {
            // Template activation support.
            softmax->initialized = true;
        }

        return softmax;
    }

    Softmax::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Softmax::Builder &featureInput(Tensor _featureInput) override {
        Softmax::validateFeatureInputDataType(_featureInput.getDataType());
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    Softmax::Builder &featureInput(RaggedTensor _featureInput) override {
        Softmax::validateFeatureInputDataType(_featureInput.getValuesDataType());
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

   protected:
    Softmax::Builder &backwardComputedExternally() {
        THOR_THROW_IF_FALSE(!_backwardComputedExternally.has_value());
        _backwardComputedExternally = true;
        return *this;
    }

   private:
    std::optional<bool> _backwardComputedExternally;

    friend class Thor::CategoricalCrossEntropy;
};

}  // namespace Thor
