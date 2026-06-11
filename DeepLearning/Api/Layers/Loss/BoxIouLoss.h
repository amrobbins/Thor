#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/BoxIouLoss.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

template <typename DerivedBuilder>
class BoxIouLossBuilderCommon;

class BoxIouLoss : public Loss {
   public:
    using Kind = ThorImplementation::BoxIouLoss::Kind;

    BoxIouLoss() = default;
    ~BoxIouLoss() override = default;

    Tensor getPredictions() const override { return predictionsTensor; }
    Tensor getLabels() const override { return labelsTensor; }

    std::optional<Tensor> getFeatureInput() const override { return predictionsTensor; }
    std::vector<Tensor> getLossInputTensors() const override { return {predictionsTensor, labelsTensor}; }

    int getConnectionType(Tensor connectingTensor) const override {
        if (connectingTensor == labelsTensor) {
            return static_cast<int>(ThorImplementation::Loss::ConnectionType::LABELS);
        }
        if (connectingTensor == predictionsTensor) {
            return static_cast<int>(ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD);
        }
        if (connectingTensor == lossTensor) {
            return 0;
        }
        THOR_UNREACHABLE();
    }

    float getEps() const { return eps; }
    const char* getBoxFormat() const { return "xyxy"; }

    nlohmann::json architectureJson() const override;

   protected:
    virtual Kind getKind() const = 0;
    virtual const char* getSerializedLayerType() const = 0;
    virtual std::shared_ptr<BoxIouLoss> makeRawSupportLoss() const = 0;

    virtual bool isMultiLayer() const { return lossShape != LossShape::RAW; }
    virtual void buildSupportLayersAndAddToNetwork();

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)placement;
        (void)drivingLayer;
        (void)drivingApiLayer;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);
        std::shared_ptr<ThorImplementation::BoxIouLoss> boxIouLoss =
            std::make_shared<ThorImplementation::BoxIouLoss>(getKind(), lossDataType, eps, lossWeight);
        boxIouLoss->setConstructForInferenceOnly(inferenceOnly);
        return boxIouLoss;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint64_t lossShaperBytes = 0;
        if (isMultiLayer()) {
            lossShaperBytes = LossShaper::Builder()
                                  .lossInput(lossTensor)
                                  .reportsBatchLoss()
                                  .getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        }

        uint64_t bytes = 4;
        bytes += batchSize * predictionsTensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * labelsTensor.getTotalSizeInBytes();
        bytes += batchSize * lossTensor.getTotalSizeInBytes();
        return bytes + lossShaperBytes;
    }

    template <typename DerivedBuilder>
    friend class BoxIouLossBuilderCommon;

    static void populateAndAdd(BoxIouLoss& loss,
                               Network* network,
                               Tensor predictions,
                               Tensor labels,
                               LossShape lossShape,
                               DataType lossDataType,
                               float eps,
                               std::optional<float> lossWeight);
    static void deserializeInto(const nlohmann::json& j, Network* network, BoxIouLoss& loss, const std::string& expectedLayerType);
    static std::vector<uint64_t> rawLossDimensionsForBoxes(const std::vector<uint64_t>& boxDims);

    float eps = 1.0e-7f;
};

template <typename DerivedBuilder>
class BoxIouLossBuilderCommon {
   public:
    virtual ~BoxIouLossBuilderCommon() = default;

    DerivedBuilder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return static_cast<DerivedBuilder&>(*this);
    }

    DerivedBuilder& predictions(Tensor predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!predictions.getDimensions().empty());
        this->_predictions = predictions;
        return static_cast<DerivedBuilder&>(*this);
    }

    DerivedBuilder& labels(Tensor labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!labels.getDimensions().empty());
        this->_labels = labels;
        return static_cast<DerivedBuilder&>(*this);
    }

    DerivedBuilder& eps(float eps) {
        THOR_THROW_IF_FALSE(!this->_eps.has_value());
        THOR_THROW_IF_FALSE(eps > 0.0f);
        this->_eps = eps;
        return static_cast<DerivedBuilder&>(*this);
    }

    DerivedBuilder& reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = Loss::LossShape::BATCH;
        return static_cast<DerivedBuilder&>(*this);
    }

    DerivedBuilder& reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = Loss::LossShape::ELEMENTWISE;
        return static_cast<DerivedBuilder&>(*this);
    }

    DerivedBuilder& reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = Loss::LossShape::CLASSWISE;
        return static_cast<DerivedBuilder&>(*this);
    }

    DerivedBuilder& reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = Loss::LossShape::RAW;
        return static_cast<DerivedBuilder&>(*this);
    }

    DerivedBuilder& lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return static_cast<DerivedBuilder&>(*this);
    }

    DerivedBuilder& lossDataType(DataType lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(lossDataType == DataType::FP16 || lossDataType == DataType::FP32);
        this->_lossDataType = lossDataType;
        return static_cast<DerivedBuilder&>(*this);
    }

   protected:
    void populate(BoxIouLoss& loss) const {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());

        Loss::LossShape effectiveLossShape = _lossShape.value_or(Loss::LossShape::BATCH);
        DataType effectiveLossDataType = _lossDataType.value_or(_predictions.value().getDataType());
        float effectiveEps = _eps.value_or(1.0e-7f);
        std::optional<float> effectiveLossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        BoxIouLoss::populateAndAdd(loss,
                                   _network.value(),
                                   _predictions.value(),
                                   _labels.value(),
                                   effectiveLossShape,
                                   effectiveLossDataType,
                                   effectiveEps,
                                   effectiveLossWeight);
    }

    std::optional<Network*> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<Loss::LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<float> _eps;
};

class IoULoss : public BoxIouLoss {
   public:
    class Builder;
    IoULoss() = default;
    std::shared_ptr<Layer> clone() const override { return std::make_shared<IoULoss>(*this); }
    std::string getLayerType() const override { return "IoULoss"; }
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    Kind getKind() const override { return Kind::IOU; }
    const char* getSerializedLayerType() const override { return "iou_loss"; }
    std::shared_ptr<BoxIouLoss> makeRawSupportLoss() const override { return std::make_shared<IoULoss>(); }
};

class IoULoss::Builder : public BoxIouLossBuilderCommon<IoULoss::Builder> {
   public:
    IoULoss build() {
        IoULoss loss;
        populate(loss);
        return loss;
    }
};

class GIoULoss : public BoxIouLoss {
   public:
    class Builder;
    GIoULoss() = default;
    std::shared_ptr<Layer> clone() const override { return std::make_shared<GIoULoss>(*this); }
    std::string getLayerType() const override { return "GIoULoss"; }
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    Kind getKind() const override { return Kind::GIOU; }
    const char* getSerializedLayerType() const override { return "giou_loss"; }
    std::shared_ptr<BoxIouLoss> makeRawSupportLoss() const override { return std::make_shared<GIoULoss>(); }
};

class GIoULoss::Builder : public BoxIouLossBuilderCommon<GIoULoss::Builder> {
   public:
    GIoULoss build() {
        GIoULoss loss;
        populate(loss);
        return loss;
    }
};

class DIoULoss : public BoxIouLoss {
   public:
    class Builder;
    DIoULoss() = default;
    std::shared_ptr<Layer> clone() const override { return std::make_shared<DIoULoss>(*this); }
    std::string getLayerType() const override { return "DIoULoss"; }
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    Kind getKind() const override { return Kind::DIOU; }
    const char* getSerializedLayerType() const override { return "diou_loss"; }
    std::shared_ptr<BoxIouLoss> makeRawSupportLoss() const override { return std::make_shared<DIoULoss>(); }
};

class DIoULoss::Builder : public BoxIouLossBuilderCommon<DIoULoss::Builder> {
   public:
    DIoULoss build() {
        DIoULoss loss;
        populate(loss);
        return loss;
    }
};

class CIoULoss : public BoxIouLoss {
   public:
    class Builder;
    CIoULoss() = default;
    std::shared_ptr<Layer> clone() const override { return std::make_shared<CIoULoss>(*this); }
    std::string getLayerType() const override { return "CIoULoss"; }
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    Kind getKind() const override { return Kind::CIOU; }
    const char* getSerializedLayerType() const override { return "ciou_loss"; }
    std::shared_ptr<BoxIouLoss> makeRawSupportLoss() const override { return std::make_shared<CIoULoss>(); }
};

class CIoULoss::Builder : public BoxIouLossBuilderCommon<CIoULoss::Builder> {
   public:
    CIoULoss build() {
        CIoULoss loss;
        populate(loss);
        return loss;
    }
};

}  // namespace Thor
