#include "DeepLearning/Implementation/ThorError.h"
#include <optional>
// #pragma once
//
// #include "DeepLearning/Api/Layers/Layer.h"
// #include "DeepLearning/Implementation/Layers/NeuralNetwork/DeepDpm.h"
//
// #include <string>
// #include <vector>
//
//  namespace Thor {
//
///**
// * DeepDPM as described in https://arxiv.org/pdf/2203.14309.pdf
// * <p>
// * Plus some knobs to allow customization when desired.
// */
// class DeepDpm : public Layer {
//   public:
//    class Builder;
//    enum class MODE { PRE_TRAIN_ENCODER = 3, INITIALIZE_CLUSTER_CENTERS, TRAIN, INFER };
//
//    DeepDpm() {}
//
//    virtual ~DeepDpm() {}
//
//    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<DeepDpm>(*this); }
//
//    /**
//     * Operation is to run:
//     *      1. in PRE_TRAIN_ENCODER mode for some number of epochs
//     *      2. then in INITIALIZE_CLUSTER_CENTERS mode for some number of epochs
//     *      3. then in TRAIN mode for some number of epochs, where on even epochs the encoder is refined while the clusters are
//     *         held constant and on odd epochs the encoder is held constant while the clusters are updated.
//     * @param operatingMode
//     */
//    virtual void setOperatingMode(DeepDpm::MODE operatingMode);
//    virtual void preTrainEncoder(uint32_t numEpochs);
//    virtual void initializeClustersCenters(uint32_t numEpochs);
//
//    virtual std::optional<Tensor> getSoftAssignments() const { return featureOutput; }
//
//    virtual std::string getLayerType() const { return "DeepDpm"; }
//
//    virtual void buildSupportLayersAndAddToNetwork();
//
//   protected:
//    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
//                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
//                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
//                                                             Thor::Tensor connectingApiTensor) const;
//
//    // mem requirements are the weights
//    virtual uint64_t getFirstInstanceFixedMemRequirementInBytes() const { return 0; }
//
//    // mem requirements are the input output tensors
//    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const;
//
//    virtual uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize,
//                                                              ThorImplementation::TensorPlacement tensorPlacement) const;
//
//   private:
//    virtual std::optional<Tensor> getFeatureOutput() const { return featureOutput; }
//
//    bool disableFeatureEncoder;
//    uint32_t encodedFeatureWidth;
//    uint32_t clusteringNetNumHiddenLayers;
//    uint32_t clusteringNetNumHiddenLayerUnits;
//    uint32_t clusteringNetLearningRate;
//    uint32_t subclusteringNetNumHiddenLayers;
//    uint32_t subclusteringNetNumHiddenLayerUnits;
//    uint32_t subclusteringNetLearningRate;
//    uint32_t d;
//    std::optional<std::vector<std::vector<float>>> m;
//    float v;
//    float psiScalar;
//    float beta;
//    float alpha;
//    float kappa;
//    uint32_t initialK;
//
//    Network *network;
//};
//
///**
// * When no parameters are changed, DeepDpm defaults to the values specified in the paper.
// * They additionally used a batch size of 128. See section 7 in the paper's supplemental materials.
// * https://arxiv.org/pdf/2203.14309.pdf
// */
// class DeepDpm::Builder {
//   public:
//    virtual DeepDpm build() {
//        THOR_THROW_IF_FALSE(_network.has_value());
//        THOR_THROW_IF_FALSE(_featureInput.has_value());
//
//        THOR_THROW_IF_FALSE(!!_d.has_value());
//
//        if (!_clusteringNetNumHiddenLayers.has_value())
//            _clusteringNetNumHiddenLayers = 1;
//        if (!_clusteringNetNumHiddenLayerUnits.has_value())
//            _clusteringNetNumHiddenLayerUnits = 50;
//        if (!_subclusteringNetNumHiddenLayers.has_value())
//            _subclusteringNetNumHiddenLayers = 1;
//        if (!_subclusteringNetNumHiddenLayerUnits.has_value())
//            _subclusteringNetNumHiddenLayerUnits = 50;
//        if (!_initialK.has_value())
//            _initialK = 1;
//        if (!_beta.has_value())
//            _beta = 1.0f;
//        if (!_alpha.has_value())
//            _alpha = 10.0f;
//        if (!_kappa.has_value())
//            _kappa = 10.0f;
//        if (!_v.has_value())
//            _v = _d.value() + 2;
//        if (!_psiScalar.has_value())
//            _psiScalar = 0.005f;
//
//        if (_disableFeatureEncoder.has_value()) {
//            THOR_THROW_IF_FALSE(_m.has_value());
//            THOR_THROW_IF_FALSE(_m.value().size() == _initialK.value());
//            for (uint32_t i = 0; i < _m.value().size(); ++i)
//                THOR_THROW_IF_FALSE(_m.value()[i].size() == _d.value());
//        } else {
//            THOR_THROW_IF_FALSE(!_m.has_value());
//        }
//
//        DeepDpm deepDpm;
//        deepDpm.network = _network;
//        deepDpm.featureInput = _featureInput;
//        deepDpm.clusteringNetNumHiddenLayers = _clusteringNetNumHiddenLayers;
//        deepDpm.clusteringNetNumHiddenLayerUnits = _clusteringNetNumHiddenLayerUnits;
//        deepDpm.subclusteringNetNumHiddenLayers = _subclusteringNetNumHiddenLayers;
//        deepDpm.subclusteringNetNumHiddenLayerUnits = _subclusteringNetNumHiddenLayerUnits;
//        deepDpm.initialK = _initialK;
//        deepDpm.d = _d;
//        deepDpm.m = _m;
//        deepDpm.beta = _beta;
//        deepDpm.alpha = _alpha.value();
//        deepDpm.kappa = _kappa;
//        deepDpm.v = _v;
//        deepDpm.psiScalar = _psiScalar;
//        deepDpm.network = _network.value();
//        deepDpm.initialized = true;
//        deepDpm.buildSupportLayersAndAddToNetwork();
//        return deepDpm;
//    }
//
//    virtual DeepDpm::Builder &network(Network &_network) {
//        THOR_THROW_IF_FALSE(!this->_network.has_value());
//        this->_network = &_network;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &featureInput(Tensor _featureInput) {
//        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
//        this->_featureInput = _featureInput;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &disableFeatureEncoder() {
//        THOR_THROW_IF_FALSE(!_disableFeatureEncoder.has_value());
//        _disableFeatureEncoder = true;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &encodedFeatureWidth(uint32_t _encodedFeatureWidth) {
//        THOR_THROW_IF_FALSE(_encodedFeatureWidth > 0);
//        THOR_THROW_IF_FALSE(!this->_encodedFeatureWidth.has_value());
//        this->_encodedFeatureWidth = _encodedFeatureWidth;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &clusteringNetNumHiddenLayers(uint32_t _clusteringNetNumHiddenLayers) {
//        THOR_THROW_IF_FALSE(!this->_clusteringNetNumHiddenLayers.has_value());
//        this->_clusteringNetNumHiddenLayers = _clusteringNetNumHiddenLayers;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &clusteringNetNumHiddenLayerUnits(uint32_t _clusteringNetNumHiddenLayerUnits) {
//        THOR_THROW_IF_FALSE(_clusteringNetNumHiddenLayerUnits > 0);
//        THOR_THROW_IF_FALSE(!this->_clusteringNetNumHiddenLayerUnits.has_value());
//        this->_clusteringNetNumHiddenLayerUnits = _clusteringNetNumHiddenLayerUnits;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &clusteringNetLearningRate(float _clusteringNetLearningRate) {
//        THOR_THROW_IF_FALSE(!this->_clusteringNetLearningRate.has_value());
//        this->_clusteringNetLearningRate = _clusteringNetLearningRate;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &subclusteringNetNumHiddenLayers(uint32_t _subclusteringNetNumHiddenLayers) {
//        THOR_THROW_IF_FALSE(!this->_subclusteringNetNumHiddenLayers.has_value());
//        this->_subclusteringNetNumHiddenLayers = _subclusteringNetNumHiddenLayers;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &subclusteringNetNumHiddenLayerUnits(uint32_t _subclusteringNetNumHiddenLayerUnits) {
//        THOR_THROW_IF_FALSE(_subclusteringNetNumHiddenLayerUnits > 0);
//        THOR_THROW_IF_FALSE(!this->_subclusteringNetNumHiddenLayerUnits.has_value());
//        this->_subclusteringNetNumHiddenLayerUnits = _subclusteringNetNumHiddenLayerUnits;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &subclusteringNetLearningRate(float _subclusteringNetLearningRate) {
//        THOR_THROW_IF_FALSE(!this->_subclusteringNetLearningRate.has_value());
//        this->_subclusteringNetLearningRate = _subclusteringNetLearningRate;
//        return *this;
//    }
//
//    /**
//     * d is the number of neurons in the input layer of the clustering model,
//     * i.e. the dimension of the input to the clustering model,
//     * i.e. the output dimension of the auto encoder
//     */
//    virtual DeepDpm::Builder &d(uint32_t _d) {
//        THOR_THROW_IF_FALSE(!this->_d.has_value());
//        THOR_THROW_IF_FALSE(_d > 0);
//        if (_v.has_value())
//            THOR_THROW_IF_FALSE(_v.value() > _d - 1);
//        this->_d = _d;
//        return *this;
//    }
//
//    /**
//     * v must be > d - 1
//     */
//    virtual DeepDpm::Builder &d(float _v) {
//        THOR_THROW_IF_FALSE(!this->_v.has_value());
//        THOR_THROW_IF_FALSE(_v > 0);
//        if (_d.has_value())
//            THOR_THROW_IF_FALSE(_v > _d.value() - 1);
//        this->_v = _v;
//        return *this;
//    }
//
//    /**
//     * m is the per centroid mean of the initial centroids
//     * so m is a initialK x d matrix
//     * If starting with one centroid, it makes sense to have that centroid begin at the mean of the input data,
//     * if starting with more than one centroid it seems reasonable to evenly distribute the centroids across the feature space,
//     * so when a feature encoder is used, specifying m is not supported, instead the centroids will be spread through the feature space
//     * with near uniform density with a small amount of random adjustment.
//     *
//     * When the auto-encoder is not used, then m needs to be explicitly specified.
//     *
//     * FIXME: only options should be to use k-means - this has to be the case so that new subclustering nets can be initialized
//     */
//    virtual DeepDpm::Builder &m(std::vector<std::vector<float>> _m) { return *this; }
//
//    /**
//     * psi will be initialized to psiScalar * I where I is the intialK x initialK identity matrix.
//     */
//    virtual DeepDpm::Builder &psiScalar(float _psiScalar) {
//        THOR_THROW_IF_FALSE(!this->_psiScalar.has_value());
//        this->_psiScalar = _psiScalar;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &beta(float _beta) {
//        THOR_THROW_IF_FALSE(!this->_beta.has_value());
//        this->_beta = _beta;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &alpha(float _alpha) {
//        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
//        this->_alpha = _alpha.value();
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &kappa(float _kappa) {
//        THOR_THROW_IF_FALSE(!this->_kappa.has_value());
//        this->_kappa = _kappa;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &initialK(uint32_t _initialK) {
//        THOR_THROW_IF_FALSE(!this->_initialK.has_value());
//        THOR_THROW_IF_FALSE(_initialK > 0);
//        this->_initialK = _initialK;
//        return *this;
//    }
//
//   private:
//    std::optional<Network *> _network;
//    std::optional<Tensor> _featureInput;
//    std::optional<bool> _disableFeatureEncoder;
//    std::optional<uint32_t> _encodedFeatureWidth;
//    std::optional<uint32_t> _clusteringNetNumHiddenLayers;
//    std::optional<uint32_t> _clusteringNetNumHiddenLayerUnits;
//    std::optional<uint32_t> _clusteringNetLearningRate;
//    std::optional<uint32_t> _subclusteringNetNumHiddenLayers;
//    std::optional<uint32_t> _subclusteringNetNumHiddenLayerUnits;
//    std::optional<uint32_t> _subclusteringNetLearningRate;
//    std::optional<uint32_t> _d;
//    std::optional<std::vector<std::vector<float>>> _m;
//    std::optional<float> _v;
//    std::optional<float> _psiScalar;
//    std::optional<float> _beta;
//    std::optional<float> _alpha;
//    std::optional<float> _kappa;
//    std::optional<uint32_t> _initialK;
//};
//
//}  // namespace Thor
