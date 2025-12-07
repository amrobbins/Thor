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
//    virtual Optional<Tensor> getSoftAssignments() const { return featureOutput; }
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
//    virtual Optional<Tensor> getFeatureOutput() const { return featureOutput; }
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
//    Optional<std::vector<std::vector<float>>> m;
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
//        assert(_network.isPresent());
//        assert(_featureInput.isPresent());
//
//        assert(!_d.isEmpty());
//
//        if (_clusteringNetNumHiddenLayers.isEmpty())
//            _clusteringNetNumHiddenLayers = 1;
//        if (_clusteringNetNumHiddenLayerUnits.isEmpty())
//            _clusteringNetNumHiddenLayerUnits = 50;
//        if (_subclusteringNetNumHiddenLayers.isEmpty())
//            _subclusteringNetNumHiddenLayers = 1;
//        if (_subclusteringNetNumHiddenLayerUnits.isEmpty())
//            _subclusteringNetNumHiddenLayerUnits = 50;
//        if (_initialK.isEmpty())
//            _initialK = 1;
//        if (_beta.isEmpty())
//            _beta = 1.0f;
//        if (_alpha.isEmpty())
//            _alpha = 10.0f;
//        if (_kappa.isEmpty())
//            _kappa = 10.0f;
//        if (_v.isEmpty())
//            _v = _d.get() + 2;
//        if (_psiScalar.isEmpty())
//            _psiScalar = 0.005f;
//
//        if (_disableFeatureEncoder.isPresent()) {
//            assert(_m.isPresent());
//            assert(_m.get().size() == _initialK.get());
//            for (uint32_t i = 0; i < _m.get().size(); ++i)
//                assert(_m.get()[i].size() == _d.get());
//        } else {
//            assert(_m.isEmpty());
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
//        deepDpm.alpha = _alpha;
//        deepDpm.kappa = _kappa;
//        deepDpm.v = _v;
//        deepDpm.psiScalar = _psiScalar;
//        deepDpm.network = _network.get();
//        deepDpm.initialized = true;
//        deepDpm.buildSupportLayersAndAddToNetwork();
//        return deepDpm;
//    }
//
//    virtual DeepDpm::Builder &network(Network &_network) {
//        assert(!this->_network.isPresent());
//        this->_network = &_network;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &featureInput(Tensor _featureInput) {
//        assert(this->_featureInput.isEmpty());
//        this->_featureInput = _featureInput;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &disableFeatureEncoder() {
//        assert(_disableFeatureEncoder.isEmpty());
//        _disableFeatureEncoder = true;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &encodedFeatureWidth(uint32_t _encodedFeatureWidth) {
//        assert(_encodedFeatureWidth > 0);
//        assert(!this->_encodedFeatureWidth.isPresent());
//        this->_encodedFeatureWidth = _encodedFeatureWidth;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &clusteringNetNumHiddenLayers(uint32_t _clusteringNetNumHiddenLayers) {
//        assert(!this->_clusteringNetNumHiddenLayers.isPresent());
//        this->_clusteringNetNumHiddenLayers = _clusteringNetNumHiddenLayers;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &clusteringNetNumHiddenLayerUnits(uint32_t _clusteringNetNumHiddenLayerUnits) {
//        assert(_clusteringNetNumHiddenLayerUnits > 0);
//        assert(!this->_clusteringNetNumHiddenLayerUnits.isPresent());
//        this->_clusteringNetNumHiddenLayerUnits = _clusteringNetNumHiddenLayerUnits;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &clusteringNetLearningRate(float _clusteringNetLearningRate) {
//        assert(!this->_clusteringNetLearningRate.isPresent());
//        this->_clusteringNetLearningRate = _clusteringNetLearningRate;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &subclusteringNetNumHiddenLayers(uint32_t _subclusteringNetNumHiddenLayers) {
//        assert(!this->_subclusteringNetNumHiddenLayers.isPresent());
//        this->_subclusteringNetNumHiddenLayers = _subclusteringNetNumHiddenLayers;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &subclusteringNetNumHiddenLayerUnits(uint32_t _subclusteringNetNumHiddenLayerUnits) {
//        assert(_subclusteringNetNumHiddenLayerUnits > 0);
//        assert(!this->_subclusteringNetNumHiddenLayerUnits.isPresent());
//        this->_subclusteringNetNumHiddenLayerUnits = _subclusteringNetNumHiddenLayerUnits;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &subclusteringNetLearningRate(float _subclusteringNetLearningRate) {
//        assert(!this->_subclusteringNetLearningRate.isPresent());
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
//        assert(!this->_d.isPresent());
//        assert(_d > 0);
//        if (_v.isPresent())
//            assert(_v.get() > _d - 1);
//        this->_d = _d;
//        return *this;
//    }
//
//    /**
//     * v must be > d - 1
//     */
//    virtual DeepDpm::Builder &d(float _v) {
//        assert(!this->_v.isPresent());
//        assert(_v > 0);
//        if (_d.isPresent())
//            assert(_v > _d.get() - 1);
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
//        assert(!this->_psiScalar.isPresent());
//        this->_psiScalar = _psiScalar;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &beta(float _beta) {
//        assert(!this->_beta.isPresent());
//        this->_beta = _beta;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &alpha(float _alpha) {
//        assert(!this->_alpha.isPresent());
//        this->_alpha = _alpha;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &kappa(float _kappa) {
//        assert(!this->_kappa.isPresent());
//        this->_kappa = _kappa;
//        return *this;
//    }
//
//    virtual DeepDpm::Builder &initialK(uint32_t _initialK) {
//        assert(!this->_initialK.isPresent());
//        assert(_initialK > 0);
//        this->_initialK = _initialK;
//        return *this;
//    }
//
//   private:
//    Optional<Network *> _network;
//    Optional<Tensor> _featureInput;
//    Optional<bool> _disableFeatureEncoder;
//    Optional<uint32_t> _encodedFeatureWidth;
//    Optional<uint32_t> _clusteringNetNumHiddenLayers;
//    Optional<uint32_t> _clusteringNetNumHiddenLayerUnits;
//    Optional<uint32_t> _clusteringNetLearningRate;
//    Optional<uint32_t> _subclusteringNetNumHiddenLayers;
//    Optional<uint32_t> _subclusteringNetNumHiddenLayerUnits;
//    Optional<uint32_t> _subclusteringNetLearningRate;
//    Optional<uint32_t> _d;
//    Optional<std::vector<std::vector<float>>> _m;
//    Optional<float> _v;
//    Optional<float> _psiScalar;
//    Optional<float> _beta;
//    Optional<float> _alpha;
//    Optional<float> _kappa;
//    Optional<uint32_t> _initialK;
//};
//
//}  // namespace Thor
