#include "DeepLearning/Api/Optimizers/Adam.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

shared_ptr<ThorImplementation::Optimizer> Adam::stamp(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer) {
    return make_shared<ThorImplementation::Adam>(trainableLayer, alpha, beta1, beta2, epsilon);
}

void Adam::setAlpha(float newAlpha) {
    alpha = newAlpha;
    updateParameters();
}

void Adam::setBeta1(float newBeta1) {
    beta1 = newBeta1;
    updateParameters();
}

void Adam::setBeta2(float newBeta2) {
    beta2 = newBeta2;
    updateParameters();
}

void Adam::setEpsilon(float newEpsilon) {
    epsilon = newEpsilon;
    updateParameters();
}

float Adam::getAlpha() { return alpha; }

float Adam::getBeta1() { return beta1; }

float Adam::getBeta2() { return beta2; }

float Adam::getEpsilon() { return epsilon; }

shared_ptr<Optimizer> Adam::clone() const { return make_shared<Adam>(*this); }

void Adam::updateParameters() {
    assert(network != nullptr);
    uint32_t numStamps = network->getNumStamps();
    for (uint32_t i = 0; i < numStamps; ++i) {
        ThorImplementation::StampedNetwork &stampedNetwork = network->getStampedNetwork(i);
        uint32_t numTrainableLayers = stampedNetwork.getNumTrainableLayers();
        for (uint32_t j = 0; j < numTrainableLayers; ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> &trainableLayer = stampedNetwork.getTrainableLayer(j);
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybePhysicalOptimizer = trainableLayer->getOptimizer();
            assert(maybePhysicalOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybePhysicalOptimizer.get();
            shared_ptr<ThorImplementation::Adam> physicalAdam = dynamic_pointer_cast<ThorImplementation::Adam>(optimizer);
            if (physicalAdam == nullptr || physicalAdam->getId() != getId())
                continue;
            physicalAdam->setAlpha(alpha);
            physicalAdam->setBeta1(beta1);
            physicalAdam->setBeta2(beta2);
            physicalAdam->setEpsilon(epsilon);
        }
    }
}

json Adam::serialize(thor_file::TarWriter &archiveWriter,
                     Stream stream,
                     TrainableWeightsBiasesLayer const *owningLayer,
                     shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> physicalOwningLayer) const {
    json j;
    j["optimizer_type"] = string("adam");
    j["version"] = getVersion();

    // Get the params and weights from the physical layer.
    // Record the params and dump the weights to files.
    shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = physicalOwningLayer->getOptimizer();
    shared_ptr<ThorImplementation::Adam> physicalAdam = dynamic_pointer_cast<ThorImplementation::Adam>(physicalOptimizer);
    assert(physicalAdam != nullptr);
    j["t"] = physicalAdam->getT();
    j["alpha"] = physicalAdam->getAlpha();
    j["beta1"] = physicalAdam->getBeta1();
    j["beta2"] = physicalAdam->getBeta2();
    j["epsilon"] = physicalAdam->getEpsilon();

    ThorImplementation::TensorPlacement cpuPlacement =
        ThorImplementation::TensorPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);

    string optimizerName = string("layer") + to_string(owningLayer->getId()) + "_adam";
    string mFile = optimizerName + "_m.gds";
    string vFile = optimizerName + "_v.gds";
    j["m_tensor"] = mFile;
    j["v_tensor"] = vFile;

    ThorImplementation::Tensor m = physicalAdam->getM();
    ThorImplementation::Tensor mBuffer = m.clone(cpuPlacement);
    mBuffer.copyFromAsync(m, stream);

    ThorImplementation::Tensor v = physicalAdam->getM();
    ThorImplementation::Tensor vBuffer = v.clone(cpuPlacement);
    vBuffer.copyFromAsync(v, stream);

    ThorImplementation::Tensor mBias;
    ThorImplementation::Tensor mBiasBuffer;
    ThorImplementation::Tensor vBias;
    ThorImplementation::Tensor vBiasBuffer;
    string mBiasFile;
    string vBiasFile;
    if (physicalAdam->getMBias().isPresent()) {
        assert(physicalAdam->getVBias().isPresent());
        mBiasFile = optimizerName + "_m_bias.gds";
        vBiasFile = optimizerName + "_v_bias.gds";
        j["m_bias_tensor"] = mBiasFile;
        j["v_bias_tensor"] = vBiasFile;

        mBias = physicalAdam->getMBias();
        mBiasBuffer = mBias.clone(cpuPlacement);
        mBiasBuffer.copyFromAsync(mBias, stream);

        vBias = physicalAdam->getVBias();
        vBiasBuffer = vBias.clone(cpuPlacement);
        vBiasBuffer.copyFromAsync(vBias, stream);
    }

    stream.synchronize();

    archiveWriter.addArchiveFile(mFile, mBuffer.getMemPtr(), m.getArraySizeInBytes());
    archiveWriter.addArchiveFile(vFile, vBuffer.getMemPtr(), v.getArraySizeInBytes());
    if (physicalAdam->getMBias().isPresent()) {
        archiveWriter.addArchiveFile(mBiasFile, mBiasBuffer.getMemPtr(), mBias.getArraySizeInBytes());
        archiveWriter.addArchiveFile(vBiasFile, vBiasBuffer.getMemPtr(), vBias.getArraySizeInBytes());
    }

    return j;
}

shared_ptr<Optimizer> Adam::deserialize(thor_file::TarReader &archiveReader, const json &j) {
    if (j.at("optimizer_type").get<string>() != "adam")
        throw runtime_error("Layer type mismatch in Adam::deserialize: " + j.at("type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Adam::deserialize: " + j["version"].get<string>());

    float t = j.at("t").get<float>();
    float alpha = j.at("alpha").get<float>();
    float beta1 = j.at("beta1").get<float>();
    float beta2 = j.at("beta2").get<float>();
    float epsilon = j.at("epsilon").get<float>();

    Optional<string> mFile;
    Optional<string> vFile;
    Optional<string> mBiasFile;
    Optional<string> vBiasFile;
    if (j.contains("weights_m_file")) {
        assert(j.contains("weights_v_file"));
        mFile = j.at("weights_m_file").get<string>();
        vFile = j.at("weights_v_file").get<string>();
    }
    if (j.contains("biases_m_file")) {
        assert(j.contains("biases_v_file"));
        mBiasFile = j.at("biases_m_file").get<string>();
        vBiasFile = j.at("biases_v_file").get<string>();
    }

    Adam adam;
    adam.t = t;
    adam.alpha = alpha;
    adam.beta1 = beta1;
    adam.beta2 = beta2;
    adam.epsilon = epsilon;
    adam.archiveReader = &archiveReader;
    adam.mFile = mFile;
    adam.vFile = vFile;
    adam.mBiasFile = mBiasFile;
    adam.vBiasFile = vBiasFile;
    return adam.clone();
}

vector<Event> Adam::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                               bool isFirstStamp,
                               shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                               Optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::Adam> physicalAdam = dynamic_pointer_cast<ThorImplementation::Adam>(physicalOptimizer);
    assert(physicalAdam != nullptr);

    ThorImplementation::Tensor m = physicalAdam->getM();
    ThorImplementation::Tensor v = physicalAdam->getV();
    Optional<ThorImplementation::Tensor> mBias = physicalAdam->getMBias();
    Optional<ThorImplementation::Tensor> vBias = physicalAdam->getVBias();
    Stream stream = physicalAdam->getGradientUpdateStream();

    // Parameter values are initialized right now, based on 1 of 3 methods:
    // 1. Copy from another optimizer whose parameters have already been set - when stamping more than one stamp
    //      * So this is once per GPU since multiple stamps on the same GPU share the weights
    // 2. Copy from a file - when loading a saved network with a saved optimizer
    // 3. Run an initializer to set the weights - on an untrained network or when the optimizer has not been saved
    if (!isFirstStamp) {
        // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
        assert(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.isPresent())
            stream.waitEvent(sisterOptimizerLoadedEvent);
        shared_ptr<ThorImplementation::Adam> sisterPhysicalAdam = dynamic_pointer_cast<ThorImplementation::Adam>(sisterPhysicalOptimizer);
        assert(sisterPhysicalAdam != nullptr);
        m.copyFromAsync(sisterPhysicalAdam->getM(), stream);
        v.copyFromAsync(sisterPhysicalAdam->getV(), stream);
        assert(vBias.isPresent() == mBias.isPresent());
        if (mBias.isPresent()) {
            mBias.get().copyFromAsync(sisterPhysicalAdam->getMBias(), stream);
            vBias.get().copyFromAsync(sisterPhysicalAdam->getVBias(), stream);
        }
    } else if (mFile.isPresent()) {
        assert(vFile.isPresent());
        assert(archiveReader != nullptr);
        if (physicalAdam->getMBias().isPresent()) {
            assert(mBiasFile.isPresent());
            assert(vBiasFile.isPresent());
        }

        ThorImplementation::TensorPlacement cpuPlacement =
            ThorImplementation::TensorPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);

        ThorImplementation::Tensor mBuffer = m.clone(cpuPlacement);
        archiveReader->readFile(mFile.get(), mBuffer.getMemPtr<void>(), m.getArraySizeInBytes());
        m.copyFromAsync(mBuffer, stream);

        ThorImplementation::Tensor vBuffer = v.clone(cpuPlacement);
        archiveReader->readFile(vFile.get(), vBuffer.getMemPtr<void>(), v.getArraySizeInBytes());
        v.copyFromAsync(vBuffer, stream);

        if (mBias.isPresent()) {
            assert(mBiasFile.isPresent());
            assert(vBiasFile.isPresent());

            ThorImplementation::Tensor mBiasBuffer = mBias.get().clone(cpuPlacement);
            archiveReader->readFile(mBiasFile.get(), mBiasBuffer.getMemPtr<void>(), mBias.get().getArraySizeInBytes());
            mBias.get().copyFromAsync(mBiasBuffer, stream);

            ThorImplementation::Tensor vBiasBuffer = vBias.get().clone(cpuPlacement);
            archiveReader->readFile(vBiasFile.get(), vBiasBuffer.getMemPtr<void>(), vBias.get().getArraySizeInBytes());
            vBias.get().copyFromAsync(vBiasBuffer, stream);
        }

        // Can't use the files later, they may not still be there
        archiveReader = nullptr;
        mFile.clear();
        vFile.clear();
        mBiasFile.clear();
        vBiasFile.clear();

        // I need to synchronize here to keep bounce buffer alive
        stream.synchronize();
    } else {
        m.memsetAsync(stream, 0);
        v.memsetAsync(stream, 0);
        if (mBias.isPresent()) {
            mBias.get().memsetAsync(stream, 0);
            vBias.get().memsetAsync(stream, 0);
        }
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("adam", &Thor::Adam::deserialize);
    return true;
}();
}  // namespace
