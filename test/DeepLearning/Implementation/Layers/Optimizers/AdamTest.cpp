#include "Thor.h"
#include "gtest/gtest.h"

#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

using namespace ThorImplementation;
using namespace std;

shared_ptr<TrainableWeightsBiasesLayer> constructTrainableLayer(bool hasBias = true) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, {(uint64_t)(rand() % 80) + 1, (uint64_t)(rand() % 150) + 1});

    Tensor exampleInputTensor(gpuPlacement, descriptor);
    shared_ptr<NetworkInput> networkInput = make_shared<NetworkInput>(exampleInputTensor);
    shared_ptr<FullyConnected> fc0 = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, hasBias);
    shared_ptr<FullyConnected> fc1 = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, true);
    networkInput->connectToNextLayer(fc0.get());
    fc0->connectToNextLayer(fc1.get());
    return fc0;
}

shared_ptr<TrainableWeightsBiasesLayer> constructTrainableLayerWithMultipleConnections(uint32_t numConnections, bool hasBias = true) {
    assert(numConnections > 0);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, {(uint64_t)(rand() % 80) + 1, (uint64_t)(rand() % 150) + 1});

    Tensor exampleInputTensor(gpuPlacement, descriptor);
    shared_ptr<NetworkInput> networkInput = make_shared<NetworkInput>(exampleInputTensor);
    shared_ptr<FullyConnected> fc0 = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, hasBias);
    networkInput->connectToNextLayer(fc0.get());

    for (uint32_t connection = 0; connection < numConnections; ++connection) {
        shared_ptr<NetworkInput> networkInputOther = make_shared<NetworkInput>(exampleInputTensor);
        networkInputOther->connectToNextLayer(fc0.get());
        shared_ptr<FullyConnected> fcOther = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, true);
        fc0->connectToNextLayer(fcOther.get());
    }
    return fc0;
}

template <typename T>
void computeBiasesGradientCpu(T *errorIn, T *biasesGradient, uint32_t batchSize, uint32_t exampleSize, bool accumulate) {
    if (!accumulate) {
        for (uint32_t i = 0; i < exampleSize; ++i)
            biasesGradient[i] = T(0.0f);
    }

    for (uint32_t i = 0; i < batchSize; ++i) {
        for (uint32_t j = 0; j < exampleSize; ++j) {
            T gradientComponent = errorIn[i * exampleSize + j];
            if (accumulate || i != 0)
                biasesGradient[j] += gradientComponent;
            else
                biasesGradient[j] = gradientComponent;
        }
    }
}

// Test the Adam constructor
TEST(AdamTest, Constructor) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f);
    EXPECT_EQ(adam.getAlpha(), 0.1f);
    EXPECT_EQ(adam.getBeta1(), 0.9f);
    EXPECT_EQ(adam.getBeta2(), 0.999f);
    EXPECT_EQ(adam.getEpsilon(), 1e-8f);
    EXPECT_EQ(adam.getT(), 0.0f);
    adam.setT(5.0f);
    EXPECT_EQ(adam.getT(), 5.0f);
}

// Test the Adam::setAlpha function
TEST(AdamTest, SetAlpha) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f);
    adam.setAlpha(0.2f);
    EXPECT_EQ(adam.getAlpha(), 0.2f);
}

// Test the Adam::setBeta1 function
TEST(AdamTest, SetBeta1) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f);
    adam.setBeta1(0.8f);
    EXPECT_EQ(adam.getBeta1(), 0.8f);
}

// Test the Adam::setBeta2 function
TEST(AdamTest, SetBeta2) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f);
    adam.setBeta2(0.998f);
    EXPECT_EQ(adam.getBeta2(), 0.998f);
}

// Test the Adam::setEpsilon function
TEST(AdamTest, SetEpsilon) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f);
    adam.setEpsilon(1e-7f);
    EXPECT_EQ(adam.getEpsilon(), 1e-7f);
}

TEST(AdamTest, updateHyperParameters) {
    // FIXME: Implement
}
TEST(AdamTest, getAllHyperParameters) {
    // FIXME: Implement
}

TEST(AdamTest, Adam_SingleStep_FromInjectedGradient_IsCorrect) {
    TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 2;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = false;

    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpu, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpu));
    LayerTestHelper::connectNetwork(layers);

    const half lr0 = 0.1f;
    const half beta1 = 0.9f;
    const half beta2 = 0.999f;
    const half eps_adam = 1e-8f;

    auto adam = std::make_shared<Adam>(fc, lr0, beta1, beta2, eps_adam);
    adam->testSetDataType(TensorDescriptor::DataType::FP16);
    adam->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(adam));

    LayerTestHelper::initializeNetwork(layers);
    Stream stream = layers.front()->getStream();

    Tensor w = fc->getWeights();
    Tensor w_h = w.clone(cpu);
    half *w0 = w_h.getMemPtr<half>();

    // W0 = [[1,2],[3,4]]
    w0[0] = (half)1.0f;
    w0[1] = (half)2.0f;
    w0[2] = (half)3.0f;
    w0[3] = (half)4.0f;
    w.copyFromAsync(w_h, stream);

    Tensor g = adam->getWeightsGradient();
    Tensor m = adam->getM();
    Tensor v = adam->getV();

    // Make sure these start at 0
    m.memsetAsync(stream, 0);
    v.memsetAsync(stream, 0);

    // Inject gradient G = [[10,20],[30,40]]
    Tensor g_h = g.clone(cpu);
    half *gh = g_h.getMemPtr<half>();
    gh[0] = (half)10.0f;
    gh[1] = (half)20.0f;
    gh[2] = (half)30.0f;
    gh[3] = (half)40.0f;
    g.copyFromAsync(g_h, stream);

    stream.synchronize();

    // ---- Act: compute Adam update from injected gradient (NEW helper) ----
    adam->stepFromPrecomputedGradient(false);

    // Apply the update to weights the same way the framework does.
    adam->updateWeights(w, Optional<Tensor>(), batchSize);

    adam->getGradientUpdateStream().synchronize();

    // ---- CPU expected ----
    const half LS = Loss::getLossScalingFactor();
    const half invLS = 1.0f / (float)LS;

    half m_exp[4], v_exp[4], du_exp[4], w_exp[4];

    half zero(0.0f);
    half one(1.0f);
    const half invBatch = one / half((float)batchSize);

    // Bias correction for t=1
    const half inv_1mb1 = one / (one - beta1);
    const half inv_1mb2 = one / (one - beta2);

    for (int i = 0; i < 4; ++i) {
        half m1 = beta1 * zero + (one - beta1) * gh[i];
        half v1 = beta2 * zero + (one - beta2) * (gh[i] * gh[i]);

        half mhat = m1 * inv_1mb1;
        half vhat = v1 * inv_1mb2;

        half delta = -lr0 * mhat / ((half)std::sqrt((float)vhat) + eps_adam);

        half w1 = w0[i] + delta * invBatch * invLS;

        m_exp[i] = m1;
        v_exp[i] = v1;
        du_exp[i] = delta;  // what weightsUpdate should hold (pre-loss-scale)
        w_exp[i] = w1;
    }

    // ---- Read back ----
    Tensor w_rb = w.clone(cpu);
    Tensor m_rb = m.clone(cpu);
    Tensor v_rb = v.clone(cpu);
    Tensor du_rb = adam->getWeightsUpdate().clone(cpu);  // if you have getter; otherwise use internal weightsUpdate

    w_rb.copyFromAsync(w, adam->getGradientUpdateStream());
    m_rb.copyFromAsync(m, adam->getGradientUpdateStream());
    v_rb.copyFromAsync(v, adam->getGradientUpdateStream());
    du_rb.copyFromAsync(adam->getWeightsUpdate(), adam->getGradientUpdateStream());
    adam->getGradientUpdateStream().synchronize();

    half *w_gpu = w_rb.getMemPtr<half>();
    half *m_gpu = m_rb.getMemPtr<half>();
    half *v_gpu = v_rb.getMemPtr<half>();
    half *du_gpu = du_rb.getMemPtr<half>();

    const float eps = 5e-3f;

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR((float)m_exp[i], (float)m_gpu[i], eps) << "m i=" << i;
        EXPECT_NEAR((float)v_exp[i], (float)v_gpu[i], eps) << "v i=" << i;
        EXPECT_NEAR((float)du_exp[i], (float)du_gpu[i], eps) << "update i=" << i;
        EXPECT_NEAR((float)w_exp[i], (float)w_gpu[i], eps) << "w i=" << i;
    }
}

TEST(AdamTest, Adam_TwoStep_FromInjectedGradient_CarriesMomentsAndBiasCorrection) {
    TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 2;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = false;

    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpu, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpu));
    LayerTestHelper::connectNetwork(layers);

    const half lr0 = half(0.1f);
    const half beta1 = half(0.9f);
    const half beta2 = half(0.999f);
    const half eps_adam = half(1e-8f);

    auto adam = std::make_shared<Adam>(fc, lr0, beta1, beta2, eps_adam);
    adam->testSetDataType(TensorDescriptor::DataType::FP16);
    adam->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(adam));

    LayerTestHelper::initializeNetwork(layers);
    Stream stream = layers.front()->getStream();

    // Tensors
    Tensor w = fc->getWeights();
    Tensor g = adam->getWeightsGradient();
    Tensor m = adam->getM();
    Tensor v = adam->getV();

    // Init weights W0 = [[1,2],[3,4]]
    Tensor w_h = w.clone(cpu);
    half *w0 = w_h.getMemPtr<half>();
    w0[0] = half(1.0f);
    w0[1] = half(2.0f);
    w0[2] = half(3.0f);
    w0[3] = half(4.0f);
    w.copyFromAsync(w_h, stream);

    // Zero moments
    m.memsetAsync(stream, 0);
    v.memsetAsync(stream, 0);

    stream.synchronize();

    // CPU constants
    const half one(1.0f);
    const half zero(0.0f);

    const half LS = half((float)Loss::getLossScalingFactor());
    const half invLS = one / LS;
    const half invBatch = one / half((float)batchSize);

    // use effective epsilon after compile (FP16 clamp)
    const half eps_eff = half(adam->getEpsilon());

    auto hpow = [](half a, int p) -> half {
        // small integer power, computed in float then cast to half (matches typical impl)
        float af = (float)a;
        float r = 1.0f;
        for (int i = 0; i < p; ++i)
            r *= af;
        return half(r);
    };

    // Expected state holders (FP16)
    half m_exp[4] = {zero, zero, zero, zero};
    half v_exp[4] = {zero, zero, zero, zero};
    half w_exp[4] = {w0[0], w0[1], w0[2], w0[3]};

    // Helper: inject a 2x2 gradient, step Adam, apply update
    auto doStep = [&](half g00, half g01, half g10, half g11) {
        Tensor g_h = g.clone(cpu);
        half *gh = g_h.getMemPtr<half>();
        gh[0] = g00;
        gh[1] = g01;
        gh[2] = g10;
        gh[3] = g11;
        g.copyFromAsync(g_h, stream);
        stream.synchronize();

        adam->stepFromPrecomputedGradient(/*accumulateValues=*/false);
        adam->updateWeights(w, Optional<Tensor>(), batchSize);
        adam->getGradientUpdateStream().synchronize();
    };

    // -----------------------
    // Step 1: G1 = [[10,20],[30,40]]
    // -----------------------
    doStep(half(10.0f), half(20.0f), half(30.0f), half(40.0f));

    // CPU expected after step 1 (t=1)
    const half beta1_t1 = hpow(beta1, 1);
    const half beta2_t1 = hpow(beta2, 1);
    const half inv_1mb1_t1 = one / (one - beta1_t1);
    const half inv_1mb2_t1 = one / (one - beta2_t1);

    const half g1[4] = {half(10.0f), half(20.0f), half(30.0f), half(40.0f)};
    half du1_exp[4];

    for (int i = 0; i < 4; ++i) {
        // moments
        m_exp[i] = beta1 * m_exp[i] + (one - beta1) * g1[i];
        v_exp[i] = beta2 * v_exp[i] + (one - beta2) * (g1[i] * g1[i]);

        // bias-corrected
        half mhat = m_exp[i] * inv_1mb1_t1;
        half vhat = v_exp[i] * inv_1mb2_t1;

        // update (weightsUpdate)
        half du = -lr0 * mhat / (half(std::sqrt((float)vhat)) + eps_eff);
        du1_exp[i] = du;

        // apply through base optimizer: invBatch and invLS
        w_exp[i] = w_exp[i] + du * invBatch * invLS;
    }

    // Read back GPU after step 1
    Tensor w1_rb = w.clone(cpu);
    Tensor m1_rb = m.clone(cpu);
    Tensor v1_rb = v.clone(cpu);
    Tensor du1_rb = adam->getWeightsUpdate().clone(cpu);

    w1_rb.copyFromAsync(w, adam->getGradientUpdateStream());
    m1_rb.copyFromAsync(m, adam->getGradientUpdateStream());
    v1_rb.copyFromAsync(v, adam->getGradientUpdateStream());
    du1_rb.copyFromAsync(adam->getWeightsUpdate(), adam->getGradientUpdateStream());
    adam->getGradientUpdateStream().synchronize();

    half *w1_gpu = w1_rb.getMemPtr<half>();
    half *m1_gpu = m1_rb.getMemPtr<half>();
    half *v1_gpu = v1_rb.getMemPtr<half>();
    half *du1_gpu = du1_rb.getMemPtr<half>();

    const float eps = 5e-3f;
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR((float)m_exp[i], (float)m1_gpu[i], eps) << "step1 m i=" << i;
        EXPECT_NEAR((float)v_exp[i], (float)v1_gpu[i], eps) << "step1 v i=" << i;
        EXPECT_NEAR((float)du1_exp[i], (float)du1_gpu[i], eps) << "step1 update i=" << i;
        EXPECT_NEAR((float)w_exp[i], (float)w1_gpu[i], eps) << "step1 w i=" << i;
    }

    // -----------------------
    // Step 2: pick a different gradient to ensure carry matters
    // G2 = [[1,2],[3,4]]
    // -----------------------
    doStep(half(1.0f), half(2.0f), half(3.0f), half(4.0f));

    // CPU expected after step 2 (t=2)
    const half beta1_t2 = hpow(beta1, 2);
    const half beta2_t2 = hpow(beta2, 2);
    const half inv_1mb1_t2 = one / (one - beta1_t2);
    const half inv_1mb2_t2 = one / (one - beta2_t2);

    const half g2[4] = {half(1.0f), half(2.0f), half(3.0f), half(4.0f)};
    half du2_exp[4];

    for (int i = 0; i < 4; ++i) {
        m_exp[i] = beta1 * m_exp[i] + (one - beta1) * g2[i];
        v_exp[i] = beta2 * v_exp[i] + (one - beta2) * (g2[i] * g2[i]);

        half mhat = m_exp[i] * inv_1mb1_t2;
        half vhat = v_exp[i] * inv_1mb2_t2;

        half du = -lr0 * mhat / (half(std::sqrt((float)vhat)) + eps_eff);
        du2_exp[i] = du;

        w_exp[i] = w_exp[i] + du * invBatch * invLS;
    }

    // Read back GPU after step 2
    Tensor w2_rb = w.clone(cpu);
    Tensor m2_rb = m.clone(cpu);
    Tensor v2_rb = v.clone(cpu);
    Tensor du2_rb = adam->getWeightsUpdate().clone(cpu);

    w2_rb.copyFromAsync(w, adam->getGradientUpdateStream());
    m2_rb.copyFromAsync(m, adam->getGradientUpdateStream());
    v2_rb.copyFromAsync(v, adam->getGradientUpdateStream());
    du2_rb.copyFromAsync(adam->getWeightsUpdate(), adam->getGradientUpdateStream());
    adam->getGradientUpdateStream().synchronize();

    half *w2_gpu = w2_rb.getMemPtr<half>();
    half *m2_gpu = m2_rb.getMemPtr<half>();
    half *v2_gpu = v2_rb.getMemPtr<half>();
    half *du2_gpu = du2_rb.getMemPtr<half>();

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR((float)m_exp[i], (float)m2_gpu[i], eps) << "step2 m i=" << i;
        EXPECT_NEAR((float)v_exp[i], (float)v2_gpu[i], eps) << "step2 v i=" << i;
        EXPECT_NEAR((float)du2_exp[i], (float)du2_gpu[i], eps) << "step2 update i=" << i;
        EXPECT_NEAR((float)w_exp[i], (float)w2_gpu[i], eps) << "step2 w i=" << i;
    }
}

TEST(AdamTest, Adam_Integrated_ForwardBackward_SingleStep_IsCorrect) {
    TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 2;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = false;

    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpu, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpu));
    LayerTestHelper::connectNetwork(layers);

    const half lr0 = half(0.1f);
    const half beta1 = half(0.9f);
    const half beta2 = half(0.999f);
    const half eps_adam = half(1e-8f);

    auto adam = std::make_shared<Adam>(fc, lr0, beta1, beta2, eps_adam);
    adam->testSetDataType(TensorDescriptor::DataType::FP16);
    adam->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(adam));

    LayerTestHelper::initializeNetwork(layers);
    Stream stream = layers.front()->getStream();

    // Grab tensors from the real compute path
    Tensor x = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureInputs());
    Tensor y = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureOutputs());
    Tensor dy = MultiConnectionLayer::getFirstPresentTensor(fc->getErrorInputs());

    Tensor w = fc->getWeights();

    Tensor g = adam->getWeightsGradient();
    Tensor m = adam->getM();
    Tensor v = adam->getV();
    Tensor du = adam->getWeightsUpdate();

    // Deterministic X: [[1,2],[3,4]]
    {
        Tensor x_h = x.clone(cpu);
        half *xp = x_h.getMemPtr<half>();
        xp[0] = half(1.0f);
        xp[1] = half(2.0f);
        xp[2] = half(3.0f);
        xp[3] = half(4.0f);
        x.copyFromAsync(x_h, stream);
    }

    // Deterministic dY: [[1,0],[0,1]]
    {
        Tensor dy_h = dy.clone(cpu);
        half *dyp = dy_h.getMemPtr<half>();
        dyp[0] = half(1.0f);
        dyp[1] = half(0.0f);
        dyp[2] = half(0.0f);
        dyp[3] = half(1.0f);
        dy.copyFromAsync(dy_h, stream);
    }

    // Deterministic W0: [[1,2],[3,4]]
    Tensor w0_h = w.clone(cpu);
    half *w0 = w0_h.getMemPtr<half>();
    w0[0] = half(1.0f);
    w0[1] = half(2.0f);
    w0[2] = half(3.0f);
    w0[3] = half(4.0f);
    w.copyFromAsync(w0_h, stream);

    // Zero moments (so the step is unambiguous)
    m.memsetAsync(stream, 0);
    v.memsetAsync(stream, 0);

    stream.synchronize();

    // --------------------
    // Act: forward/backward
    // --------------------
    fc->forward(x, /*inference=*/false);
    fc->backward(dy);

    adam->getGradientUpdateStream().synchronize();

    // --------------------
    // CPU expected (FP16)
    // --------------------
    const half one(1.0f), zero(0.0f);

    const half LS = half((float)Loss::getLossScalingFactor());
    const half invLS = one / LS;
    const half invBatch = one / half((float)batchSize);

    const half eps_eff = half((float)adam->getEpsilon());  // post-compile clamp

    // Expected gradient: dW = X^T * dY
    // X = [[1,2],[3,4]], X^T = [[1,3],[2,4]]
    // dY = [[1,0],[0,1]]
    // dW = [[1,3],[2,4]]
    const half dW_exp[4] = {half(1.0f), half(3.0f), half(2.0f), half(4.0f)};

    // Adam step t=1
    const half inv_1mb1 = one / (one - beta1);
    const half inv_1mb2 = one / (one - beta2);

    half m_exp[4], v_exp[4], du_exp[4], w_exp[4];
    for (int i = 0; i < 4; ++i) {
        half m1 = beta1 * zero + (one - beta1) * dW_exp[i];
        half v1 = beta2 * zero + (one - beta2) * (dW_exp[i] * dW_exp[i]);

        half mhat = m1 * inv_1mb1;
        half vhat = v1 * inv_1mb2;

        half delta = -lr0 * mhat / (half(std::sqrt((float)vhat)) + eps_eff);

        // Base optimizer applies update: invBatch * invLS
        half w1 = w0[i] + delta * invBatch * invLS;

        m_exp[i] = m1;
        v_exp[i] = v1;
        du_exp[i] = delta;
        w_exp[i] = w1;
    }

    // --------------------
    // Read back GPU tensors
    // --------------------
    Tensor w_rb = w.clone(cpu);
    Tensor g_rb = g.clone(cpu);
    Tensor m_rb = m.clone(cpu);
    Tensor v_rb = v.clone(cpu);
    Tensor du_rb = du.clone(cpu);

    w_rb.copyFromAsync(w, adam->getGradientUpdateStream());
    g_rb.copyFromAsync(g, adam->getGradientUpdateStream());
    m_rb.copyFromAsync(m, adam->getGradientUpdateStream());
    v_rb.copyFromAsync(v, adam->getGradientUpdateStream());
    du_rb.copyFromAsync(du, adam->getGradientUpdateStream());
    adam->getGradientUpdateStream().synchronize();

    half *w_gpu = w_rb.getMemPtr<half>();
    half *g_gpu = g_rb.getMemPtr<half>();
    half *m_gpu = m_rb.getMemPtr<half>();
    half *v_gpu = v_rb.getMemPtr<half>();
    half *du_gpu = du_rb.getMemPtr<half>();

    const float eps = 5e-3f;

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR((float)dW_exp[i], (float)g_gpu[i], eps) << "dW i=" << i;
        EXPECT_NEAR((float)m_exp[i], (float)m_gpu[i], eps) << "m i=" << i;
        EXPECT_NEAR((float)v_exp[i], (float)v_gpu[i], eps) << "v i=" << i;
        EXPECT_NEAR((float)du_exp[i], (float)du_gpu[i], eps) << "update i=" << i;
        EXPECT_NEAR((float)w_exp[i], (float)w_gpu[i], eps) << "w i=" << i;
    }
}

TEST(AdamTest, Adam_Integrated_ForwardBackward_WithBias_Batch3_SingleStep_IsCorrect) {
    TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 3;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = true;

    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpu, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpu));
    LayerTestHelper::connectNetwork(layers);

    const half lr0 = half(0.1f);
    const half beta1 = half(0.9f);
    const half beta2 = half(0.999f);
    const half eps_adam = half(1e-8f);

    auto adam = std::make_shared<Adam>(fc, lr0, beta1, beta2, eps_adam);
    adam->testSetDataType(TensorDescriptor::DataType::FP16);
    adam->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(adam));

    LayerTestHelper::initializeNetwork(layers);
    Stream stream = layers.front()->getStream();

    // Network tensors
    Tensor x = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureInputs());
    Tensor y = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureOutputs());
    Tensor dy = MultiConnectionLayer::getFirstPresentTensor(fc->getErrorInputs());

    Tensor w = fc->getWeights();
    Optional<Tensor> bOpt = fc->getBiases();
    ASSERT_TRUE(bOpt.isPresent());
    Tensor b = bOpt.get();

    // Adam internals
    Tensor gW = adam->getWeightsGradient();
    Tensor mW = adam->getM();
    Tensor vW = adam->getV();
    Tensor duW = adam->getWeightsUpdate();

    Optional<Tensor> gBOpt = adam->getBiasesGradient();
    Optional<Tensor> mBOpt = adam->getMBias();
    Optional<Tensor> vBOpt = adam->getVBias();
    Optional<Tensor> duBOpt = adam->getBiasesUpdate();

    ASSERT_TRUE(gBOpt.isPresent());
    ASSERT_TRUE(mBOpt.isPresent());
    ASSERT_TRUE(vBOpt.isPresent());
    ASSERT_TRUE(duBOpt.isPresent());

    Tensor gB = gBOpt.get();
    Tensor mB = mBOpt.get();
    Tensor vB = vBOpt.get();
    Tensor duB = duBOpt.get();

    // -------------------------
    // Deterministic inputs
    // -------------------------
    // X (3x2):
    // [ [1,2],
    //   [3,4],
    //   [5,6] ]
    {
        Tensor x_h = x.clone(cpu);
        half *xp = x_h.getMemPtr<half>();
        xp[0] = half(1);
        xp[1] = half(2);
        xp[2] = half(3);
        xp[3] = half(4);
        xp[4] = half(5);
        xp[5] = half(6);
        x.copyFromAsync(x_h, stream);
    }

    // dY (3x2):
    // [ [1,0],
    //   [0,1],
    //   [1,1] ]
    {
        Tensor dy_h = dy.clone(cpu);
        half *dyp = dy_h.getMemPtr<half>();
        dyp[0] = half(1);
        dyp[1] = half(0);
        dyp[2] = half(0);
        dyp[3] = half(1);
        dyp[4] = half(1);
        dyp[5] = half(1);
        dy.copyFromAsync(dy_h, stream);
    }

    // W0 (2x2):
    // [ [1,2],
    //   [3,4] ]
    Tensor w0_h = w.clone(cpu);
    half *w0 = w0_h.getMemPtr<half>();
    w0[0] = half(1);
    w0[1] = half(2);
    w0[2] = half(3);
    w0[3] = half(4);
    w.copyFromAsync(w0_h, stream);

    // b0 (2): [5,6]
    Tensor b0_h = b.clone(cpu);
    half *b0 = b0_h.getMemPtr<half>();
    b0[0] = half(5);
    b0[1] = half(6);
    b.copyFromAsync(b0_h, stream);

    // Zero moments
    mW.memsetAsync(stream, 0);
    vW.memsetAsync(stream, 0);
    mB.memsetAsync(stream, 0);
    vB.memsetAsync(stream, 0);

    stream.synchronize();

    // -------------------------
    // Act: forward/backward
    // -------------------------
    fc->forward(x, /*inference=*/false);
    fc->backward(dy);
    adam->getGradientUpdateStream().synchronize();

    // -------------------------
    // CPU expected (FP16)
    // -------------------------
    const half one(1.0f), zero(0.0f);

    const half LS = half((float)Loss::getLossScalingFactor());
    const half invLS = one / LS;
    const half invBatch = one / half((float)batchSize);

    const half eps_eff = half((float)adam->getEpsilon());

    // dW = X^T * dY
    // X^T is 2x3:
    // [ [1,3,5],
    //   [2,4,6] ]
    // dY is 3x2:
    // [ [1,0],
    //   [0,1],
    //   [1,1] ]
    //
    // dW = 2x2:
    // [ [1*1+3*0+5*1 , 1*0+3*1+5*1] ] = [6, 8]
    // [ [2*1+4*0+6*1 , 2*0+4*1+6*1] ] = [8, 10]
    const half dW_exp[4] = {half(6), half(8), half(8), half(10)};

    // db = sum over batch of dY rows:
    // col0: 1 + 0 + 1 = 2
    // col1: 0 + 1 + 1 = 2
    const half db_exp[2] = {half(2), half(2)};

    // Adam t=1 bias correction
    const half inv_1mb1 = one / (one - beta1);
    const half inv_1mb2 = one / (one - beta2);

    half mW_exp[4], vW_exp[4], duW_exp[4], w_exp[4];
    half mB_exp[2], vB_exp[2], duB_exp[2], b_exp[2];

    for (int i = 0; i < 4; ++i) {
        half m1 = beta1 * zero + (one - beta1) * dW_exp[i];
        half v1 = beta2 * zero + (one - beta2) * (dW_exp[i] * dW_exp[i]);

        half mhat = m1 * inv_1mb1;
        half vhat = v1 * inv_1mb2;

        half du = -lr0 * mhat / (half(std::sqrt((float)vhat)) + eps_eff);

        // base apply: invBatch and invLS
        half w1 = w0[i] + du * invBatch * invLS;

        mW_exp[i] = m1;
        vW_exp[i] = v1;
        duW_exp[i] = du;
        w_exp[i] = w1;
    }

    for (int i = 0; i < 2; ++i) {
        half m1 = beta1 * zero + (one - beta1) * db_exp[i];
        half v1 = beta2 * zero + (one - beta2) * (db_exp[i] * db_exp[i]);

        half mhat = m1 * inv_1mb1;
        half vhat = v1 * inv_1mb2;

        half du = -lr0 * mhat / (half(std::sqrt((float)vhat)) + eps_eff);

        half b1 = b0[i] + du * invBatch * invLS;

        mB_exp[i] = m1;
        vB_exp[i] = v1;
        duB_exp[i] = du;
        b_exp[i] = b1;
    }

    // -------------------------
    // Read back GPU
    // -------------------------
    Tensor w_rb = w.clone(cpu);
    Tensor b_rb = b.clone(cpu);

    Tensor gW_rb = gW.clone(cpu);
    Tensor mW_rb = mW.clone(cpu);
    Tensor vW_rb = vW.clone(cpu);
    Tensor duW_rb = duW.clone(cpu);

    Tensor gB_rb = gB.clone(cpu);
    Tensor mB_rb = mB.clone(cpu);
    Tensor vB_rb = vB.clone(cpu);
    Tensor duB_rb = duB.clone(cpu);

    w_rb.copyFromAsync(w, adam->getGradientUpdateStream());
    b_rb.copyFromAsync(b, adam->getGradientUpdateStream());

    gW_rb.copyFromAsync(gW, adam->getGradientUpdateStream());
    mW_rb.copyFromAsync(mW, adam->getGradientUpdateStream());
    vW_rb.copyFromAsync(vW, adam->getGradientUpdateStream());
    duW_rb.copyFromAsync(duW, adam->getGradientUpdateStream());

    gB_rb.copyFromAsync(gB, adam->getGradientUpdateStream());
    mB_rb.copyFromAsync(mB, adam->getGradientUpdateStream());
    vB_rb.copyFromAsync(vB, adam->getGradientUpdateStream());
    duB_rb.copyFromAsync(duB, adam->getGradientUpdateStream());

    adam->getGradientUpdateStream().synchronize();

    half *w_gpu = w_rb.getMemPtr<half>();
    half *b_gpu = b_rb.getMemPtr<half>();

    half *gW_gpu = gW_rb.getMemPtr<half>();
    half *mW_gpu = mW_rb.getMemPtr<half>();
    half *vW_gpu = vW_rb.getMemPtr<half>();
    half *duW_gpu = duW_rb.getMemPtr<half>();

    half *gB_gpu = gB_rb.getMemPtr<half>();
    half *mB_gpu = mB_rb.getMemPtr<half>();
    half *vB_gpu = vB_rb.getMemPtr<half>();
    half *duB_gpu = duB_rb.getMemPtr<half>();

    const float eps = 5e-3f;

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR((float)dW_exp[i], (float)gW_gpu[i], eps) << "dW i=" << i;
        EXPECT_NEAR((float)mW_exp[i], (float)mW_gpu[i], eps) << "mW i=" << i;
        EXPECT_NEAR((float)vW_exp[i], (float)vW_gpu[i], eps) << "vW i=" << i;
        EXPECT_NEAR((float)duW_exp[i], (float)duW_gpu[i], eps) << "duW i=" << i;
        EXPECT_NEAR((float)w_exp[i], (float)w_gpu[i], eps) << "w i=" << i;
    }

    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR((float)db_exp[i], (float)gB_gpu[i], eps) << "db i=" << i;
        EXPECT_NEAR((float)mB_exp[i], (float)mB_gpu[i], eps) << "mB i=" << i;
        EXPECT_NEAR((float)vB_exp[i], (float)vB_gpu[i], eps) << "vB i=" << i;
        EXPECT_NEAR((float)duB_exp[i], (float)duB_gpu[i], eps) << "duB i=" << i;
        EXPECT_NEAR((float)b_exp[i], (float)b_gpu[i], eps) << "b i=" << i;
    }
}

TEST(AdamTest, Adam_Integrated_TwoIterations_WithBias_Batch3_CarriesMomentsAndBiasCorrection) {
    TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 3;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = true;

    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpu, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpu));
    LayerTestHelper::connectNetwork(layers);

    const half lr0 = half(0.1f);
    const half beta1 = half(0.9f);
    const half beta2 = half(0.999f);
    const half eps_adam = half(1e-8f);

    auto adam = std::make_shared<Adam>(fc, lr0, beta1, beta2, eps_adam);
    adam->testSetDataType(TensorDescriptor::DataType::FP16);
    adam->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(adam));

    LayerTestHelper::initializeNetwork(layers);
    Stream stream = layers.front()->getStream();

    // Network tensors
    Tensor x = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureInputs());
    Tensor y = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureOutputs());
    Tensor dy = MultiConnectionLayer::getFirstPresentTensor(fc->getErrorInputs());

    Tensor w = fc->getWeights();
    Optional<Tensor> bOpt = fc->getBiases();
    ASSERT_TRUE(bOpt.isPresent());
    Tensor b = bOpt.get();

    // Adam tensors
    Tensor gW = adam->getWeightsGradient();
    Tensor mW = adam->getM();
    Tensor vW = adam->getV();
    Tensor duW = adam->getWeightsUpdate();

    Optional<Tensor> gBOpt = adam->getBiasesGradient();
    Optional<Tensor> mBOpt = adam->getMBias();
    Optional<Tensor> vBOpt = adam->getVBias();
    Optional<Tensor> duBOpt = adam->getBiasesUpdate();

    ASSERT_TRUE(gBOpt.isPresent());
    ASSERT_TRUE(mBOpt.isPresent());
    ASSERT_TRUE(vBOpt.isPresent());
    ASSERT_TRUE(duBOpt.isPresent());

    Tensor gB = gBOpt.get();
    Tensor mB = mBOpt.get();
    Tensor vB = vBOpt.get();
    Tensor duB = duBOpt.get();

    // Init W0 and b0
    Tensor w0_h = w.clone(cpu);
    half *w0 = w0_h.getMemPtr<half>();
    w0[0] = half(1);
    w0[1] = half(2);
    w0[2] = half(3);
    w0[3] = half(4);
    w.copyFromAsync(w0_h, stream);

    Tensor b0_h = b.clone(cpu);
    half *b0 = b0_h.getMemPtr<half>();
    b0[0] = half(5);
    b0[1] = half(6);
    b.copyFromAsync(b0_h, stream);

    // Zero moments
    mW.memsetAsync(stream, 0);
    vW.memsetAsync(stream, 0);
    mB.memsetAsync(stream, 0);
    vB.memsetAsync(stream, 0);

    stream.synchronize();

    // CPU constants
    const half one(1.0f), zero(0.0f);
    const half LS = half((float)Loss::getLossScalingFactor());
    const half invLS = one / LS;
    const half invBatch = one / half((float)batchSize);
    const half eps_eff = half((float)adam->getEpsilon());

    // FP16 power helper (small exponents)
    auto hpow = [](half a, int p) -> half {
        float af = (float)a;
        float r = 1.0f;
        for (int i = 0; i < p; ++i)
            r *= af;
        return half(r);
    };

    // Helper to set X (3x2) and dY (3x2)
    auto set3x2 = [&](Tensor t, half a00, half a01, half a10, half a11, half a20, half a21) {
        Tensor th = t.clone(cpu);
        half *p = th.getMemPtr<half>();
        p[0] = a00;
        p[1] = a01;
        p[2] = a10;
        p[3] = a11;
        p[4] = a20;
        p[5] = a21;
        t.copyFromAsync(th, stream);
    };

    // Helper: compute dW = X^T * dY for 3x2
    auto dW_3x2 = [](const half X[6], const half dY[6], half dW[4]) {
        // X: rows r0,r1,r2 each 2
        // dY: rows each 2
        // dW(2x2):
        // [ sum_r X[r,0]*dY[r,0], sum_r X[r,0]*dY[r,1],
        //   sum_r X[r,1]*dY[r,0], sum_r X[r,1]*dY[r,1] ]
        dW[0] = X[0] * dY[0] + X[2] * dY[2] + X[4] * dY[4];
        dW[1] = X[0] * dY[1] + X[2] * dY[3] + X[4] * dY[5];
        dW[2] = X[1] * dY[0] + X[3] * dY[2] + X[5] * dY[4];
        dW[3] = X[1] * dY[1] + X[3] * dY[3] + X[5] * dY[5];
    };

    auto db_3x2 = [](const half dY[6], half db[2]) {
        db[0] = dY[0] + dY[2] + dY[4];
        db[1] = dY[1] + dY[3] + dY[5];
    };

    // CPU state
    half w_exp[4] = {w0[0], w0[1], w0[2], w0[3]};
    half b_exp[2] = {b0[0], b0[1]};
    half mW_exp[4] = {zero, zero, zero, zero};
    half vW_exp[4] = {zero, zero, zero, zero};
    half mB_exp[2] = {zero, zero};
    half vB_exp[2] = {zero, zero};

    auto checkReadback = [&](int step, const half dW_exp[4], const half db_exp[2], const half duW_exp[4], const half duB_exp[2]) {
        Tensor w_rb = w.clone(cpu);
        Tensor b_rb = b.clone(cpu);

        Tensor gW_rb = gW.clone(cpu);
        Tensor mW_rb = mW.clone(cpu);
        Tensor vW_rb = vW.clone(cpu);
        Tensor duW_rb = duW.clone(cpu);

        Tensor gB_rb = gB.clone(cpu);
        Tensor mB_rb = mB.clone(cpu);
        Tensor vB_rb = vB.clone(cpu);
        Tensor duB_rb = duB.clone(cpu);

        w_rb.copyFromAsync(w, adam->getGradientUpdateStream());
        b_rb.copyFromAsync(b, adam->getGradientUpdateStream());
        gW_rb.copyFromAsync(gW, adam->getGradientUpdateStream());
        mW_rb.copyFromAsync(mW, adam->getGradientUpdateStream());
        vW_rb.copyFromAsync(vW, adam->getGradientUpdateStream());
        duW_rb.copyFromAsync(duW, adam->getGradientUpdateStream());
        gB_rb.copyFromAsync(gB, adam->getGradientUpdateStream());
        mB_rb.copyFromAsync(mB, adam->getGradientUpdateStream());
        vB_rb.copyFromAsync(vB, adam->getGradientUpdateStream());
        duB_rb.copyFromAsync(duB, adam->getGradientUpdateStream());
        adam->getGradientUpdateStream().synchronize();

        half *w_gpu = w_rb.getMemPtr<half>();
        half *b_gpu = b_rb.getMemPtr<half>();

        half *gW_gpu = gW_rb.getMemPtr<half>();
        half *mW_gpu = mW_rb.getMemPtr<half>();
        half *vW_gpu = vW_rb.getMemPtr<half>();
        half *duW_gpu = duW_rb.getMemPtr<half>();

        half *gB_gpu = gB_rb.getMemPtr<half>();
        half *mB_gpu = mB_rb.getMemPtr<half>();
        half *vB_gpu = vB_rb.getMemPtr<half>();
        half *duB_gpu = duB_rb.getMemPtr<half>();

        const float eps = 5e-3f;

        for (int i = 0; i < 4; ++i) {
            EXPECT_NEAR((float)dW_exp[i], (float)gW_gpu[i], eps) << "step" << step << " dW i=" << i;
            EXPECT_NEAR((float)mW_exp[i], (float)mW_gpu[i], eps) << "step" << step << " mW i=" << i;
            EXPECT_NEAR((float)vW_exp[i], (float)vW_gpu[i], eps) << "step" << step << " vW i=" << i;
            EXPECT_NEAR((float)duW_exp[i], (float)duW_gpu[i], eps) << "step" << step << " duW i=" << i;
            EXPECT_NEAR((float)w_exp[i], (float)w_gpu[i], eps) << "step" << step << " w i=" << i;
        }

        for (int i = 0; i < 2; ++i) {
            EXPECT_NEAR((float)db_exp[i], (float)gB_gpu[i], eps) << "step" << step << " db i=" << i;
            EXPECT_NEAR((float)mB_exp[i], (float)mB_gpu[i], eps) << "step" << step << " mB i=" << i;
            EXPECT_NEAR((float)vB_exp[i], (float)vB_gpu[i], eps) << "step" << step << " vB i=" << i;
            EXPECT_NEAR((float)duB_exp[i], (float)duB_gpu[i], eps) << "step" << step << " duB i=" << i;
            EXPECT_NEAR((float)b_exp[i], (float)b_gpu[i], eps) << "step" << step << " b i=" << i;
        }
    };

    // =========================
    // Iteration 1
    // X1 = [[1,2],[3,4],[5,6]]
    // dY1 = [[1,0],[0,1],[1,1]]
    // =========================
    set3x2(x, half(1), half(2), half(3), half(4), half(5), half(6));
    set3x2(dy, half(1), half(0), half(0), half(1), half(1), half(1));
    stream.synchronize();

    fc->forward(x, /*inference=*/false);
    fc->backward(dy);
    adam->getGradientUpdateStream().synchronize();

    // CPU compute dW1/db1
    const half X1[6] = {half(1), half(2), half(3), half(4), half(5), half(6)};
    const half dY1[6] = {half(1), half(0), half(0), half(1), half(1), half(1)};
    half dW1[4], db1[2];
    dW_3x2(X1, dY1, dW1);
    db_3x2(dY1, db1);

    // Adam update t=1
    const half inv_1mb1_t1 = one / (one - hpow(beta1, 1));
    const half inv_1mb2_t1 = one / (one - hpow(beta2, 1));

    half duW1[4], duB1[2];
    for (int i = 0; i < 4; ++i) {
        mW_exp[i] = beta1 * mW_exp[i] + (one - beta1) * dW1[i];
        vW_exp[i] = beta2 * vW_exp[i] + (one - beta2) * (dW1[i] * dW1[i]);

        half mhat = mW_exp[i] * inv_1mb1_t1;
        half vhat = vW_exp[i] * inv_1mb2_t1;

        half du = -lr0 * mhat / (half(std::sqrt((float)vhat)) + eps_eff);
        duW1[i] = du;
        w_exp[i] = w_exp[i] + du * invBatch * invLS;
    }
    for (int i = 0; i < 2; ++i) {
        mB_exp[i] = beta1 * mB_exp[i] + (one - beta1) * db1[i];
        vB_exp[i] = beta2 * vB_exp[i] + (one - beta2) * (db1[i] * db1[i]);

        half mhat = mB_exp[i] * inv_1mb1_t1;
        half vhat = vB_exp[i] * inv_1mb2_t1;

        half du = -lr0 * mhat / (half(std::sqrt((float)vhat)) + eps_eff);
        duB1[i] = du;
        b_exp[i] = b_exp[i] + du * invBatch * invLS;
    }

    checkReadback(/*step=*/1, dW1, db1, duW1, duB1);

    EXPECT_NEAR(adam->getT(), 1.0f, 0.0f);

    // =========================
    // Iteration 2 (different X2/dY2)
    // X2  = [[2,1],[0,1],[1,0]]
    // dY2 = [[1,1],[0,1],[1,0]]
    // =========================
    set3x2(x, half(2), half(1), half(0), half(1), half(1), half(0));
    set3x2(dy, half(1), half(1), half(0), half(1), half(1), half(0));
    stream.synchronize();

    fc->forward(x, /*inference=*/false);
    fc->backward(dy);
    adam->getGradientUpdateStream().synchronize();

    const half X2[6] = {half(2), half(1), half(0), half(1), half(1), half(0)};
    const half dY2[6] = {half(1), half(1), half(0), half(1), half(1), half(0)};
    half dW2[4], db2[2];
    dW_3x2(X2, dY2, dW2);
    db_3x2(dY2, db2);

    // Adam update t=2 (bias correction changes!)
    const half inv_1mb1_t2 = one / (one - hpow(beta1, 2));
    const half inv_1mb2_t2 = one / (one - hpow(beta2, 2));

    half duW2[4], duB2[2];
    for (int i = 0; i < 4; ++i) {
        mW_exp[i] = beta1 * mW_exp[i] + (one - beta1) * dW2[i];
        vW_exp[i] = beta2 * vW_exp[i] + (one - beta2) * (dW2[i] * dW2[i]);

        half mhat = mW_exp[i] * inv_1mb1_t2;
        half vhat = vW_exp[i] * inv_1mb2_t2;

        half du = -lr0 * mhat / (half(std::sqrt((float)vhat)) + eps_eff);
        duW2[i] = du;
        w_exp[i] = w_exp[i] + du * invBatch * invLS;
    }
    for (int i = 0; i < 2; ++i) {
        mB_exp[i] = beta1 * mB_exp[i] + (one - beta1) * db2[i];
        vB_exp[i] = beta2 * vB_exp[i] + (one - beta2) * (db2[i] * db2[i]);

        half mhat = mB_exp[i] * inv_1mb1_t2;
        half vhat = vB_exp[i] * inv_1mb2_t2;

        half du = -lr0 * mhat / (half(std::sqrt((float)vhat)) + eps_eff);
        duB2[i] = du;
        b_exp[i] = b_exp[i] + du * invBatch * invLS;
    }

    checkReadback(/*step=*/2, dW2, db2, duW2, duB2);

    EXPECT_NEAR(adam->getT(), 2.0f, 0.0f);
}

TEST(AdamTest, Adam_T_AccumulateValues_DoesNotIncrementUntilFinal) {
    TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 2;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = false;

    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpu, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpu));
    LayerTestHelper::connectNetwork(layers);

    const half lr0 = half(0.1f);
    const half beta1 = half(0.9f);
    const half beta2 = half(0.999f);
    const half eps_adam = half(1e-8f);

    auto adam = std::make_shared<Adam>(fc, lr0, beta1, beta2, eps_adam);
    adam->testSetDataType(TensorDescriptor::DataType::FP16);
    adam->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(adam));

    LayerTestHelper::initializeNetwork(layers);
    Stream stream = layers.front()->getStream();

    // Make sure t starts at 0 (you have setT)
    adam->setT(0.0f);

    // Inject some gradient
    Tensor g = adam->getWeightsGradient();
    Tensor g_h = g.clone(cpu);
    half *gh = g_h.getMemPtr<half>();
    gh[0] = half(1);
    gh[1] = half(2);
    gh[2] = half(3);
    gh[3] = half(4);
    g.copyFromAsync(g_h, stream);
    stream.synchronize();

    // accumulateValues=true should NOT increment t
    adam->stepFromPrecomputedGradient(/*accumulateValues=*/true);
    adam->getGradientUpdateStream().synchronize();
    EXPECT_NEAR(adam->getT(), 0.0f, 0.0f);

    // Another accumulateValues=true still should not increment
    adam->stepFromPrecomputedGradient(/*accumulateValues=*/true);
    adam->getGradientUpdateStream().synchronize();
    EXPECT_NEAR(adam->getT(), 0.0f, 0.0f);

    // Now finalize (accumulateValues=false) increments exactly once
    adam->stepFromPrecomputedGradient(/*accumulateValues=*/false);
    adam->getGradientUpdateStream().synchronize();
    EXPECT_NEAR(adam->getT(), 1.0f, 0.0f);

    // Another finalize increments again
    adam->stepFromPrecomputedGradient(/*accumulateValues=*/false);
    adam->getGradientUpdateStream().synchronize();
    EXPECT_NEAR(adam->getT(), 2.0f, 0.0f);
}

// FIXME: When fp32 implemented:
// TEST(AdamTest, Adam_Integrated_TwoIterations_WithBias_Batch3_CarriesMomentsAndBiasCorrection_FP32) {
//     TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
//     TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
//
//     const uint32_t batchSize = 3;
//     const uint32_t inF = 2;
//     const uint32_t outF = 2;
//     const bool hasBias = true;
//
//     std::vector<std::shared_ptr<Layer>> layers;
//     layers.push_back(std::make_shared<NetworkInput>(gpu, TensorDescriptor::DataType::FP32, std::vector<uint64_t>{batchSize, inF}));
//     layers.push_back(std::make_shared<NoOpLayer>());
//     auto fc = std::make_shared<FullyConnected>(outF, hasBias);
//     layers.push_back(fc);
//     layers.push_back(std::make_shared<NoOpLayer>());
//     layers.push_back(std::make_shared<NetworkOutput>(cpu));
//     LayerTestHelper::connectNetwork(layers);
//
//     const float lr0 = 0.1f;
//     const float beta1 = 0.9f;
//     const float beta2 = 0.999f;
//     const float eps_adam = 1e-8f;
//
//     auto adam = std::make_shared<Adam>(fc, lr0, beta1, beta2, eps_adam);
//     adam->testSetDataType(TensorDescriptor::DataType::FP32);
//     adam->compile();
//     fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(adam));
//
//     LayerTestHelper::initializeNetwork(layers);
//     Stream stream = layers.front()->getStream();
//
//     // Network tensors
//     Tensor x = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureInputs());
//     Tensor y = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureOutputs());
//     Tensor dy = MultiConnectionLayer::getFirstPresentTensor(fc->getErrorInputs());
//
//     Tensor w = fc->getWeights();
//     Optional<Tensor> bOpt = fc->getBiases();
//     ASSERT_TRUE(bOpt.isPresent());
//     Tensor b = bOpt.get();
//
//     // Adam tensors
//     Tensor gW = adam->getWeightsGradient();
//     Tensor mW = adam->getM();
//     Tensor vW = adam->getV();
//     Tensor duW = adam->getWeightsUpdate();
//
//     Optional<Tensor> gBOpt = adam->getBiasesGradient();
//     Optional<Tensor> mBOpt = adam->getMBias();
//     Optional<Tensor> vBOpt = adam->getVBias();
//     Optional<Tensor> duBOpt = adam->getBiasesUpdate();
//
//     ASSERT_TRUE(gBOpt.isPresent());
//     ASSERT_TRUE(mBOpt.isPresent());
//     ASSERT_TRUE(vBOpt.isPresent());
//     ASSERT_TRUE(duBOpt.isPresent());
//
//     Tensor gB = gBOpt.get();
//     Tensor mB = mBOpt.get();
//     Tensor vB = vBOpt.get();
//     Tensor duB = duBOpt.get();
//
//     // Init W0 and b0
//     Tensor w0_h = w.clone(cpu);
//     float *w0 = w0_h.getMemPtr<float>();
//     w0[0] = 1.0f;
//     w0[1] = 2.0f;
//     w0[2] = 3.0f;
//     w0[3] = 4.0f;
//     w.copyFromAsync(w0_h, stream);
//
//     Tensor b0_h = b.clone(cpu);
//     float *b0 = b0_h.getMemPtr<float>();
//     b0[0] = 5.0f;
//     b0[1] = 6.0f;
//     b.copyFromAsync(b0_h, stream);
//
//     // Zero moments
//     mW.memsetAsync(stream, 0);
//     vW.memsetAsync(stream, 0);
//     mB.memsetAsync(stream, 0);
//     vB.memsetAsync(stream, 0);
//
//     stream.synchronize();
//
//     // CPU constants
//     const float LS = Loss::getLossScalingFactor();
//     const float invLS = 1.0f / LS;
//     const float invBatch = 1.0f / (float)batchSize;
//     const float eps_eff = adam->getEpsilon();  // post-compile (no FP16 clamp expected here)
//
//     auto fpow_int = [](float a, int p) -> float {
//         float r = 1.0f;
//         for (int i = 0; i < p; ++i)
//             r *= a;
//         return r;
//     };
//
//     // Helper to set X (3x2) and dY (3x2)
//     auto set3x2 = [&](Tensor t, float a00, float a01, float a10, float a11, float a20, float a21) {
//         Tensor th = t.clone(cpu);
//         float *p = th.getMemPtr<float>();
//         p[0] = a00;
//         p[1] = a01;
//         p[2] = a10;
//         p[3] = a11;
//         p[4] = a20;
//         p[5] = a21;
//         t.copyFromAsync(th, stream);
//     };
//
//     // Helper: compute dW = X^T * dY for 3x2
//     auto dW_3x2 = [](const float X[6], const float dY[6], float dW[4]) {
//         dW[0] = X[0] * dY[0] + X[2] * dY[2] + X[4] * dY[4];
//         dW[1] = X[0] * dY[1] + X[2] * dY[3] + X[4] * dY[5];
//         dW[2] = X[1] * dY[0] + X[3] * dY[2] + X[5] * dY[4];
//         dW[3] = X[1] * dY[1] + X[3] * dY[3] + X[5] * dY[5];
//     };
//
//     auto db_3x2 = [](const float dY[6], float db[2]) {
//         db[0] = dY[0] + dY[2] + dY[4];
//         db[1] = dY[1] + dY[3] + dY[5];
//     };
//
//     // CPU state
//     float w_exp[4] = {w0[0], w0[1], w0[2], w0[3]};
//     float b_exp[2] = {b0[0], b0[1]};
//     float mW_exp[4] = {0, 0, 0, 0};
//     float vW_exp[4] = {0, 0, 0, 0};
//     float mB_exp[2] = {0, 0};
//     float vB_exp[2] = {0, 0};
//
//     auto checkReadback =
//         [&](int step, const float dW_exp_step[4], const float db_exp_step[2], const float duW_exp_step[4], const float duB_exp_step[2]) {
//             Tensor w_rb = w.clone(cpu);
//             Tensor b_rb = b.clone(cpu);
//
//             Tensor gW_rb = gW.clone(cpu);
//             Tensor mW_rb = mW.clone(cpu);
//             Tensor vW_rb = vW.clone(cpu);
//             Tensor duW_rb = duW.clone(cpu);
//
//             Tensor gB_rb = gB.clone(cpu);
//             Tensor mB_rb = mB.clone(cpu);
//             Tensor vB_rb = vB.clone(cpu);
//             Tensor duB_rb = duB.clone(cpu);
//
//             w_rb.copyFromAsync(w, adam->getGradientUpdateStream());
//             b_rb.copyFromAsync(b, adam->getGradientUpdateStream());
//             gW_rb.copyFromAsync(gW, adam->getGradientUpdateStream());
//             mW_rb.copyFromAsync(mW, adam->getGradientUpdateStream());
//             vW_rb.copyFromAsync(vW, adam->getGradientUpdateStream());
//             duW_rb.copyFromAsync(duW, adam->getGradientUpdateStream());
//             gB_rb.copyFromAsync(gB, adam->getGradientUpdateStream());
//             mB_rb.copyFromAsync(mB, adam->getGradientUpdateStream());
//             vB_rb.copyFromAsync(vB, adam->getGradientUpdateStream());
//             duB_rb.copyFromAsync(duB, adam->getGradientUpdateStream());
//             adam->getGradientUpdateStream().synchronize();
//
//             float *w_gpu = w_rb.getMemPtr<float>();
//             float *b_gpu = b_rb.getMemPtr<float>();
//
//             float *gW_gpu = gW_rb.getMemPtr<float>();
//             float *mW_gpu = mW_rb.getMemPtr<float>();
//             float *vW_gpu = vW_rb.getMemPtr<float>();
//             float *duW_gpu = duW_rb.getMemPtr<float>();
//
//             float *gB_gpu = gB_rb.getMemPtr<float>();
//             float *mB_gpu = mB_rb.getMemPtr<float>();
//             float *vB_gpu = vB_rb.getMemPtr<float>();
//             float *duB_gpu = duB_rb.getMemPtr<float>();
//
//             const float eps = 1e-6f;
//
//             for (int i = 0; i < 4; ++i) {
//                 EXPECT_NEAR(dW_exp_step[i], gW_gpu[i], eps) << "step" << step << " dW i=" << i;
//                 EXPECT_NEAR(mW_exp[i], mW_gpu[i], eps) << "step" << step << " mW i=" << i;
//                 EXPECT_NEAR(vW_exp[i], vW_gpu[i], eps) << "step" << step << " vW i=" << i;
//                 EXPECT_NEAR(duW_exp_step[i], duW_gpu[i], eps) << "step" << step << " duW i=" << i;
//                 EXPECT_NEAR(w_exp[i], w_gpu[i], eps) << "step" << step << " w i=" << i;
//             }
//
//             for (int i = 0; i < 2; ++i) {
//                 EXPECT_NEAR(db_exp_step[i], gB_gpu[i], eps) << "step" << step << " db i=" << i;
//                 EXPECT_NEAR(mB_exp[i], mB_gpu[i], eps) << "step" << step << " mB i=" << i;
//                 EXPECT_NEAR(vB_exp[i], vB_gpu[i], eps) << "step" << step << " vB i=" << i;
//                 EXPECT_NEAR(duB_exp_step[i], duB_gpu[i], eps) << "step" << step << " duB i=" << i;
//                 EXPECT_NEAR(b_exp[i], b_gpu[i], eps) << "step" << step << " b i=" << i;
//             }
//         };
//
//     // =========================
//     // Iteration 1
//     // X1 = [[1,2],[3,4],[5,6]]
//     // dY1 = [[1,0],[0,1],[1,1]]
//     // =========================
//     set3x2(x, 1, 2, 3, 4, 5, 6);
//     set3x2(dy, 1, 0, 0, 1, 1, 1);
//     stream.synchronize();
//
//     fc->forward(x, /*inference=*/false);
//     fc->backward(dy);
//     adam->getGradientUpdateStream().synchronize();
//
//     const float X1[6] = {1, 2, 3, 4, 5, 6};
//     const float dY1[6] = {1, 0, 0, 1, 1, 1};
//     float dW1[4], db1[2];
//     dW_3x2(X1, dY1, dW1);
//     db_3x2(dY1, db1);
//
//     const float inv_1mb1_t1 = 1.0f / (1.0f - fpow_int(beta1, 1));
//     const float inv_1mb2_t1 = 1.0f / (1.0f - fpow_int(beta2, 1));
//
//     float duW1[4], duB1[2];
//     for (int i = 0; i < 4; ++i) {
//         mW_exp[i] = beta1 * mW_exp[i] + (1.0f - beta1) * dW1[i];
//         vW_exp[i] = beta2 * vW_exp[i] + (1.0f - beta2) * (dW1[i] * dW1[i]);
//
//         float mhat = mW_exp[i] * inv_1mb1_t1;
//         float vhat = vW_exp[i] * inv_1mb2_t1;
//
//         float du = -lr0 * mhat / (std::sqrt(vhat) + eps_eff);
//         duW1[i] = du;
//         w_exp[i] = w_exp[i] + du * invBatch * invLS;
//     }
//     for (int i = 0; i < 2; ++i) {
//         mB_exp[i] = beta1 * mB_exp[i] + (1.0f - beta1) * db1[i];
//         vB_exp[i] = beta2 * vB_exp[i] + (1.0f - beta2) * (db1[i] * db1[i]);
//
//         float mhat = mB_exp[i] * inv_1mb1_t1;
//         float vhat = vB_exp[i] * inv_1mb2_t1;
//
//         float du = -lr0 * mhat / (std::sqrt(vhat) + eps_eff);
//         duB1[i] = du;
//         b_exp[i] = b_exp[i] + du * invBatch * invLS;
//     }
//
//     checkReadback(/*step=*/1, dW1, db1, duW1, duB1);
//
//     // =========================
//     // Iteration 2
//     // X2  = [[2,1],[0,1],[1,0]]
//     // dY2 = [[1,1],[0,1],[1,0]]
//     // =========================
//     set3x2(x, 2, 1, 0, 1, 1, 0);
//     set3x2(dy, 1, 1, 0, 1, 1, 0);
//     stream.synchronize();
//
//     fc->forward(x, /*inference=*/false);
//     fc->backward(dy);
//     adam->getGradientUpdateStream().synchronize();
//
//     const float X2[6] = {2, 1, 0, 1, 1, 0};
//     const float dY2[6] = {1, 1, 0, 1, 1, 0};
//     float dW2[4], db2[2];
//     dW_3x2(X2, dY2, dW2);
//     db_3x2(dY2, db2);
//
//     const float inv_1mb1_t2 = 1.0f / (1.0f - fpow_int(beta1, 2));
//     const float inv_1mb2_t2 = 1.0f / (1.0f - fpow_int(beta2, 2));
//
//     float duW2[4], duB2[2];
//     for (int i = 0; i < 4; ++i) {
//         mW_exp[i] = beta1 * mW_exp[i] + (1.0f - beta1) * dW2[i];
//         vW_exp[i] = beta2 * vW_exp[i] + (1.0f - beta2) * (dW2[i] * dW2[i]);
//
//         float mhat = mW_exp[i] * inv_1mb1_t2;
//         float vhat = vW_exp[i] * inv_1mb2_t2;
//
//         float du = -lr0 * mhat / (std::sqrt(vhat) + eps_eff);
//         duW2[i] = du;
//         w_exp[i] = w_exp[i] + du * invBatch * invLS;
//     }
//     for (int i = 0; i < 2; ++i) {
//         mB_exp[i] = beta1 * mB_exp[i] + (1.0f - beta1) * db2[i];
//         vB_exp[i] = beta2 * vB_exp[i] + (1.0f - beta2) * (db2[i] * db2[i]);
//
//         float mhat = mB_exp[i] * inv_1mb1_t2;
//         float vhat = vB_exp[i] * inv_1mb2_t2;
//
//         float du = -lr0 * mhat / (std::sqrt(vhat) + eps_eff);
//         duB2[i] = du;
//         b_exp[i] = b_exp[i] + du * invBatch * invLS;
//     }
//
//     checkReadback(/*step=*/2, dW2, db2, duW2, duB2);
// }
