
// https://arxiv.org/abs/1409.4842
//
//[type]          [patch size/stride] [output size]  [depth]  [#1×1]  [#3×3 reduce]   [#3×3]  [#5×5 reduce]   [#5×5]  [pool proj] [params]

// convolution     7×7/2               112×112×64      1                                                                           2.7K
// max pool        3×3/2               56×56×64        0
// convolution     3×3/1               56×56×192       2               64              192     112K                                360M
// max pool        3×3/2               28×28×192       0
// inception (3a)                      28×28×256       2       64      96              128     16              32      32          159K
// inception (3b)                      28×28×480       2       128     128             192     32              96      64          380K
// max pool 3×3/2                      14×14×480       0
// inception (4a)                      14×14×512       2       192     96              208     16              48      64          364K
// inception (4b)                      14×14×512       2       160     112             224     24              64      64          437K
// inception (4c)                      14×14×512       2       128     128             256     24              64      64          463K
// inception (4d)                      14×14×528       2       112     144             288     32              64      64          580K
// inception (4e)                      14×14×832       2       256     160             320     32              128     128         840K
// max pool 3×3/2                      7×7×832         0
// inception (5a)                      7×7×832         2       256     160             320     32              128     128         1072K
// inception (5b)                      7×7×1024        2       384     192             384     48              128     128         1388K
// avg pool 7×7/1                      1×1×1024        0
// dropout (40%)                       1×1×1024        0
// linear                              1×1×1000        1                                                                           1000K
// softmax                             1×1×1000        0

#include "DeepLearning/Api/ExampleNetworks/InceptionV3.h"

using namespace Thor;
using namespace std;

Network buildInceptionV3() {
    Network inceptionV3;

    vector<uint64_t> expectedDimensions;

    UniformRandom::Builder uniformRandomInitializerBuilder = UniformRandom::Builder().minValue(-0.1).maxValue(0.1);

    NetworkInput imagesInput =
        NetworkInput::Builder().network(inceptionV3).name("images").dimensions({3, 224, 224}).dataType(Tensor::DataType::UINT8).build();

    Tensor latestOutputTensor;

    // convolution     7×7/2               112×112×64      1                                                                           2.7K
    // max pool        3×3/2               56×56×64        0
    // convolution     3×3/1               56×56×192       2               64              192     112K                                360M
    // max pool        3×3/2               28×28×192       0
    // inception (3a)                      28×28×256       2       64      96              128     16              32      32          159K
    // inception (3b)                      28×28×480       2       128     128             192     32              96      64          380K
    // max pool 3×3/2                      14×14×480       0

    latestOutputTensor = Convolution2d::Builder()
                             .network(inceptionV3)
                             .featureInput(imagesInput.getFeatureOutput())
                             .numOutputChannels(64)
                             .filterHeight(7)
                             .filterWidth(7)
                             .verticalStride(2)
                             .horizontalStride(2)
                             .horizontalPadding(3)
                             .verticalPadding(3)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {64, 112, 112};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Pooling::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .type(Pooling::Type::MAX)
                             .windowHeight(3)
                             .windowWidth(3)
                             .verticalStride(2)
                             .horizontalStride(2)
                             .verticalPadding(1)
                             .horizontalPadding(1)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {64, 56, 56};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Convolution2d::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .numOutputChannels(192)
                             .filterHeight(3)
                             .filterWidth(3)
                             .verticalStride(1)
                             .horizontalStride(1)
                             .samePadding()
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {192, 56, 56};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Pooling::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .type(Pooling::Type::MAX)
                             .windowHeight(3)
                             .windowWidth(3)
                             .verticalStride(2)
                             .horizontalStride(2)
                             .verticalPadding(1)
                             .horizontalPadding(1)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {192, 28, 28};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Inception::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .outputChannels1x1(64)
                             .inputChannels3x3(96)
                             .outputChannels3x3(128)
                             .inputChannels5x5(16)
                             .outputChannels5x5(32)
                             .outputChannelsPooling(32)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {256, 28, 28};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Inception::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .outputChannels1x1(128)
                             .inputChannels3x3(128)
                             .outputChannels3x3(192)
                             .inputChannels5x5(32)
                             .outputChannels5x5(96)
                             .outputChannelsPooling(64)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {480, 28, 28};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Pooling::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .type(Pooling::Type::MAX)
                             .windowHeight(3)
                             .windowWidth(3)
                             .verticalStride(2)
                             .horizontalStride(2)
                             .verticalPadding(1)
                             .horizontalPadding(1)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {480, 14, 14};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    // inception (4a)                      14×14×512       2       192     96              208     16              48      64          364K
    // inception (4b)                      14×14×512       2       160     112             224     24              64      64          437K
    // inception (4c)                      14×14×512       2       128     128             256     24              64      64          463K
    // inception (4d)                      14×14×528       2       112     144             288     32              64      64          580K
    // inception (4e)                      14×14×832       2       256     160             320     32              128     128         840K
    // max pool 3×3/2                      7×7×832         0

    latestOutputTensor = Inception::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .outputChannels1x1(192)
                             .inputChannels3x3(96)
                             .outputChannels3x3(208)
                             .inputChannels5x5(16)
                             .outputChannels5x5(48)
                             .outputChannelsPooling(64)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {512, 14, 14};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Inception::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .outputChannels1x1(160)
                             .inputChannels3x3(112)
                             .outputChannels3x3(224)
                             .inputChannels5x5(24)
                             .outputChannels5x5(64)
                             .outputChannelsPooling(64)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {512, 14, 14};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Inception::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .outputChannels1x1(128)
                             .inputChannels3x3(128)
                             .outputChannels3x3(256)
                             .inputChannels5x5(24)
                             .outputChannels5x5(64)
                             .outputChannelsPooling(64)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {512, 14, 14};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Inception::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .outputChannels1x1(112)
                             .inputChannels3x3(144)
                             .outputChannels3x3(288)
                             .inputChannels5x5(32)
                             .outputChannels5x5(64)
                             .outputChannelsPooling(64)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {528, 14, 14};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Inception::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .outputChannels1x1(256)
                             .inputChannels3x3(160)
                             .outputChannels3x3(320)
                             .inputChannels5x5(32)
                             .outputChannels5x5(128)
                             .outputChannelsPooling(128)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {832, 14, 14};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Pooling::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .type(Pooling::Type::MAX)
                             .windowHeight(3)
                             .windowWidth(3)
                             .verticalStride(2)
                             .horizontalStride(2)
                             .verticalPadding(1)
                             .horizontalPadding(1)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {832, 7, 7};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    // inception (5a)                      7×7×832         2       256     160             320     32              128     128         1072K
    // inception (5b)                      7×7×1024        2       384     192             384     48              128     128         1388K
    // avg pool 7×7/1                      1×1×1024        0
    // dropout (40%)                       1×1×1024        0
    // linear                              1×1×1000        1                                                                           1000K

    latestOutputTensor = Inception::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .outputChannels1x1(256)
                             .inputChannels3x3(160)
                             .outputChannels3x3(320)
                             .inputChannels5x5(32)
                             .outputChannels5x5(128)
                             .outputChannelsPooling(128)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {832, 7, 7};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Inception::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .outputChannels1x1(384)
                             .inputChannels3x3(192)
                             .outputChannels3x3(384)
                             .inputChannels5x5(48)
                             .outputChannels5x5(128)
                             .outputChannelsPooling(128)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {1024, 7, 7};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = Pooling::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .type(Pooling::Type::AVERAGE)
                             .windowHeight(7)
                             .windowWidth(7)
                             .verticalStride(7)
                             .horizontalStride(7)
                             .verticalPadding(0)
                             .horizontalPadding(0)
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {1024, 1, 1};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = FullyConnected::Builder()
                             .network(inceptionV3)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(1000)
                             .dropOut(0.4)
                             .noActivation()
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {1000, 1, 1};

    Tensor labelsTensor = NetworkInput::Builder()
                              .network(inceptionV3)
                              .name("labels")
                              .dimensions({1000})
                              .dataType(Tensor::DataType::FP16)
                              .build()
                              .getFeatureOutput();

    CategoricalCrossEntropy lossLayer = CategoricalCrossEntropy::Builder()
                                            .network(inceptionV3)
                                            .predictions(latestOutputTensor)
                                            .labels(labelsTensor)
                                            .reportsBatchLoss()
                                            .build();

    latestOutputTensor = lossLayer.getFeatureInput();
    labelsTensor = lossLayer.getLabels();

    NetworkOutput predictions = NetworkOutput::Builder()
                                    .network(inceptionV3)
                                    .name("predictions")
                                    .inputTensor(lossLayer.getPredictions())
                                    .dataType(Tensor::DataType::FP32)
                                    .build();
    NetworkOutput loss = NetworkOutput::Builder()
                             .network(inceptionV3)
                             .name("loss")
                             .inputTensor(lossLayer.getLoss())
                             .dataType(Tensor::DataType::FP32)
                             .build();

    return inceptionV3;
}
