// #include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
// #include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"
// #include "DeepLearning/Api/Network/PlacedNetwork.h"
// #include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
//
// #include "gtest/gtest.h"
//
// #include <stdio.h>
// #include <memory>
//
// #include "DeepLearning/Api/Optimizers/Sgd.h"
//
// using namespace std;
// using namespace Thor;
// using json = nlohmann::json;
//
// TEST(CategoricalCrossEntropy, ClassIndexLabelsBatchLossBuilds) {
//     srand(time(nullptr));
//
//     for (uint32_t t = 0; t < 10; ++t) {
//         Network network("testNetwork");
//
//         // API layer does not have a batch dimension
//         vector<uint64_t> labelDimensions = {1};
//         uint64_t numClasses = 2UL + (rand() % 1000);
//         vector<uint64_t> predictionsDimensions = {numClasses};
//         DataType predictionsDataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
//         DataType lossDataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
//
//         DataType labelsDataType;
//         uint32_t r = rand() % 3;
//         if (r == 0)
//             labelsDataType = DataType::UINT8;
//         else if (r == 1)
//             labelsDataType = DataType::UINT16;
//         else
//             labelsDataType = DataType::UINT32;
//
//         Tensor predictions(predictionsDataType, predictionsDimensions);
//         Tensor labels(labelsDataType, labelDimensions);
//
//         CategoricalCrossEntropy::Builder crossEntropyBuilder = CategoricalCrossEntropy::Builder()
//                                                                    .network(network)
//                                                                    .predictions(predictions)
//                                                                    .reportsBatchLoss()
//                                                                    .receivesClassIndexLabels(numClasses)
//                                                                    .lossDataType(lossDataType)
//                                                                    .labels(labels);
//         CategoricalCrossEntropy crossEntropy = crossEntropyBuilder.build();
//
//         ASSERT_TRUE(crossEntropy.isInitialized());
//
//         std::optional<Tensor> actualInput = crossEntropy.getFeatureInput();
//         ASSERT_TRUE(actualInput.has_value());
//         ASSERT_EQ(actualInput.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(actualInput.value().getDimensions(), predictionsDimensions);
//
//         std::optional<Tensor> actualLabels = crossEntropy.getLabels();
//         ASSERT_TRUE(actualLabels.has_value());
//         ASSERT_EQ(actualLabels.value().getDataType(), labelsDataType);
//         ASSERT_EQ(actualLabels.value().getDimensions(), labelDimensions);
//
//         std::optional<Tensor> actualPredictions = crossEntropy.getPredictions();
//         ASSERT_TRUE(actualPredictions.has_value());
//         ASSERT_EQ(actualPredictions.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(actualPredictions.value().getDimensions(), predictionsDimensions);
//
//         std::optional<Tensor> actualLoss = crossEntropy.getLoss();
//         ASSERT_TRUE(actualLoss.has_value());
//         ASSERT_EQ(actualLoss.value().getDataType(), lossDataType);
//         ASSERT_EQ(actualLoss.value().getDimensions(), vector<uint64_t>(1, 1));
//
//         shared_ptr<Layer> cloneLayer = crossEntropy.clone();
//         CategoricalCrossEntropy *clone = dynamic_cast<CategoricalCrossEntropy *>(cloneLayer.get());
//         assert(clone != nullptr);
//
//         ASSERT_TRUE(clone->isInitialized());
//
//         std::optional<Tensor> cloneInput = clone->getFeatureInput();
//         ASSERT_TRUE(cloneInput.has_value());
//         ASSERT_EQ(cloneInput.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(cloneInput.value().getDimensions(), predictionsDimensions);
//
//         std::optional<Tensor> cloneLabels = clone->getLabels();
//         ASSERT_TRUE(cloneLabels.has_value());
//         ASSERT_EQ(cloneLabels.value().getDataType(), labelsDataType);
//         ASSERT_EQ(cloneLabels.value().getDimensions(), labelDimensions);
//
//         std::optional<Tensor> clonePredictions = clone->getPredictions();
//         ASSERT_TRUE(clonePredictions.has_value());
//         ASSERT_EQ(clonePredictions.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(clonePredictions.value().getDimensions(), predictionsDimensions);
//
//         std::optional<Tensor> cloneLoss = clone->getLoss();
//         ASSERT_TRUE(cloneLoss.has_value());
//         ASSERT_EQ(cloneLoss.value().getDataType(), lossDataType);
//         ASSERT_EQ(cloneLoss.value().getDimensions(), vector<uint64_t>(1, 1));
//
//         ASSERT_EQ(crossEntropy.getId(), clone->getId());
//         ASSERT_GT(crossEntropy.getId(), 1u);
//
//         ASSERT_TRUE(crossEntropy == *clone);
//         ASSERT_FALSE(crossEntropy != *clone);
//         ASSERT_FALSE(crossEntropy > *clone);
//         ASSERT_FALSE(crossEntropy < *clone);
//     }
// }
//
// TEST(CategoricalCrossEntropy, OneHotLabelsClasswiseLossBuilds) {
//     srand(time(nullptr));
//
//     for (uint32_t t = 0; t < 10; ++t) {
//         Network network("testNetwork");
//
//         vector<uint64_t> dimensions;
//         dimensions = {2UL + (rand() % 300)};
//         DataType predictionsDataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
//         DataType lossDataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
//
//         DataType labelsDataType;
//         uint32_t r = rand() % 5;
//         if (r == 0)
//             labelsDataType = DataType::UINT8;
//         else if (r == 1)
//             labelsDataType = DataType::UINT16;
//         else if (r == 2)
//             labelsDataType = DataType::UINT32;
//         else if (r == 3)
//             labelsDataType = DataType::FP16;
//         else
//             labelsDataType = DataType::FP32;
//
//         Tensor predictions(predictionsDataType, dimensions);
//         Tensor labels(labelsDataType, dimensions);
//
//         CategoricalCrossEntropy::Builder crossEntropyBuilder = CategoricalCrossEntropy::Builder()
//                                                                    .network(network)
//                                                                    .predictions(predictions)
//                                                                    .reportsClasswiseLoss()
//                                                                    .receivesOneHotLabels()
//                                                                    .lossDataType(lossDataType)
//                                                                    .labels(labels);
//         CategoricalCrossEntropy crossEntropy = crossEntropyBuilder.build();
//
//         ASSERT_TRUE(crossEntropy.isInitialized());
//
//         std::optional<Tensor> actualInput = crossEntropy.getFeatureInput();
//         ASSERT_TRUE(actualInput.has_value());
//         ASSERT_EQ(actualInput.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(actualInput.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> actualLabels = crossEntropy.getLabels();
//         ASSERT_TRUE(actualLabels.has_value());
//         ASSERT_EQ(actualLabels.value().getDataType(), labelsDataType);
//         ASSERT_EQ(actualLabels.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> actualPredictions = crossEntropy.getPredictions();
//         ASSERT_TRUE(actualPredictions.has_value());
//         ASSERT_EQ(actualPredictions.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(actualPredictions.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> actualLoss = crossEntropy.getLoss();
//         ASSERT_TRUE(actualLoss.has_value());
//         ASSERT_EQ(actualLoss.value().getDataType(), lossDataType);
//         ASSERT_EQ(actualLoss.value().getDimensions(), dimensions);
//
//         shared_ptr<Layer> cloneLayer = crossEntropy.clone();
//         CategoricalCrossEntropy *clone = dynamic_cast<CategoricalCrossEntropy *>(cloneLayer.get());
//         assert(clone != nullptr);
//
//         ASSERT_TRUE(clone->isInitialized());
//
//         std::optional<Tensor> cloneInput = clone->getFeatureInput();
//         ASSERT_TRUE(cloneInput.has_value());
//         ASSERT_EQ(cloneInput.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(cloneInput.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> cloneLabels = clone->getLabels();
//         ASSERT_TRUE(cloneLabels.has_value());
//         ASSERT_EQ(cloneLabels.value().getDataType(), labelsDataType);
//         ASSERT_EQ(cloneLabels.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> clonePredictions = clone->getPredictions();
//         ASSERT_TRUE(clonePredictions.has_value());
//         ASSERT_EQ(clonePredictions.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(clonePredictions.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> cloneLoss = clone->getLoss();
//         ASSERT_TRUE(cloneLoss.has_value());
//         ASSERT_EQ(cloneLoss.value().getDataType(), lossDataType);
//         ASSERT_EQ(cloneLoss.value().getDimensions(), dimensions);
//
//         ASSERT_EQ(crossEntropy.getId(), clone->getId());
//         ASSERT_GT(crossEntropy.getId(), 1u);
//
//         ASSERT_TRUE(crossEntropy == *clone);
//         ASSERT_FALSE(crossEntropy != *clone);
//         ASSERT_FALSE(crossEntropy > *clone);
//         ASSERT_FALSE(crossEntropy < *clone);
//     }
// }
//
// TEST(CategoricalCrossEntropy, OneHotLabelsElementwiseLossBuilds) {
//     srand(time(nullptr));
//
//     for (uint32_t t = 0; t < 10; ++t) {
//         Network network("testNetwork");
//
//         vector<uint64_t> dimensions;
//         dimensions = {2UL + (rand() % 300)};
//         DataType predictionsDataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
//         DataType lossDataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
//
//         DataType labelsDataType;
//         uint32_t r = rand() % 5;
//         if (r == 0)
//             labelsDataType = DataType::UINT8;
//         else if (r == 1)
//             labelsDataType = DataType::UINT16;
//         else if (r == 2)
//             labelsDataType = DataType::UINT32;
//         else if (r == 3)
//             labelsDataType = DataType::FP16;
//         else if (r == 4)
//             labelsDataType = DataType::FP32;
//
//         Tensor predictions(predictionsDataType, dimensions);
//         Tensor labels(labelsDataType, dimensions);
//
//         CategoricalCrossEntropy::Builder crossEntropyBuilder = CategoricalCrossEntropy::Builder()
//                                                                    .network(network)
//                                                                    .predictions(predictions)
//                                                                    .reportsElementwiseLoss()
//                                                                    .receivesOneHotLabels()
//                                                                    .lossDataType(lossDataType)
//                                                                    .labels(labels);
//         CategoricalCrossEntropy crossEntropy = crossEntropyBuilder.build();
//
//         ASSERT_TRUE(crossEntropy.isInitialized());
//
//         std::optional<Tensor> actualInput = crossEntropy.getFeatureInput();
//         ASSERT_TRUE(actualInput.has_value());
//         ASSERT_EQ(actualInput.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(actualInput.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> actualLabels = crossEntropy.getLabels();
//         ASSERT_TRUE(actualLabels.has_value());
//         ASSERT_EQ(actualLabels.value().getDataType(), labelsDataType);
//         ASSERT_EQ(actualLabels.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> actualPredictions = crossEntropy.getPredictions();
//         ASSERT_TRUE(actualPredictions.has_value());
//         ASSERT_EQ(actualPredictions.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(actualPredictions.value().getDimensions(), dimensions);
//
//         vector<uint64_t> batchLossDimensions = {1UL};
//         std::optional<Tensor> actualLoss = crossEntropy.getLoss();
//         ASSERT_TRUE(actualLoss.has_value());
//         ASSERT_EQ(actualLoss.value().getDataType(), lossDataType);
//         ASSERT_EQ(actualLoss.value().getDimensions(), batchLossDimensions);
//
//         shared_ptr<Layer> cloneLayer = crossEntropy.clone();
//         CategoricalCrossEntropy *clone = dynamic_cast<CategoricalCrossEntropy *>(cloneLayer.get());
//         assert(clone != nullptr);
//
//         ASSERT_TRUE(clone->isInitialized());
//
//         std::optional<Tensor> cloneInput = clone->getFeatureInput();
//         ASSERT_TRUE(cloneInput.has_value());
//         ASSERT_EQ(cloneInput.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(cloneInput.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> cloneLabels = clone->getLabels();
//         ASSERT_TRUE(cloneLabels.has_value());
//         ASSERT_EQ(cloneLabels.value().getDataType(), labelsDataType);
//         ASSERT_EQ(cloneLabels.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> clonePredictions = clone->getPredictions();
//         ASSERT_TRUE(clonePredictions.has_value());
//         ASSERT_EQ(clonePredictions.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(clonePredictions.value().getDimensions(), dimensions);
//
//         std::optional<Tensor> cloneLoss = clone->getLoss();
//         ASSERT_TRUE(cloneLoss.has_value());
//         ASSERT_EQ(cloneLoss.value().getDataType(), lossDataType);
//         ASSERT_EQ(cloneLoss.value().getDimensions(), batchLossDimensions);
//
//         ASSERT_EQ(crossEntropy.getId(), clone->getId());
//         ASSERT_GT(crossEntropy.getId(), 1u);
//
//         ASSERT_TRUE(crossEntropy == *clone);
//         ASSERT_FALSE(crossEntropy != *clone);
//         ASSERT_FALSE(crossEntropy > *clone);
//         ASSERT_FALSE(crossEntropy < *clone);
//     }
// }
//
// TEST(CategoricalCrossEntropy, ClassIndexLabelsRawLossBuilds) {
//     srand(time(nullptr));
//
//     for (uint32_t t = 0; t < 10; ++t) {
//         Network network("testNetwork");
//
//         vector<uint64_t> labelDimensions = {1};
//         uint64_t numClasses = 2UL + (rand() % 1000);
//         vector<uint64_t> predictionsDimensions = {numClasses};
//         DataType predictionsDataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
//         DataType lossDataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
//
//         DataType labelsDataType;
//         uint32_t r = rand() % 3;
//         if (r == 0)
//             labelsDataType = DataType::UINT8;
//         else if (r == 1)
//             labelsDataType = DataType::UINT16;
//         else
//             labelsDataType = DataType::UINT32;
//
//         Tensor predictions(predictionsDataType, predictionsDimensions);
//         Tensor labels(labelsDataType, labelDimensions);
//
//         CategoricalCrossEntropy::Builder crossEntropyBuilder = CategoricalCrossEntropy::Builder()
//                                                                    .network(network)
//                                                                    .predictions(predictions)
//                                                                    .reportsRawLoss()
//                                                                    .receivesClassIndexLabels(numClasses)
//                                                                    .lossDataType(lossDataType)
//                                                                    .labels(labels);
//         CategoricalCrossEntropy crossEntropy = crossEntropyBuilder.build();
//
//         ASSERT_TRUE(crossEntropy.isInitialized());
//
//         std::optional<Tensor> actualInput = crossEntropy.getFeatureInput();
//         ASSERT_TRUE(actualInput.has_value());
//         ASSERT_EQ(actualInput.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(actualInput.value().getDimensions(), predictionsDimensions);
//
//         std::optional<Tensor> actualLabels = crossEntropy.getLabels();
//         ASSERT_TRUE(actualLabels.has_value());
//         ASSERT_EQ(actualLabels.value().getDataType(), labelsDataType);
//         ASSERT_EQ(actualLabels.value().getDimensions(), labelDimensions);
//
//         std::optional<Tensor> actualPredictions = crossEntropy.getPredictions();
//         ASSERT_TRUE(actualPredictions.has_value());
//         ASSERT_EQ(actualPredictions.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(actualPredictions.value().getDimensions(), predictionsDimensions);
//
//         std::optional<Tensor> actualLoss = crossEntropy.getLoss();
//         ASSERT_TRUE(actualLoss.has_value());
//         ASSERT_EQ(actualLoss.value().getDataType(), lossDataType);
//         ASSERT_EQ(actualLoss.value().getDimensions(), vector<uint64_t>(1, numClasses));
//
//         shared_ptr<Layer> cloneLayer = crossEntropy.clone();
//         CategoricalCrossEntropy *clone = dynamic_cast<CategoricalCrossEntropy *>(cloneLayer.get());
//         assert(clone != nullptr);
//
//         ASSERT_TRUE(clone->isInitialized());
//
//         std::optional<Tensor> cloneInput = clone->getFeatureInput();
//         ASSERT_TRUE(cloneInput.has_value());
//         ASSERT_EQ(cloneInput.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(cloneInput.value().getDimensions(), predictionsDimensions);
//
//         std::optional<Tensor> cloneLabels = clone->getLabels();
//         ASSERT_TRUE(cloneLabels.has_value());
//         ASSERT_EQ(cloneLabels.value().getDataType(), labelsDataType);
//         ASSERT_EQ(cloneLabels.value().getDimensions(), labelDimensions);
//
//         std::optional<Tensor> clonePredictions = clone->getPredictions();
//         ASSERT_TRUE(clonePredictions.has_value());
//         ASSERT_EQ(clonePredictions.value().getDataType(), predictionsDataType);
//         ASSERT_EQ(clonePredictions.value().getDimensions(), predictionsDimensions);
//
//         std::optional<Tensor> cloneLoss = clone->getLoss();
//         ASSERT_TRUE(cloneLoss.has_value());
//         ASSERT_EQ(cloneLoss.value().getDataType(), lossDataType);
//         ASSERT_EQ(cloneLoss.value().getDimensions(), vector<uint64_t>(1, numClasses));
//
//         ASSERT_EQ(crossEntropy.getId(), clone->getId());
//         ASSERT_GT(crossEntropy.getId(), 1u);
//
//         ASSERT_TRUE(crossEntropy == *clone);
//         ASSERT_FALSE(crossEntropy != *clone);
//         ASSERT_FALSE(crossEntropy > *clone);
//         ASSERT_FALSE(crossEntropy < *clone);
//     }
// }
//
// TEST(CategoricalCrossEntropy, SerializeDeserialize) {
//     srand(time(nullptr));
//
//     Network initialNetwork("initialNetwork");
//     DataType predictionsDataType = DataType::FP16;
//     uint32_t numClasses = 2 + (rand() % 100);
//     vector<uint64_t> inputDimensions = {numClasses};
//     DataType lossDataType = rand() % 2 ? DataType::FP16 : DataType::FP32;
//     DataType labelDataType = rand() % 2 ? DataType::UINT16 : DataType::UINT32;
//     NetworkInput predictionsInput = NetworkInput::Builder()
//                                         .network(initialNetwork)
//                                         .name("predictionsInput")
//                                         .dimensions(inputDimensions)
//                                         .dataType(predictionsDataType)
//                                         .build();
//
//     vector<uint64_t> labelDimensions;
//     uint32_t labelType = rand() % 2;
//     if (labelType == 0) {
//         labelDimensions = {1UL};
//     } else {
//         labelDimensions = {numClasses};
//     }
//
//     FullyConnected fullyConnected = FullyConnected::Builder()
//                                         .network(initialNetwork)
//                                         .featureInput(predictionsInput.getFeatureOutput())
//                                         .numOutputFeatures(numClasses)
//                                         .hasBias(false)
//                                         .noActivation()
//                                         .build();
//
//     CategoricalCrossEntropy::Builder categoricalCrossEntropyBuilder = CategoricalCrossEntropy::Builder()
//                                                                           .network(initialNetwork)
//                                                                           .predictions(fullyConnected.getFeatureOutput())
//                                                                           .lossDataType(lossDataType);
//
//     if (labelType == 0) {
//         categoricalCrossEntropyBuilder.receivesClassIndexLabels(numClasses);
//     } else {
//         categoricalCrossEntropyBuilder.receivesOneHotLabels();
//     }
//
//     NetworkInput labelsInput =
//         NetworkInput::Builder().network(initialNetwork).name("labelsInput").dimensions(labelDimensions).dataType(labelDataType).build();
//
//     categoricalCrossEntropyBuilder.labels(labelsInput.getFeatureOutput());
//
//     uint32_t lossShape = rand() % 4;
//     vector<uint64_t> lossDimensions;
//     if (lossShape == 0) {
//         categoricalCrossEntropyBuilder.reportsBatchLoss();
//         lossDimensions = {1UL};
//     } else if (lossShape == 1) {
//         categoricalCrossEntropyBuilder.reportsClasswiseLoss();
//         lossDimensions = {numClasses};
//     } else if (lossShape == 2) {
//         categoricalCrossEntropyBuilder.reportsElementwiseLoss();
//         lossDimensions = {1UL};
//     } else {
//         categoricalCrossEntropyBuilder.reportsRawLoss();
//         lossDimensions = inputDimensions;
//     }
//
//     CategoricalCrossEntropy categoricalCrossEntropy = categoricalCrossEntropyBuilder.build();
//
//     shared_ptr<Sgd> sgd = Sgd::Builder().network(initialNetwork).initialLearningRate(0.1).decay(0.1).build();
//
//     NetworkOutput lossOutput = NetworkOutput::Builder()
//                                    .network(initialNetwork)
//                                    .name("lossOutput")
//                                    .inputTensor(categoricalCrossEntropy.getLoss())
//                                    .dataType(lossDataType)
//                                    .build();
//
//     ASSERT_TRUE(categoricalCrossEntropy.isInitialized());
//
//     // Now stamp the network and test serialization
//     Stream stream(0);
//     uint32_t batchSize = 1 + (rand() % 16);
//     vector<Event> initDoneEvents;
//     shared_ptr<PlacedNetwork> initialPlacedNetwork = initialNetwork.place(batchSize, initDoneEvents);
//     ASSERT_TRUE(initialPlacedNetwork != nullptr);
//     for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
//         stream.waitEvent(initDoneEvents[i]);
//     }
//     initDoneEvents.clear();
//
//     // Fetch the layer from the network
//     ASSERT_EQ(initialPlacedNetwork->getNumStamps(), 1UL);
//     ThorImplementation::StampedNetwork &stampedNetwork = initialPlacedNetwork->getStampedNetwork(0);
//
//     // Find the softmax layer in the network so can serialize it for this test case
//     shared_ptr<Softmax> softmax;
//     shared_ptr<LossShaper> lossShaper;
//     bool softmaxFound = false;
//     bool lossShaperFound = false;
//     for (int32_t i = 0; i < initialNetwork.getNumLayers(); ++i) {
//         shared_ptr<Layer> layer = initialNetwork.getLayer(i);
//         if (!softmaxFound) {
//             softmax = dynamic_pointer_cast<Softmax>(layer);
//             if (softmax)
//                 softmaxFound = true;
//         }
//         if (!lossShaperFound) {
//             lossShaper = dynamic_pointer_cast<LossShaper>(layer);
//             if (lossShaper)
//                 lossShaperFound = true;
//         }
//         if (softmaxFound && lossShaperFound)
//             break;
//     }
//     ASSERT_TRUE(softmaxFound);
//     ASSERT_EQ(lossShaperFound, lossShape != 3);
//
//     thor_file::TarWriter archiveWriter("testModel");
//
//     json labelsInputJ = labelsInput.serialize(archiveWriter, stream);
//     json networkInputJ = predictionsInput.serialize(archiveWriter, stream);
//     json softmaxJ = softmax->serialize(archiveWriter, stream);
//     Layer *layer = &categoricalCrossEntropy;
//     json categoricalCrossEntropyJ = layer->serialize(archiveWriter, stream);
//     json lossShaperJ;
//     if (lossShaper)
//         lossShaperJ = lossShaper->serialize(archiveWriter, stream);
//     json lossOutputJ = lossOutput.serialize(archiveWriter, stream);
//
//     ASSERT_EQ(categoricalCrossEntropyJ["factory"], "loss");
//     ASSERT_EQ(categoricalCrossEntropyJ["version"], "1.0.0");
//     ASSERT_EQ(categoricalCrossEntropyJ["layer_type"], "categorical_cross_entropy");
//     EXPECT_TRUE(categoricalCrossEntropyJ.contains("layer_name"));
//     ASSERT_EQ(categoricalCrossEntropyJ.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::RAW);
//     ASSERT_EQ(categoricalCrossEntropyJ.at("loss_data_type").get<DataType>(), lossDataType);
//
//     const json &labelsJ = categoricalCrossEntropyJ["labels_tensor"];
//     ASSERT_EQ(labelsJ.at("data_type").get<DataType>(), labelDataType);
//     ASSERT_EQ(labelsJ.at("dimensions").get<vector<uint64_t>>(), labelDimensions);
//     ASSERT_TRUE(labelsJ.at("id").is_number_integer());
//
//     const json &predictionsJ = categoricalCrossEntropyJ["predictions_tensor"];
//     ASSERT_EQ(predictionsJ.at("data_type").get<DataType>(), predictionsDataType);
//     ASSERT_EQ(predictionsJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
//     ASSERT_TRUE(predictionsJ.at("id").is_number_integer());
//
//     const json &softmaxOutputJ = categoricalCrossEntropyJ["softmax_output_tensor"];
//     ASSERT_EQ(softmaxOutputJ.at("data_type").get<DataType>(), predictionsDataType);
//     ASSERT_EQ(softmaxOutputJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
//     ASSERT_TRUE(softmaxOutputJ.at("id").is_number_integer());
//
//     if (lossShaper) {
//         const json &lossShaperInputJ = categoricalCrossEntropyJ["loss_shaper_input_tensor"];
//         ASSERT_EQ(lossShaperInputJ.at("data_type").get<DataType>(), lossDataType);
//         ASSERT_EQ(lossShaperInputJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
//         ASSERT_TRUE(lossShaperInputJ.at("id").is_number_integer());
//     }
//
//     const json &lossJ = categoricalCrossEntropyJ["loss_tensor"];
//     ASSERT_EQ(lossJ.at("data_type").get<DataType>(), lossDataType);
//     ASSERT_EQ(lossJ.at("dimensions").get<vector<uint64_t>>(), lossDimensions);
//     ASSERT_TRUE(lossJ.at("id").is_number_integer());
//
//     // The network attached the optimizer to its copy of the FC layer
//     json fullyConnectedJ;
//     bool fcFound = false;
//     shared_ptr<FullyConnected> initalNetworkFC;
//     for (int32_t i = 0; i < initialNetwork.getNumTrainableLayers(); ++i) {
//         shared_ptr<TrainableWeightsBiasesLayer> layer = initialNetwork.getTrainableLayer(i);
//         initalNetworkFC = dynamic_pointer_cast<FullyConnected>(layer);
//         if (initalNetworkFC) {
//             fullyConnectedJ = initalNetworkFC->serialize(archiveWriter, stream, true, initialPlacedNetwork->getStampedNetwork(0));
//             fcFound = true;
//             break;
//         }
//     }
//     ASSERT_TRUE(fcFound);
//
//     // printf("%s\n", networkInputJ.dump(4).c_str());
//     // printf("%s\n", labelsInputJ.dump(4).c_str());
//     // printf("%s\n", fullyConnectedJ.dump(4).c_str());
//     // printf("%s\n", softmaxJ.dump(4).c_str());
//     // printf("%s\n", categoricalCrossEntropyJ.dump(4).c_str());
//     // if (lossShaper)&
//     //     printf("%s\n", lossShaperJ.dump(4).c_str());
//     // printf("%s\n", lossOutputJ.dump(4).c_str());
//
//     // ////////////////////////////
//     // // Deserialize
//     // ////////////////////////////
//     // Verify that the layer gets added to the network and that its weights are set to the correct values
//     Network newNetwork("newNetwork");
//
//     archiveWriter.createArchive("/tmp/", true);
//     shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("testModel", "/tmp/");
//
//     Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
//     Layer::deserialize(archiveReader, labelsInputJ, &newNetwork);
//     Layer::deserialize(archiveReader, fullyConnectedJ, &newNetwork);
//     Layer::deserialize(archiveReader, softmaxJ, &newNetwork);
//     Layer::deserialize(archiveReader, categoricalCrossEntropyJ, &newNetwork);
//     if (lossShaper)
//         Layer::deserialize(archiveReader, lossShaperJ, &newNetwork);
//     Layer::deserialize(archiveReader, lossOutputJ, &newNetwork);
//
//     batchSize = 1 + (rand() % 16);
//     shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
//     ASSERT_TRUE(newPlacedNetwork != nullptr);
//     for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
//         stream.waitEvent(initDoneEvents[i]);
//     }
//     initDoneEvents.clear();
//
//     ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
//     stampedNetwork = newPlacedNetwork->getStampedNetwork(0);
//
//     vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
//     if (lossShaper)
//         ASSERT_EQ(otherLayers.size(), 3U);
//     else
//         ASSERT_EQ(otherLayers.size(), 2U);
//     shared_ptr<ThorImplementation::Softmax> stampedSoftmax;
//     shared_ptr<ThorImplementation::CrossEntropy> stampedCategoricalCrossEntropy;
//     shared_ptr<ThorImplementation::LossShaper> stampedLossShaper;
//     softmaxFound = false;
//     bool crossEntropyFound = false;
//     lossShaperFound = false;
//     for (shared_ptr<ThorImplementation::Layer> layer : otherLayers) {
//         if (!softmaxFound) {
//             stampedSoftmax = dynamic_pointer_cast<ThorImplementation::Softmax>(layer);
//             if (stampedSoftmax != nullptr)
//                 softmaxFound = true;
//         }
//         if (!crossEntropyFound) {
//             stampedCategoricalCrossEntropy = dynamic_pointer_cast<ThorImplementation::CrossEntropy>(layer);
//             if (stampedCategoricalCrossEntropy != nullptr)
//                 crossEntropyFound = true;
//         }
//         if (!lossShaperFound) {
//             stampedLossShaper = dynamic_pointer_cast<ThorImplementation::LossShaper>(layer);
//             if (stampedLossShaper != nullptr)
//                 lossShaperFound = true;
//         }
//     }
//     ASSERT_TRUE(softmaxFound);
//     ASSERT_TRUE(crossEntropyFound);
//     ASSERT_EQ(lossShaperFound, lossShape != 3);
//
//     ASSERT_NE(stampedCategoricalCrossEntropy, nullptr);
//
//     shared_ptr<ThorImplementation::FullyConnected> stampedFC =
//         dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(0));
//     ASSERT_NE(stampedFC, nullptr);
//
//     // vector<shared_ptr<ThorImplementation::NetworkInput>> stampedInputs = stampedNetwork.getInputs();
//     shared_ptr<ThorImplementation::NetworkInput> stampedLabelsInput;
//     for (auto input : stampedNetwork.getInputs()) {
//         if (input->getName() == "labelsInput") {
//             stampedLabelsInput = input;
//         }
//     }
//     ASSERT_TRUE(stampedLabelsInput != nullptr);
//
//     shared_ptr<ThorImplementation::NetworkOutput> stampedLossOutput;
//     stampedLossOutput = stampedNetwork.getOutputs()[0];
//
//     ASSERT_EQ(stampedSoftmax->getFeatureInput().value(), stampedFC->getFeatureOutputs()[0].value());
//     ASSERT_EQ(stampedCategoricalCrossEntropy->getPredictionsInput().value(), stampedSoftmax->getFeatureOutput().value());
//     ASSERT_EQ(stampedCategoricalCrossEntropy->getLabelsInput().value(), stampedLabelsInput->getFeatureOutput().value());
//     if (lossShaper) {
//         ASSERT_EQ(stampedCategoricalCrossEntropy->getLossOutput().value(), stampedLossShaper->getFeatureInput().value());
//         ASSERT_EQ(stampedLossShaper->getFeatureOutput().value(), stampedLossOutput->getFeatureInput().value());
//     } else {
//         ASSERT_EQ(stampedCategoricalCrossEntropy->getLossOutput().value(), stampedLossOutput->getFeatureInput().value());
//     }
//
//     filesystem::remove("/tmp/testModel.thor.tar");
// }
#include <optional>
