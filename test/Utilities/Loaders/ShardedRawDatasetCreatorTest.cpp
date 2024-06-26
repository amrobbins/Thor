#include "Thor.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <memory.h>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "omp.h"

using std::mutex;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::unordered_set;
using std::vector;

using namespace boost::filesystem;
using namespace std;

class TestDataProcessor : public DataProcessor {
    virtual uint64_t outputTensorSizeInBytes() { return 100; }

    virtual DataElement operator()(DataElement &input) { return input; }
};

void verifyImages(Shard &shard) {
    uint8_t buffer[224 * 224 * 3];
    string label;
    string filename;
    for (int exampleTypeInt = (int)ExampleType::TRAIN; exampleTypeInt <= (int)ExampleType::TEST; ++exampleTypeInt) {
        ExampleType exampleType = (ExampleType)exampleTypeInt;
        set<string> exampleNames;
        for (uint32_t i = 0; i < shard.getNumExamples(exampleType); ++i) {
            shard.loadExample(buffer, label, filename, exampleType, i);
            string qualifiedExampleName = label + '/' + filename;
            ASSERT_EQ(exampleNames.count(qualifiedExampleName), 0u);
            exampleNames.insert(qualifiedExampleName);

            uint8_t pixel[3];
            if (filename == "white.png") {
                pixel[0] = 255;
                pixel[1] = 255;
                pixel[2] = 255;
            } else if (filename == "black.png") {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 0;
            } else if (filename == "red.png") {
                pixel[0] = 255;
                pixel[1] = 0;
                pixel[2] = 0;
            } else if (filename == "green.png") {
                pixel[0] = 0;
                pixel[1] = 255;
                pixel[2] = 0;
            } else if (filename == "blue.png") {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 255;
            } else {
                ASSERT_TRUE(false);
            }

            for (uint32_t p = 0; p < 224 * 224; ++p)
                ASSERT_EQ(buffer[p], pixel[0]);
            for (uint32_t p = 0; p < 224 * 224; ++p)
                ASSERT_EQ(buffer[224 * 224 + p], pixel[1]);
            for (uint32_t p = 0; p < 224 * 224; ++p)
                ASSERT_EQ(buffer[2 * 224 * 224 + p], pixel[2]);
        }
    }
}

void verifyBatch(uint64_t batchSize, vector<ThorImplementation::Tensor> batchTensors, vector<ThorImplementation::Tensor> labelTensors) {
    const uint64_t NUM_CLASSES = 4;
    uint64_t numMiniBatches = batchTensors.size();
    ASSERT_GT(batchSize, 0u);
    ASSERT_GT(numMiniBatches, 0u);

    uint64_t globalCount = 0;
    map<string, uint64_t> colorCount;

    for (uint64_t i = 0; i < batchTensors.size(); ++i) {
        ASSERT_EQ(batchTensors[i].getDescriptor().getTotalNumElements(), batchSize * 224 * 224 * 3);
        ASSERT_EQ(labelTensors[i].getDescriptor().getTotalNumElements(), batchSize * NUM_CLASSES);

        for (uint64_t j = 0; j < batchSize; ++j) {
            bool oneHotFound = false;
            float *oneHotLabelArray = (float *)labelTensors[i].getMemPtr();
            for (uint64_t k = 0; k < NUM_CLASSES; ++k) {
                if (oneHotLabelArray[j * NUM_CLASSES + k] != 0) {
                    ASSERT_EQ(oneHotLabelArray[j * NUM_CLASSES + k], 1.0f);
                    ASSERT_EQ(oneHotFound, false);
                    oneHotFound = true;
                }
            }
            ASSERT_EQ(oneHotFound, true);

            string imageColor = "";
            uint8_t *image = (uint8_t *)batchTensors[i].getMemPtr();
            image += j * 224 * 224 * 3;
            if (image[0] == 255 && image[224 * 224] == 255 && image[2 * 224 * 224] == 255) {
                imageColor = "white";
                for (int k = 0; k < 224 * 224; ++k) {
                    ASSERT_EQ(image[k], 255);
                    ASSERT_EQ(image[224 * 224 + k], 255);
                    ASSERT_EQ(image[2 * 224 * 224 + k], 255);
                }
            } else if (image[0] == 0 && image[224 * 224] == 0 && image[2 * 224 * 224] == 0) {
                imageColor = "black";
                for (int k = 0; k < 224 * 224; ++k) {
                    ASSERT_EQ(image[k], 0);
                    ASSERT_EQ(image[224 * 224 + k], 0);
                    ASSERT_EQ(image[2 * 224 * 224 + k], 0);
                }
            } else if (image[0] == 255 && image[224 * 224] == 0 && image[2 * 224 * 224] == 0) {
                imageColor = "red";
                for (int k = 0; k < 224 * 224; ++k) {
                    ASSERT_EQ(image[k], 255);
                    ASSERT_EQ(image[224 * 224 + k], 0);
                    ASSERT_EQ(image[2 * 224 * 224 + k], 0);
                }
            } else if (image[0] == 0 && image[224 * 224] == 255 && image[2 * 224 * 224] == 0) {
                imageColor = "green";
                for (int k = 0; k < 224 * 224; ++k) {
                    ASSERT_EQ(image[k], 0);
                    ASSERT_EQ(image[224 * 224 + k], 255);
                    ASSERT_EQ(image[2 * 224 * 224 + k], 0);
                }
            } else if (image[0] == 0 && image[224 * 224] == 0 && image[2 * 224 * 224] == 255) {
                imageColor = "blue";
                for (int k = 0; k < 224 * 224; ++k) {
                    ASSERT_EQ(image[k], 0);
                    ASSERT_EQ(image[224 * 224 + k], 0);
                    ASSERT_EQ(image[2 * 224 * 224 + k], 255);
                }
            }
            ASSERT_GT(imageColor.size(), 0u);

            // printf("%s\n", imageColor.c_str());

            globalCount += 1;
            colorCount[imageColor] += 1;
            if (globalCount % 7 == 0) {
                ASSERT_EQ(colorCount["white"], 1u);
                ASSERT_EQ(colorCount["green"], 1u);
                ASSERT_EQ(colorCount["red"], 1u);
                ASSERT_EQ(colorCount["black"], 2u);
                ASSERT_EQ(colorCount["blue"], 2u);
                colorCount.clear();
            }
        }
    }
}

void verifyBatchAssembler(std::vector<shared_ptr<Shard>> shards, const uint32_t numClasses) {
    srand(time(nullptr));

    uint64_t batchSize = (rand() % 3) + 1;

    BatchAssembler batchAssembler(
        shards,
        ExampleType::TEST,
        ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {3, 224, 224}),
        ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::FP32, {numClasses}),
        batchSize);

    ASSERT_EQ(batchAssembler.getNumBatchesPerEpoch(), (7 + (batchSize - 1)) / batchSize);

    ThorImplementation::Tensor batchTensor;
    ThorImplementation::Tensor labelTensor;
    vector<ThorImplementation::Tensor> batchTensors;
    vector<ThorImplementation::Tensor> labelTensors;
    uint64_t batchNum;

    batchAssembler.getBatch(batchTensor, labelTensor, batchNum);
    do {
        batchTensors.push_back(batchTensor);
        labelTensors.push_back(labelTensor);
        batchAssembler.getBatch(batchTensor, labelTensor, batchNum);
    } while (batchNum != 1);
    do {
        batchTensors.push_back(batchTensor);
        labelTensors.push_back(labelTensor);
        batchAssembler.getBatch(batchTensor, labelTensor, batchNum);
    } while (batchNum != 1);

    verifyBatch(batchSize, batchTensors, labelTensors);

    // Next line just to cover that function, does not return all buffers.
    batchAssembler.returnBuffer(batchTensor, labelTensor);
}

TEST(ShardedRawDatasetCreator, evaluatesDataset) {
    string baseFilename = "testDataset";
    string testDatasetDir("test/DeepLearning/DataSet");

    path tempDirectoryPath = temp_directory_path();
    tempDirectoryPath /= "ThorFrameworkDatasetTest";
    remove_all(tempDirectoryPath);
    create_directory(tempDirectoryPath);

    unordered_set<string> sourceDirectories;
    unordered_set<string> destDirectories;

    sourceDirectories.insert(testDatasetDir);
    destDirectories.insert(tempDirectoryPath.native());

    std::vector<shared_ptr<Shard>> shards;
    ShardedRawDatasetCreator creator(sourceDirectories, destDirectories, baseFilename);
    creator.createDataset(unique_ptr<ImageProcessor>(new ImageProcessor(0.05, 10, 224, 224, 1, false, false)), shards);

    // load the dataset, ensure it contains the expected contents
    ASSERT_EQ(shards.size(), 1u);
    EXPECT_EQ(shards[0]->getNumExamples(ExampleType::TRAIN), 3u);
    EXPECT_EQ(shards[0]->getNumExamples(ExampleType::VALIDATE), 1u);
    EXPECT_EQ(shards[0]->getNumExamples(ExampleType::TEST), 7u);

    uint32_t NUM_CLASSES = 4;
    verifyBatchAssembler(shards, NUM_CLASSES);

    if (rand() % 2 == 0)
        shards.clear();

    Shard reopenedShard;
    path shardPath;
    shardPath = tempDirectoryPath;
    shardPath /= (baseFilename + "_1_of_1.shard");
    reopenedShard.openShard(shardPath.native());
    verifyImages(reopenedShard);

    remove_all(tempDirectoryPath);
}

/*
bool alexnetImagePreprocessorUint8(uint8_t *rgbPixelArray) {
    for (uint32_t row = 0; row < 224; ++row) {
        for (uint32_t col = 0; col < 224; ++col) {
            rgbPixelArray[row * 224 + col] = rgbPixelArray[row * 224 + col] - 124;
            rgbPixelArray[224 * 224 + row * 224 + col] = rgbPixelArray[224 * 224 + row * 224 + col] - 117;
            rgbPixelArray[2 * 224 * 224 + row * 224 + col] = rgbPixelArray[2 * 224 * 224 + row * 224 + col] - 104;
        }
    }
    return true;
}
*/
/*
bool alexnetImagePreprocessorHalf(half *rgbPixelArray) {
    for (uint32_t row = 0; row < 224; ++row) {
        for (uint32_t col = 0; col < 224; ++col) {
            rgbPixelArray[row * 224 + col] = ((float)rgbPixelArray[row * 224 + col] - 124.0f) / 255.0f;
            rgbPixelArray[224 * 224 + row * 224 + col] = ((float)rgbPixelArray[224 * 224 + row * 224 + col] - 117.0f) / 255.0f;
            rgbPixelArray[2 * 224 * 224 + row * 224 + col] = ((float)rgbPixelArray[2 * 224 * 224 + row * 224 + col] - 104.0f) / 255.0f;
        }
    }
    return true;
}
*/
/*
TEST(ShardedRawDatasetCreator, createImagenet) {
    string baseFilename = "ImageNet2012";
    string testDatasetDir("/media/andrew/SSD_Storage/ImageNet_2012");

    unordered_set<string> sourceDirectories;
    unordered_set<string> destDirectories;

    sourceDirectories.insert(testDatasetDir);
    destDirectories.insert("/PCIE_SSD/");

    std::vector<shared_ptr<Shard>> shards;
    ShardedRawDatasetCreator creator(sourceDirectories, destDirectories, baseFilename);
    creator.createDataset(unique_ptr<ImageProcessor>(new ImageProcessor(0.05, 20, 224, 224, alexnetImagePreprocessorHalf)), shards);
}
*/
/*
TEST(ShardedRawDatasetCreator, createImagenet10Classes) {
    string baseFilename = "ImageNet2012";
    string testDatasetDir("/media/andrew-local/SSD_Storage/ImageNet_2012");

    unordered_set<string> sourceDirectories;
    unordered_set<string> destDirectories;

    sourceDirectories.insert(testDatasetDir);
    destDirectories.insert("/PCIE_SSD/");

    std::vector<shared_ptr<Shard>> shards;
    ShardedRawDatasetCreator creator(sourceDirectories, destDirectories, baseFilename, 10);
    creator.createDataset(unique_ptr<ImageProcessor>(new ImageProcessor(0.05, 20, 224, 224, alexnetImagePreprocessorHalf)), shards);
}
*/
/*
class MnistDataProcessor : public DataProcessor {
   public:
    MnistDataProcessor(uint64_t numOutputTensorBytes, ThorImplementation::TensorDescriptor::DataType dataType) {
        this->numOutputTensorBytes = numOutputTensorBytes;
        this->dataType = dataType;
    }

    virtual ~MnistDataProcessor() {}

    virtual uint64_t outputTensorSizeInBytes() { return numOutputTensorBytes; }
    virtual ThorImplementation::TensorDescriptor::DataType getDataType() { return dataType; }

    virtual DataElement operator()(DataElement &input) {
        const uint64_t numElements = outputTensorSizeInBytes() / 4;
        assert(input.numDataBytes == numElements);
        DataElement output = input;
        output.numDataBytes *= 4;
        output.data = shared_ptr<uint8_t>(new uint8_t[numElements * 4]);
        float *outputDataFloat = (float *)output.data.get();
        for (uint32_t i = 0; i < numElements; ++i) {
            outputDataFloat[i] = ((float)(input.data.get()[i]) - 128.0f) / 128.0f;
        }
        return output;
    }

   private:
    uint64_t numOutputTensorBytes;
    ThorImplementation::TensorDescriptor::DataType dataType;
};

TEST(ShardedRawDatasetCreator, createMnist) {
    constexpr uint64_t NUM_GRAYSCALE_PIXELS_PER_IMAGE = 28 * 28;

    string baseFilename = "Mnist";
    string testDatasetDir("/home/andrew/mnist/rawHierarchicallyLabeled");

    unordered_set<string> sourceDirectories;
    unordered_set<string> destDirectories;

    sourceDirectories.insert(testDatasetDir);
    destDirectories.insert("/PCIE_SSD/");

    std::vector<shared_ptr<Shard>> shards;
    ShardedRawDatasetCreator creator(sourceDirectories, destDirectories, baseFilename);
    creator.createDataset(unique_ptr<MnistDataProcessor>(new MnistDataProcessor(NUM_GRAYSCALE_PIXELS_PER_IMAGE * sizeof(float),
                                                                                ThorImplementation::TensorDescriptor::DataType::FP32)),
                          shards);
}
*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
