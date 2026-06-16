#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Loaders/LocalBatchLoader.h"

using namespace std;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;

LocalBatchLoader::LocalBatchLoader(set<string> shardPaths,
                                   ThorImplementation::TensorDescriptor exampleDescriptor,
                                   ThorImplementation::TensorDescriptor labelDescriptor,
                                   uint64_t batchSize,
                                   uint64_t batchQueueDepth) {
    this->batchSize = batchSize;

    const uint64_t exampleSizeInBytes = exampleDescriptor.getArraySizeInBytes();
    const uint64_t labelSizeInBytes = labelDescriptor.getArraySizeInBytes();
    const bool canUseInlinePayloadLabels = exampleDescriptor.getDataType() == labelDescriptor.getDataType();

    for (auto it = shardPaths.begin(); it != shardPaths.end(); ++it) {
        shards.push_back(make_shared<Shard>());
        string shardPath = *it;
        shards.back()->openShard(shardPath);
        const uint64_t shardRecordSizeInBytes = shards.back()->getExampleSizeInBytes();
        const bool classMetadataLabels = shardRecordSizeInBytes == exampleSizeInBytes;
        const bool inlinePayloadLabels = canUseInlinePayloadLabels &&
                                         shardRecordSizeInBytes == exampleSizeInBytes + labelSizeInBytes;
        THOR_THROW_IF_FALSE(classMetadataLabels || inlinePayloadLabels);
        THOR_THROW_IF_FALSE(shards.back()->getDataType() == exampleDescriptor.getDataType());
    }

    batchAssemblerTrain = make_shared<BatchAssembler>(
        shards,
        ExampleType::TRAIN,
        ThorImplementation::TensorDescriptor(exampleDescriptor.getDataType(), exampleDescriptor.getDimensions()),
        ThorImplementation::TensorDescriptor(labelDescriptor.getDataType(), labelDescriptor.getDimensions()),
        batchSize,
        batchQueueDepth);
    batchAssemblerValidate = make_shared<BatchAssembler>(
        shards,
        ExampleType::VALIDATE,
        ThorImplementation::TensorDescriptor(exampleDescriptor.getDataType(), exampleDescriptor.getDimensions()),
        ThorImplementation::TensorDescriptor(labelDescriptor.getDataType(), labelDescriptor.getDimensions()),
        batchSize,
        batchQueueDepth);
    batchAssemblerTest = make_shared<BatchAssembler>(
        shards,
        ExampleType::TEST,
        ThorImplementation::TensorDescriptor(exampleDescriptor.getDataType(), exampleDescriptor.getDimensions()),
        ThorImplementation::TensorDescriptor(labelDescriptor.getDataType(), labelDescriptor.getDimensions()),
        batchSize,
        batchQueueDepth);
}

uint64_t LocalBatchLoader::getNextBatchNum(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN)
        return batchAssemblerTrain->getNextBatchNum();
    else if (exampleType == ExampleType::VALIDATE)
        return batchAssemblerValidate->getNextBatchNum();
    else if (exampleType == ExampleType::TEST)
        return batchAssemblerTest->getNextBatchNum();
    else
        THOR_UNREACHABLE();
}

uint64_t LocalBatchLoader::getNumBatchesPerEpoch(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN)
        return batchAssemblerTrain->getNumBatchesPerEpoch();
    else if (exampleType == ExampleType::VALIDATE)
        return batchAssemblerValidate->getNumBatchesPerEpoch();
    else if (exampleType == ExampleType::TEST)
        return batchAssemblerTest->getNumBatchesPerEpoch();
    else
        THOR_UNREACHABLE();
}

uint64_t LocalBatchLoader::getNumExamples(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN)
        return batchAssemblerTrain->getNumExamples();
    else if (exampleType == ExampleType::VALIDATE)
        return batchAssemblerValidate->getNumExamples();
    else if (exampleType == ExampleType::TEST)
        return batchAssemblerTest->getNumExamples();
    else
        THOR_UNREACHABLE();
}

map<string, Tensor> LocalBatchLoader::getBatch(ExampleType exampleType, uint64_t &batchNum) {
    map<string, Tensor> tensorMap;

    if (exampleType == ExampleType::TRAIN) {
        batchAssemblerTrain->getBatch(tensorMap["examples"], tensorMap["labels"], batchNum);
    } else if (exampleType == ExampleType::VALIDATE) {
        batchAssemblerValidate->getBatch(tensorMap["examples"], tensorMap["labels"], batchNum);
    } else if (exampleType == ExampleType::TEST) {
        batchAssemblerTest->getBatch(tensorMap["examples"], tensorMap["labels"], batchNum);
    } else {
        THOR_UNREACHABLE();
    }

    return tensorMap;
}

void LocalBatchLoader::returnBatchBuffers(ExampleType exampleType, map<std::string, Tensor>&& tensorMap) {
    THOR_THROW_IF_FALSE(tensorMap.count("examples") == 1);
    THOR_THROW_IF_FALSE(tensorMap.count("labels") == 1);

    if (exampleType == ExampleType::TRAIN) {
        batchAssemblerTrain->returnBuffer(tensorMap["examples"], tensorMap["labels"]);
    } else if (exampleType == ExampleType::VALIDATE) {
        batchAssemblerValidate->returnBuffer(tensorMap["examples"], tensorMap["labels"]);
    } else if (exampleType == ExampleType::TEST) {
        batchAssemblerTest->returnBuffer(tensorMap["examples"], tensorMap["labels"]);
    } else {
        THOR_UNREACHABLE();
    }
}
