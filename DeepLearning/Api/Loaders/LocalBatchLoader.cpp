#include "DeepLearning/Api/Loaders/LocalBatchLoader.h"

using ThorImplementation::TensorDescriptor;

using std::make_shared;
using std::string;
using ThorImplementation::Tensor;

LocalBatchLoader::LocalBatchLoader(set<string> shardPaths,
                                   ThorImplementation::TensorDescriptor exampleDescriptor,
                                   ThorImplementation::TensorDescriptor labelDescriptor,
                                   uint64_t batchSize) {
    this->batchSize = batchSize;

    uint64_t exampleSizeInBytes = exampleDescriptor.getArraySizeInBytes();

    for (auto it = shardPaths.begin(); it != shardPaths.end(); ++it) {
        shards.push_back(make_shared<Shard>());
        string shardPath = *it;
        shards.back()->openShard(shardPath);
        assert(shards.back()->getExampleSizeInBytes() == exampleSizeInBytes);
        assert(shards.back()->getDataType() == exampleDescriptor.getDataType());
    }

    batchAssemblerTrain = make_shared<BatchAssembler>(
        shards,
        ExampleType::TRAIN,
        ThorImplementation::TensorDescriptor(exampleDescriptor.getDataType(), exampleDescriptor.getDimensions()),
        ThorImplementation::TensorDescriptor(labelDescriptor.getDataType(), labelDescriptor.getDimensions()),
        batchSize);
    batchAssemblerValidate = make_shared<BatchAssembler>(
        shards,
        ExampleType::VALIDATE,
        ThorImplementation::TensorDescriptor(exampleDescriptor.getDataType(), exampleDescriptor.getDimensions()),
        ThorImplementation::TensorDescriptor(labelDescriptor.getDataType(), labelDescriptor.getDimensions()),
        batchSize);
    batchAssemblerTest = make_shared<BatchAssembler>(
        shards,
        ExampleType::TEST,
        ThorImplementation::TensorDescriptor(exampleDescriptor.getDataType(), exampleDescriptor.getDimensions()),
        ThorImplementation::TensorDescriptor(labelDescriptor.getDataType(), labelDescriptor.getDimensions()),
        batchSize);
}

uint64_t LocalBatchLoader::getNextBatchNum(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN)
        return batchAssemblerTrain->getNextBatchNum();
    else if (exampleType == ExampleType::VALIDATE)
        return batchAssemblerValidate->getNextBatchNum();
    else if (exampleType == ExampleType::TEST)
        return batchAssemblerTest->getNextBatchNum();
    else
        assert(false);
}

uint64_t LocalBatchLoader::getNumBatchesPerEpoch(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN)
        return batchAssemblerTrain->getNumBatchesPerEpoch();
    else if (exampleType == ExampleType::VALIDATE)
        return batchAssemblerValidate->getNumBatchesPerEpoch();
    else if (exampleType == ExampleType::TEST)
        return batchAssemblerTest->getNumBatchesPerEpoch();
    else
        assert(false);
}

uint64_t LocalBatchLoader::getNumExamples(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN)
        return batchAssemblerTrain->getNumExamples();
    else if (exampleType == ExampleType::VALIDATE)
        return batchAssemblerValidate->getNumExamples();
    else if (exampleType == ExampleType::TEST)
        return batchAssemblerTest->getNumExamples();
    else
        assert(false);
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
        assert(false);
    }

    return tensorMap;
}

void LocalBatchLoader::returnBatchBuffers(ExampleType exampleType, map<std::string, Tensor> tensorMap) {
    assert(tensorMap.count("examples") == 1);
    assert(tensorMap.count("labels") == 1);

    if (exampleType == ExampleType::TRAIN) {
        batchAssemblerTrain->returnBuffer(tensorMap["examples"], tensorMap["labels"]);
    } else if (exampleType == ExampleType::VALIDATE) {
        batchAssemblerValidate->returnBuffer(tensorMap["examples"], tensorMap["labels"]);
    } else if (exampleType == ExampleType::TEST) {
        batchAssemblerTest->returnBuffer(tensorMap["examples"], tensorMap["labels"]);
    } else {
        assert(false);
    }
}
