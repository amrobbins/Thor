#pragma once

#include "Utilities/Loaders/Shard.h"

#include <vector>

struct ExecutionState {
    std::string networkName;
    std::string datasetName;

    std::string outputDirectory;

    // Note: All fields pertain to the specific exectution mode only,
    // i.e. if training is occuring then all items pertain to training,
    // if validation is occuring then all items pertain to validation.
    ExampleType executionMode;
    uint64_t epochsToTrain;
    uint64_t epochNum;
    uint64_t batchNum;
    uint64_t batchSize;
    uint64_t batchesPerEpoch;

    float learningRate;
    float momentum;

    uint64_t numTrainingExamples;
    uint64_t numValidationExamples;
    uint64_t numTestExamples;

    double runningAverageTimePerTrainingBatch;
    double runningAverageTimePerValidationBatch;

    double batchLoss;
    std::vector<float> lossPerExamplePerClass;
    double batchAccuracy;

    uint64_t flopsPerExample;
};
