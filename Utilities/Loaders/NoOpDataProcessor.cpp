#include "Utilities/Loaders/NoOpDataProcessor.h"
#include "DeepLearning/Implementation/ThorError.h"

NoOpDataProcessor::NoOpDataProcessor(uint64_t numOutputTensorBytes, ThorImplementation::DataType dataType) {
    this->numOutputTensorBytes = numOutputTensorBytes;
    this->dataType = dataType;
}

uint64_t NoOpDataProcessor::outputTensorSizeInBytes() { return numOutputTensorBytes; }

ThorImplementation::DataType NoOpDataProcessor::getDataType() { return dataType; }

DataElement NoOpDataProcessor::operator()(DataElement &input) {
    THOR_THROW_IF_FALSE(input.numDataBytes == outputTensorSizeInBytes());
    return input;
}