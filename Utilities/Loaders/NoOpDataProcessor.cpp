#include "Utilities/Loaders/NoOpDataProcessor.h"

NoOpDataProcessor::NoOpDataProcessor(uint64_t numOutputTensorBytes, ThorImplementation::TensorDescriptor::DataType dataType) {
    this->numOutputTensorBytes = numOutputTensorBytes;
    this->dataType = dataType;
}

uint64_t NoOpDataProcessor::outputTensorSizeInBytes() { return numOutputTensorBytes; }

ThorImplementation::TensorDescriptor::DataType NoOpDataProcessor::getDataType() { return dataType; }

DataElement NoOpDataProcessor::operator()(DataElement &input) {
    assert(input.numDataBytes == outputTensorSizeInBytes());
    return input;
}