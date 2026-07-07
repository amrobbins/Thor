#include "DeepLearning/Api/Loaders/DeviceDatasetMaterialization.h"

const char *deviceDatasetMaterializationSplitName(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN) {
        return "train";
    }
    if (exampleType == ExampleType::VALIDATE) {
        return "validate";
    }
    if (exampleType == ExampleType::TEST) {
        return "test";
    }
    return "unknown";
}

const DeviceDatasetMaterializationSplitView *DeviceDatasetMaterializationView::findSplit(ExampleType exampleType) const {
    for (const DeviceDatasetMaterializationSplitView &candidate : splits) {
        if (candidate.exampleType == exampleType) {
            return &candidate;
        }
    }
    return nullptr;
}

const DeviceDatasetMaterializationSplitView &DeviceDatasetMaterializationView::split(ExampleType exampleType) const {
    const DeviceDatasetMaterializationSplitView *candidate = findSplit(exampleType);
    if (candidate == nullptr) {
        throw std::runtime_error(std::string("Device dataset materialization view is missing split: ") +
                                 deviceDatasetMaterializationSplitName(exampleType));
    }
    return *candidate;
}
