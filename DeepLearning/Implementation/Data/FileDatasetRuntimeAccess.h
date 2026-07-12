#pragma once

#include "DeepLearning/Api/Data/DatasetLayout.h"

#include <memory>

class IndexedLocalNamedExampleReader;

namespace Thor {
class FileDataset;
}

namespace ThorImplementation {

/** Internal bridge to FileDataset physical storage details. */
class FileDatasetRuntimeAccess {
   public:
    [[nodiscard]] static const DatasetLayout &layout(const Thor::FileDataset &dataset);
    [[nodiscard]] static const std::shared_ptr<IndexedLocalNamedExampleReader> &reader(
        const Thor::FileDataset &dataset);
};

}  // namespace ThorImplementation
