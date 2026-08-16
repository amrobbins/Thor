#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"

#include <sstream>

namespace ThorImplementation {

std::string RowPartitionDescriptor::toString() const {
    std::ostringstream out;
    out << "RowPartitionDescriptor(batch_size=" << batchSize << ", max_total_values=" << maxTotalValues
        << ", offsets_data_type=" << TensorDescriptor::getElementTypeName(offsetsDataType) << ")";
    return out.str();
}

}  // namespace ThorImplementation
