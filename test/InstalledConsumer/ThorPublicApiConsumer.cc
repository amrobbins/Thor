#include "Thor.h"

#include <type_traits>

static_assert(std::is_class_v<DatasetLayout>);
static_assert(std::is_class_v<DatasetWriter>);
static_assert(std::is_class_v<Thor::TrainingData>);
static_assert(std::is_class_v<Thor::Trainer>);
static_assert(std::is_base_of_v<Thor::NamedDataset, Thor::FileDataset>);

void thorInstalledPublicApiConsumerCompileSmoke() {}
