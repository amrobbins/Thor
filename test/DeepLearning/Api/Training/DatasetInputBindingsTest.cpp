#include "DeepLearning/Api/Training/DatasetInputBindings.h"

#include "DeepLearning/Api/Data/NamedDataset.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <string_view>
#include <utility>
#include <vector>

using namespace Thor;
using ThorImplementation::DataType;

namespace {

class InMemoryNamedDataset final : public NamedDataset {
   public:
    explicit InMemoryNamedDataset(std::vector<DatasetField> fields)
        : id(DatasetId::generate()), schema(std::move(fields)) {}

    const DatasetId &getId() const override { return id; }
    uint64_t getNumExamples() const override { return 8; }
    const DatasetSchema &getSchema() const override { return schema; }
    const DatasetField &getField(std::string_view name) const override {
        return schema.getField(name);
    }

   protected:
    std::shared_ptr<BatchSession> openBatchSession(
        const DatasetSplitManifest &,
        const BatchPolicy &,
        const DatasetAccessPolicy &,
        uint64_t,
        const std::set<DatasetFieldId> &) const override {
        return nullptr;
    }

   private:
    DatasetId id;
    DatasetSchema schema;
};

std::shared_ptr<InMemoryNamedDataset> makeDataset() {
    return std::make_shared<InMemoryNamedDataset>(std::vector<DatasetField>{
        DatasetField{.id = 1,
                     .name = "history",
                     .dataType = DataType::FP32,
                     .dimensions = {3, 1},
                     .kind = DatasetFieldKind::WINDOWED},
        DatasetField{.id = 2,
                     .name = "labels",
                     .dataType = DataType::FP32,
                     .dimensions = {1},
                     .kind = DatasetFieldKind::DENSE},
    });
}

}  // namespace

TEST(DatasetInputBindings, ExplicitBindingsCompileWithoutChangingFieldContracts) {
    auto dataset = makeDataset();
    Network network("explicit-dataset-bindings");
    NetworkInput history = NetworkInput::Builder()
                               .network(network)
                               .name("history_input")
                               .dimensions({3, 1})
                               .dataType(DataType::FP32)
                               .build();
    NetworkInput labels = NetworkInput::Builder()
                              .network(network)
                              .name("target_input")
                              .dimensions({1})
                              .dataType(DataType::FP32)
                              .build();

    DatasetInputBindings bindings;
    bindings.bind(history, dataset->getField("history"))
        .bind(labels, dataset->getField("labels"));

    CompiledDatasetInputBindings compiled = bindings.compile(network, *dataset, 4);
    ASSERT_EQ(compiled.trainingInputBindings.size(), 2u);
    EXPECT_EQ(compiled.trainingInputBindings[0].getNetworkInputName(), "history_input");
    EXPECT_EQ(compiled.trainingInputBindings[0].getBatchInputName(), "history");
    EXPECT_EQ(compiled.trainingInputBindings[1].getNetworkInputName(), "target_input");
    EXPECT_EQ(compiled.trainingInputBindings[1].getBatchInputName(), "labels");
    EXPECT_EQ(compiled.requiredFieldIds, (std::set<DatasetFieldId>{1, 2}));
}

TEST(DatasetInputBindings, ExactNameAutobindingIsStrictAndComplete) {
    auto dataset = makeDataset();
    Network network("exact-name-dataset-bindings");
    NetworkInput::Builder().network(network).name("history").dimensions({3, 1}).dataType(DataType::FP32).build();
    NetworkInput::Builder().network(network).name("labels").dimensions({1}).dataType(DataType::FP32).build();

    DatasetInputBindings bindings = DatasetInputBindings::byExactName(network, *dataset);
    EXPECT_EQ(bindings.size(), 2u);
    EXPECT_NO_THROW(bindings.compile(network, *dataset, 4));

    Network mismatched("missing-exact-name");
    NetworkInput::Builder().network(mismatched).name("renamed_history").dimensions({3, 1}).dataType(DataType::FP32).build();
    EXPECT_THROW(DatasetInputBindings::byExactName(mismatched, *dataset), std::runtime_error);
}

TEST(DatasetInputBindings, RejectsDtypeShapeMissingAndForeignFieldContracts) {
    auto dataset = makeDataset();

    Network dtypeNetwork("dtype-mismatch");
    NetworkInput dtypeInput = NetworkInput::Builder()
                                  .network(dtypeNetwork)
                                  .name("history")
                                  .dimensions({3, 1})
                                  .dataType(DataType::FP16)
                                  .build();
    DatasetInputBindings dtypeBindings;
    dtypeBindings.bind(dtypeInput, dataset->getField("history"));
    EXPECT_THROW(dtypeBindings.compile(dtypeNetwork, *dataset, 4), std::runtime_error);

    Network shapeNetwork("shape-mismatch");
    NetworkInput shapeInput = NetworkInput::Builder()
                                  .network(shapeNetwork)
                                  .name("history")
                                  .dimensions({3})
                                  .dataType(DataType::FP32)
                                  .build();
    DatasetInputBindings shapeBindings;
    shapeBindings.bind(shapeInput, dataset->getField("history"));
    EXPECT_THROW(shapeBindings.compile(shapeNetwork, *dataset, 4), std::runtime_error);

    Network missingNetwork("missing-binding");
    NetworkInput history = NetworkInput::Builder()
                               .network(missingNetwork)
                               .name("history")
                               .dimensions({3, 1})
                               .dataType(DataType::FP32)
                               .build();
    NetworkInput::Builder().network(missingNetwork).name("labels").dimensions({1}).dataType(DataType::FP32).build();
    DatasetInputBindings missingBindings;
    missingBindings.bind(history, dataset->getField("history"));
    EXPECT_THROW(missingBindings.compile(missingNetwork, *dataset, 4), std::runtime_error);

    auto otherDataset = std::make_shared<InMemoryNamedDataset>(std::vector<DatasetField>{
        DatasetField{.id = 99,
                     .name = "history",
                     .dataType = DataType::FP32,
                     .dimensions = {3, 1},
                     .kind = DatasetFieldKind::WINDOWED}});
    Network foreignNetwork("foreign-field");
    NetworkInput foreignInput = NetworkInput::Builder()
                                    .network(foreignNetwork)
                                    .name("history")
                                    .dimensions({3, 1})
                                    .dataType(DataType::FP32)
                                    .build();
    DatasetInputBindings foreignBindings;
    foreignBindings.bind(foreignInput, otherDataset->getField("history"));
    EXPECT_THROW(foreignBindings.compile(foreignNetwork, *dataset, 4), std::runtime_error);
}

TEST(DatasetInputBindings, RejectsDuplicateInputsAndFields) {
    auto dataset = makeDataset();
    Network network("duplicate-bindings");
    NetworkInput first = NetworkInput::Builder()
                             .network(network)
                             .name("first")
                             .dimensions({3, 1})
                             .dataType(DataType::FP32)
                             .build();
    NetworkInput second = NetworkInput::Builder()
                              .network(network)
                              .name("second")
                              .dimensions({3, 1})
                              .dataType(DataType::FP32)
                              .build();

    DatasetInputBindings duplicateInput;
    duplicateInput.bind(first, dataset->getField("history"));
    EXPECT_THROW(duplicateInput.bind(first, dataset->getField("labels")), std::runtime_error);

    DatasetInputBindings duplicateField;
    duplicateField.bind(first, dataset->getField("history"));
    EXPECT_THROW(duplicateField.bind(second, dataset->getField("history")), std::runtime_error);
}

TEST(DatasetInputBindings, GraphTypeConversionIsExplicitAndSupported) {
    auto dataset = makeDataset();
    Network network("graph-type-conversion");
    NetworkInput history = NetworkInput::Builder()
                               .network(network)
                               .name("history")
                               .dimensions({3, 1})
                               .dataType(DataType::FP32)
                               .build();
    TypeConverter converted = TypeConverter::Builder()
                                  .network(network)
                                  .featureInput(history.getFeatureOutput().value())
                                  .newDataType(DataType::FP16)
                                  .build();
    ASSERT_EQ(converted.getFeatureOutput().value().getDataType(), DataType::FP16);

    DatasetInputBindings bindings;
    bindings.bind(history, dataset->getField("history"));

    NetworkInput labels = NetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .dimensions({1})
                              .dataType(DataType::FP32)
                              .build();
    bindings.bind(labels, dataset->getField("labels"));
    EXPECT_NO_THROW(bindings.compile(network, *dataset, 4));
}

TEST(DatasetInputBindings, ValidatesExplicitBatchDimensionAgainstBatchPolicy) {
    auto dataset = makeDataset();
    Network network("batch-dimension-dataset-bindings");
    NetworkInput history = NetworkInput::Builder()
                               .network(network)
                               .name("history")
                               .dimensions({4, 3, 1})
                               .dimensionsIncludeBatch(true)
                               .dataType(DataType::FP32)
                               .build();
    NetworkInput labels = NetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .dimensions({4, 1})
                              .dimensionsIncludeBatch(true)
                              .dataType(DataType::FP32)
                              .build();

    DatasetInputBindings bindings;
    bindings.bind(history, dataset->getField("history"))
        .bind(labels, dataset->getField("labels"));

    EXPECT_NO_THROW(bindings.compile(network, *dataset, 4));
    EXPECT_THROW(bindings.compile(network, *dataset, 2), std::runtime_error);
}

TEST(DatasetInputBindings, CompileByNameConsumesOnlyNetworkInputSubset) {
    auto dataset = makeDataset();
    Network network("phase-subset-bindings");
    NetworkInput::Builder().network(network).name("labels").dimensions({1}).dataType(DataType::FP32).build();

    CompiledDatasetInputBindings compiled = DatasetInputBindings::compileByName(network, *dataset, 4);
    ASSERT_EQ(compiled.trainingInputBindings.size(), 1u);
    EXPECT_EQ(compiled.trainingInputBindings.front().getNetworkInputName(), "labels");
    EXPECT_EQ(compiled.trainingInputBindings.front().getBatchInputName(), "labels");
    EXPECT_EQ(compiled.requiredFieldIds, (std::set<DatasetFieldId>{2}));
}

TEST(DatasetInputBindings, CompileByNameRejectsInputMissingFromDataset) {
    auto dataset = makeDataset();
    Network network("phase-missing-binding");
    NetworkInput::Builder().network(network).name("unknown").dimensions({1}).dataType(DataType::FP32).build();

    EXPECT_THROW(DatasetInputBindings::compileByName(network, *dataset, 4), std::runtime_error);
}
