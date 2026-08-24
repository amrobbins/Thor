#include "DeepLearning/Api/Training/DatasetInputBindings.h"

#include "DeepLearning/Api/Data/NamedDataset.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string_view>
#include <utility>
#include <vector>

using namespace Thor;
using ThorImplementation::DataType;

namespace {

class RequirementOnlyBatchSession final : public BatchSession {
   public:
    RequirementOnlyBatchSession(uint64_t batchSize,
                                DatasetFieldMaterializationRequirements requirements)
        : requirements(std::move(requirements)) {
        this->batchSize = batchSize;
    }

    uint64_t getNumBatchesPerEpoch(ExampleType) override { return 0; }
    uint64_t getNumExamples(ExampleType) override { return 0; }
    uint64_t getNextBatchNum(ExampleType) override { return 0; }
    const DatasetFieldMaterializationRequirements& getDatasetFieldMaterializationRequirements() const override {
        return requirements;
    }

   private:
    Batch acquireBatch(ExampleType, uint64_t&) override { return {}; }
    void recycleBatch(ExampleType, Batch&&) override {}

    DatasetFieldMaterializationRequirements requirements;
};

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
        const BatchPolicy &batching,
        const DatasetAccessPolicy &,
        uint64_t,
        const DatasetFieldMaterializationRequirements &requirements) const override {
        return std::make_shared<RequirementOnlyBatchSession>(batching.getBatchSize(), requirements);
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
    EXPECT_EQ(datasetFieldIds(compiled.fieldRequirements), (std::set<DatasetFieldId>{1, 2}));
}

TEST(DatasetInputBindings, ExactNameAutobindingIsStrictAndComplete) {
    auto dataset = makeDataset();
    Network network("exact-name-dataset-bindings");
    NetworkInput::Builder().network(network).name("history").dimensions({3, 1}).dataType(DataType::FP32).build();
    NetworkInput::Builder().network(network).name("labels").dimensions({1}).dataType(DataType::FP32).build();

    DatasetInputBindings bindings = DatasetInputBindings::byExactName(network, *dataset);
    EXPECT_EQ(bindings.size(), 2u);
    EXPECT_NO_THROW(static_cast<void>(bindings.compile(network, *dataset, 4)));

    Network mismatched("missing-exact-name");
    NetworkInput::Builder().network(mismatched).name("renamed_history").dimensions({3, 1}).dataType(DataType::FP32).build();
    EXPECT_THROW(static_cast<void>(DatasetInputBindings::byExactName(mismatched, *dataset)), std::runtime_error);
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
    EXPECT_THROW(static_cast<void>(dtypeBindings.compile(dtypeNetwork, *dataset, 4)), std::runtime_error);

    Network shapeNetwork("shape-mismatch");
    NetworkInput shapeInput = NetworkInput::Builder()
                                  .network(shapeNetwork)
                                  .name("history")
                                  .dimensions({3})
                                  .dataType(DataType::FP32)
                                  .build();
    DatasetInputBindings shapeBindings;
    shapeBindings.bind(shapeInput, dataset->getField("history"));
    EXPECT_THROW(static_cast<void>(shapeBindings.compile(shapeNetwork, *dataset, 4)), std::runtime_error);

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
    EXPECT_THROW(static_cast<void>(missingBindings.compile(missingNetwork, *dataset, 4)), std::runtime_error);

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
    EXPECT_THROW(static_cast<void>(foreignBindings.compile(foreignNetwork, *dataset, 4)), std::runtime_error);
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
    EXPECT_NO_THROW(static_cast<void>(bindings.compile(network, *dataset, 4)));
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

    EXPECT_NO_THROW(static_cast<void>(bindings.compile(network, *dataset, 4)));
    EXPECT_THROW(static_cast<void>(bindings.compile(network, *dataset, 2)), std::runtime_error);
}

TEST(DatasetInputBindings, CompileByNameConsumesOnlyNetworkInputSubset) {
    auto dataset = makeDataset();
    Network network("phase-subset-bindings");
    NetworkInput::Builder().network(network).name("labels").dimensions({1}).dataType(DataType::FP32).build();

    CompiledDatasetInputBindings compiled = DatasetInputBindings::compileByName(network, *dataset, 4);
    ASSERT_EQ(compiled.trainingInputBindings.size(), 1u);
    EXPECT_EQ(compiled.trainingInputBindings.front().getNetworkInputName(), "labels");
    EXPECT_EQ(compiled.trainingInputBindings.front().getBatchInputName(), "labels");
    EXPECT_EQ(datasetFieldIds(compiled.fieldRequirements), (std::set<DatasetFieldId>{2}));
}

TEST(DatasetInputBindings, CompileByNameRejectsInputMissingFromDataset) {
    auto dataset = makeDataset();
    Network network("phase-missing-binding");
    NetworkInput::Builder().network(network).name("unknown").dimensions({1}).dataType(DataType::FP32).build();

    EXPECT_THROW(static_cast<void>(DatasetInputBindings::compileByName(network, *dataset, 4)), std::runtime_error);
}


TEST(DatasetInputBindings, RaggedLogicalInputBindsOneDatasetFieldAndCarriesMaterializationContract) {
    auto dataset = std::make_shared<InMemoryNamedDataset>(std::vector<DatasetField>{
        DatasetField{.id = 7,
                     .name = "labels",
                     .dataType = DataType::INT32,
                     .dimensions = {},
                     .kind = DatasetFieldKind::RAGGED},
    });
    Network network("ragged-dataset-binding");
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(DataType::INT32)
                              .offsetsDataType(DataType::UINT64)
                              .trailingDimensions({})
                              .maxTotalValues(23)
                              .maxValuesPerRow(7)
                              .batchSize(4)
                              .build();

    DatasetInputBindings exact = DatasetInputBindings::byExactName(network, *dataset);
    EXPECT_EQ(exact.size(), 1u);
    CompiledDatasetInputBindings compiled = exact.compile(network, *dataset, 4);
    ASSERT_EQ(compiled.trainingInputBindings.size(), 1u);
    EXPECT_EQ(compiled.trainingInputBindings.front().getNetworkInputName(), "labels");
    EXPECT_EQ(compiled.trainingInputBindings.front().getBatchInputName(), "labels");
    ASSERT_EQ(compiled.fieldRequirements.size(), 1u);
    const DatasetFieldMaterializationRequirement& requirement = compiled.fieldRequirements.at(7);
    ASSERT_TRUE(requirement.raggedTensorDescriptor.has_value());
    EXPECT_EQ(requirement.raggedTensorDescriptor.value(), labels.getDescriptor());
    EXPECT_EQ(requirement.raggedTensorDescriptor->getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(requirement.raggedTensorDescriptor->getMaxTotalValues(), 23u);
    ASSERT_TRUE(requirement.raggedTensorDescriptor->hasMaxValuesPerRow());
    EXPECT_EQ(requirement.raggedTensorDescriptor->getMaxValuesPerRow(), 7u);
    EXPECT_EQ(requirement.raggedTensorDescriptor->getBatchSize(), 4u);
}

TEST(DatasetInputBindings, RaggedExplicitBindingUsesLogicalNameAndRejectsPhysicalHalfwayBindings) {
    auto dataset = std::make_shared<InMemoryNamedDataset>(std::vector<DatasetField>{
        DatasetField{.id = 7,
                     .name = "targets",
                     .dataType = DataType::FP32,
                     .dimensions = {3},
                     .kind = DatasetFieldKind::RAGGED},
    });
    Network network("ragged-explicit-binding");
    RaggedTensor sequences = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("sequence_input")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({3})
                                 .maxTotalValues(32)
                                 .batchSize(4)
                                 .build();

    DatasetInputBindings bindings;
    bindings.bind(network, sequences, dataset->getField("targets"));
    CompiledDatasetInputBindings compiled = bindings.compile(network, *dataset, 4);
    ASSERT_EQ(compiled.trainingInputBindings.size(), 1u);
    EXPECT_EQ(compiled.trainingInputBindings.front().getNetworkInputName(), "sequence_input");
    EXPECT_EQ(compiled.trainingInputBindings.front().getBatchInputName(), "targets");

    CompiledDatasetInputBindings byName = DatasetInputBindings::compileByName(
        network,
        *dataset,
        4,
        {TrainingInputBinding("sequence_input", "targets")});
    EXPECT_EQ(byName, compiled);

    EXPECT_THROW(
        static_cast<void>(DatasetInputBindings::compileByName(
            network,
            *dataset,
            4,
            {TrainingInputBinding("sequence_input.values", "targets")})),
        std::runtime_error);

    DatasetInputBindings legacyPhysical;
    const auto physicalInputs = network.getExternalNetworkInputs();
    auto valuesIt = std::find_if(physicalInputs.begin(), physicalInputs.end(), [](const auto& input) {
        return input->getName() == "sequence_input.values";
    });
    ASSERT_NE(valuesIt, physicalInputs.end());
    legacyPhysical.bind(**valuesIt, dataset->getField("targets"));
    EXPECT_THROW(static_cast<void>(legacyPhysical.compile(network, *dataset, 4)), std::runtime_error);
}

TEST(DatasetInputBindings, RaggedBindingRejectsKindDtypeShapeAndBatchMismatches) {
    Network network("ragged-contract-mismatches");
    RaggedTensor sequences = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("sequences")
                                 .valuesDataType(DataType::BF16)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({5})
                                 .maxTotalValues(20)
                                 .batchSize(4)
                                 .build();

    auto denseKind = std::make_shared<InMemoryNamedDataset>(std::vector<DatasetField>{
        DatasetField{.id = 1, .name = "sequences", .dataType = DataType::BF16, .dimensions = {5}, .kind = DatasetFieldKind::DENSE}});
    EXPECT_THROW(static_cast<void>(DatasetInputBindings::byExactName(network, *denseKind).compile(network, *denseKind, 4)),
                 std::runtime_error);

    auto wrongDtype = std::make_shared<InMemoryNamedDataset>(std::vector<DatasetField>{
        DatasetField{.id = 1, .name = "sequences", .dataType = DataType::FP32, .dimensions = {5}, .kind = DatasetFieldKind::RAGGED}});
    EXPECT_THROW(static_cast<void>(DatasetInputBindings::byExactName(network, *wrongDtype).compile(network, *wrongDtype, 4)),
                 std::runtime_error);

    auto wrongShape = std::make_shared<InMemoryNamedDataset>(std::vector<DatasetField>{
        DatasetField{.id = 1, .name = "sequences", .dataType = DataType::BF16, .dimensions = {6}, .kind = DatasetFieldKind::RAGGED}});
    EXPECT_THROW(static_cast<void>(DatasetInputBindings::byExactName(network, *wrongShape).compile(network, *wrongShape, 4)),
                 std::runtime_error);

    auto matching = std::make_shared<InMemoryNamedDataset>(std::vector<DatasetField>{
        DatasetField{.id = 1, .name = "sequences", .dataType = DataType::BF16, .dimensions = {5}, .kind = DatasetFieldKind::RAGGED}});
    EXPECT_THROW(static_cast<void>(DatasetInputBindings::byExactName(network, *matching).compile(network, *matching, 2)),
                 std::runtime_error);
}


TEST(DatasetInputBindings, TrainingDataPropagatesRaggedMaterializationRequirementWithoutGuessingCapacity) {
    auto dataset = std::make_shared<InMemoryNamedDataset>(std::vector<DatasetField>{
        DatasetField{.id = 7,
                     .name = "labels",
                     .dataType = DataType::INT32,
                     .dimensions = {},
                     .kind = DatasetFieldKind::RAGGED},
    });
    Network network("ragged-training-data-requirement");
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(DataType::INT32)
                              .offsetsDataType(DataType::UINT64)
                              .trailingDimensions({})
                              .maxTotalValues(41)
                              .batchSize(4)
                              .build();
    (void)labels;
    CompiledDatasetInputBindings compiled = DatasetInputBindings::compileByName(network, *dataset, 4);

    TrainingData data(dataset,
                      DatasetSplitManifest(*dataset, {0, 1, 2, 3}, {4, 5, 6, 7}),
                      BatchPolicy(4, false));
    EXPECT_THROW(static_cast<void>(data.openSession(2)), std::runtime_error);
    std::shared_ptr<BatchSession> session = data.openSession(2, compiled.fieldRequirements);
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(session->getDatasetFieldMaterializationRequirements(), compiled.fieldRequirements);
    EXPECT_EQ(session->getRequiredDatasetFieldIds(), (std::set<DatasetFieldId>{7}));
    ASSERT_TRUE(session->getDatasetFieldMaterializationRequirements().at(7).raggedTensorDescriptor.has_value());
    EXPECT_EQ(session->getDatasetFieldMaterializationRequirements().at(7).raggedTensorDescriptor->getMaxTotalValues(), 41u);
}
