#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Parameter/ParameterReference.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Training/StepExecutable.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/TrainingStep.h"
#include "Utilities/Common/Event.h"

#include "gtest/gtest.h"

#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;

namespace {

std::shared_ptr<Initializer> testInitializer() { return UniformRandom::Builder().minValue(-0.01f).maxValue(0.01f).build(); }

FullyConnected buildFullyConnected(Network& network) {
    NetworkInput input = NetworkInput::Builder().network(network).name("input").dimensions({3}).dataType(DataType::FP32).build();
    return FullyConnected::Builder()
        .network(network)
        .featureInput(input.getFeatureOutput().value())
        .numOutputFeatures(2)
        .hasBias(true)
        .weightsInitializer(testInitializer())
        .biasInitializer(testInitializer())
        .noActivation()
        .build();
}

}  // namespace

TEST(TrainingProgramApi, ParameterReferencesIdentifyOwningLayerAndParameterName) {
    Network network("training_program_parameter_refs");
    FullyConnected fc = buildFullyConnected(network);

    std::vector<ParameterReference> refs = fc.getParameterReferences();
    ASSERT_EQ(refs.size(), 2u);

    std::set<std::string> names;
    for (const ParameterReference& ref : refs) {
        EXPECT_TRUE(ref.isInitialized());
        EXPECT_EQ(ref.getParameterizableId(), fc.getId());
        names.insert(ref.getParameterName());
    }

    EXPECT_TRUE(names.contains("weights"));
    EXPECT_TRUE(names.contains("biases"));

    ParameterReference weights = fc.getParameterReference("weights");
    EXPECT_EQ(weights.getParameterizableId(), fc.getId());
    EXPECT_EQ(weights.getParameterName(), "weights");
}

TEST(TrainingProgramApi, NetworkTrainableParameterReferencesRespectFreezeState) {
    Network network("training_program_network_refs");
    FullyConnected fc = buildFullyConnected(network);

    std::vector<ParameterReference> enabledRefs = network.getTrainableParameterReferences();
    ASSERT_EQ(enabledRefs.size(), 2u);

    (void)fc;
    network.freezeTraining();
    EXPECT_TRUE(network.getTrainableParameterReferences().empty());

    std::vector<ParameterReference> allTrainableRefs = network.getTrainableParameterReferences(/*trainingEnabledOnly=*/false);
    ASSERT_EQ(allTrainableRefs.size(), 2u);

    network.unfreezeTraining();
    EXPECT_EQ(network.getTrainableParameterReferences().size(), 2u);
}

TEST(TrainingProgramApi, TrainingStepSerializesLogicalExecutionView) {
    Tensor dLoss(DataType::FP32, {1});
    auto sgd = Sgd::Builder().initialLearningRate(0.01f).build();
    std::vector<ParameterReference> params{ParameterReference(123, "weights"), ParameterReference(123, "biases")};

    TrainingStep step("discriminator", {dLoss}, sgd, params, 2, TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP);

    EXPECT_TRUE(step.isInitialized());
    EXPECT_EQ(step.getName(), "discriminator");
    EXPECT_EQ(step.getRepeatCount(), 2u);
    EXPECT_TRUE(step.updatesParameter(params[0]));
    EXPECT_FALSE(step.updatesParameter(ParameterReference(999, "weights")));

    nlohmann::json j = step.architectureJson();
    EXPECT_EQ(j.at("version").get<std::string>(), "1.0.0");
    EXPECT_EQ(j.at("name").get<std::string>(), "discriminator");
    EXPECT_EQ(j.at("repeat_count").get<uint32_t>(), 2u);
    EXPECT_EQ(j.at("loss_roots").size(), 1u);
    EXPECT_EQ(j.at("update_parameters").size(), 2u);
    EXPECT_TRUE(j.contains("optimizer"));

    TrainingStep restored = TrainingStep::deserialize(j);
    EXPECT_TRUE(restored.isInitialized());
    EXPECT_EQ(restored.getName(), step.getName());
    EXPECT_EQ(restored.getRepeatCount(), step.getRepeatCount());
    EXPECT_EQ(restored.getGradientClearPolicy(), step.getGradientClearPolicy());
    EXPECT_EQ(restored.getLossRoots().size(), step.getLossRoots().size());
    EXPECT_EQ(restored.getLossRoots()[0].getOriginalId(), step.getLossRoots()[0].getOriginalId());
    EXPECT_EQ(restored.getUpdateParameters(), step.getUpdateParameters());
    EXPECT_NE(restored.getOptimizer(), nullptr);
}

TEST(TrainingProgramApi, TrainingStepRejectsDuplicateUpdateParameters) {
    Tensor loss(DataType::FP32, {1});
    auto sgd = Sgd::Builder().initialLearningRate(0.01f).build();
    ParameterReference weights(123, "weights");

    EXPECT_THROW(TrainingStep("bad", {loss}, sgd, {weights, weights}), std::runtime_error);
}

TEST(TrainingProgramApi, TrainingStepRejectsUpdatesWithoutOptimizer) {
    Tensor loss(DataType::FP32, {1});
    ParameterReference weights(123, "weights");

    EXPECT_THROW(TrainingStep("bad", {loss}, nullptr, {weights}), std::runtime_error);
}

TEST(TrainingProgramApi, TrainingProgramKeepsOrderedUniqueSteps) {
    Tensor dLoss(DataType::FP32, {1});
    Tensor gLoss(DataType::FP32, {1});
    auto dSgd = Sgd::Builder().initialLearningRate(0.01f).build();
    auto gSgd = Sgd::Builder().initialLearningRate(0.02f).build();

    TrainingStep dStep("discriminator", {dLoss}, dSgd, {ParameterReference(1, "weights")});
    TrainingStep gStep("generator", {gLoss}, gSgd, {ParameterReference(2, "weights")});

    TrainingProgram program;
    EXPECT_FALSE(program.isInitialized());
    program.addStep(dStep);
    program.addStep(gStep);

    EXPECT_TRUE(program.isInitialized());
    EXPECT_EQ(program.getNumSteps(), 2u);
    EXPECT_EQ(program.getStep(0).getName(), "discriminator");
    EXPECT_EQ(program.getStep(1).getName(), "generator");

    nlohmann::json j = program.architectureJson();
    ASSERT_EQ(j.at("steps").size(), 2u);
    EXPECT_EQ(j.at("steps").at(0).at("name").get<std::string>(), "discriminator");
    EXPECT_EQ(j.at("steps").at(1).at("name").get<std::string>(), "generator");

    TrainingProgram restored = TrainingProgram::deserialize(j);
    ASSERT_EQ(restored.getNumSteps(), 2u);
    EXPECT_EQ(restored.getStep(0).getName(), "discriminator");
    EXPECT_EQ(restored.getStep(1).getName(), "generator");
    EXPECT_EQ(restored.getStep(0).getUpdateParameters()[0], ParameterReference(1, "weights"));

    EXPECT_THROW(program.addStep(dStep), std::runtime_error);
}

TEST(TrainingProgramApi, TrainingProgramRejectsEmptyProgramsAtConstructionSerializationAndCompileTime) {
    EXPECT_THROW(TrainingProgram(std::vector<TrainingStep>{}), std::runtime_error);

    TrainingProgram program;
    EXPECT_FALSE(program.isInitialized());
    EXPECT_THROW(program.architectureJson(), std::runtime_error);

    nlohmann::json emptyProgramJson{{"version", "1.0.0"}, {"steps", nlohmann::json::array()}};
    EXPECT_THROW(TrainingProgram::deserialize(emptyProgramJson), std::runtime_error);
}

TEST(TrainingProgramApi, TrainingInputBindingsSerializeAndAttachToStep) {
    TrainingInputBinding zBinding("z", "z_discriminator");
    EXPECT_TRUE(zBinding.isInitialized());
    EXPECT_EQ(zBinding.getNetworkInputName(), "z");
    EXPECT_EQ(zBinding.getBatchInputName(), "z_discriminator");

    nlohmann::json bindingJson = zBinding.architectureJson();
    EXPECT_EQ(bindingJson.at("version").get<std::string>(), "1.0.0");
    EXPECT_EQ(bindingJson.at("network_input_name").get<std::string>(), "z");
    EXPECT_EQ(bindingJson.at("batch_input_name").get<std::string>(), "z_discriminator");
    EXPECT_EQ(TrainingInputBinding::deserialize(bindingJson), zBinding);

    Tensor loss(DataType::FP32, {1});
    TrainingStep step("discriminator",
                      {loss},
                      nullptr,
                      {},
                      1,
                      TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
                      {TrainingInputBinding("real_images", "real_images"), zBinding});

    ASSERT_EQ(step.getInputBindings().size(), 2u);
    nlohmann::json stepJson = step.architectureJson();
    ASSERT_EQ(stepJson.at("input_bindings").size(), 2u);
    EXPECT_EQ(stepJson.at("input_bindings").at(0).at("network_input_name").get<std::string>(), "real_images");
    EXPECT_EQ(stepJson.at("input_bindings").at(1).at("batch_input_name").get<std::string>(), "z_discriminator");

    EXPECT_THROW(TrainingInputBinding("", "batch"), std::runtime_error);
    EXPECT_THROW(TrainingInputBinding("input", ""), std::runtime_error);
    EXPECT_THROW(TrainingStep("bad",
                              {loss},
                              nullptr,
                              {},
                              1,
                              TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
                              {TrainingInputBinding("input", "a"), TrainingInputBinding("input", "b")}),
                 std::runtime_error);
}

TEST(TrainingProgramApi, PlacedNetworkResolvesParameterReferencesAndStepExecutablePlansUpdateSet) {
    constexpr uint32_t batchSize = 2;

    Network network("training_program_step_executable_plan");
    auto sgd = Sgd::Builder().initialLearningRate(0.01f).build();
    NetworkInput input = NetworkInput::Builder().network(network).name("input").dimensions({3}).dataType(DataType::FP32).build();
    FullyConnected fc = FullyConnected::Builder()
                            .network(network)
                            .featureInput(input.getFeatureOutput().value())
                            .numOutputFeatures(2)
                            .hasBias(true)
                            .weightsInitializer(testInitializer())
                            .biasInitializer(testInitializer())
                            .weightsOptimizer(sgd)
                            .biasesOptimizer(sgd)
                            .noActivation()
                            .build();
    NetworkInput labels = NetworkInput::Builder().network(network).name("labels").dimensions({2}).dataType(DataType::FP32).build();
    const Tensor lossRoot = fc.getFeatureOutput().value();
    NetworkOutput::Builder().network(network).name("scores").inputTensor(lossRoot).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(network).name("labels_out").inputTensor(labels.getFeatureOutput().value()).dataType(DataType::FP32).build();

    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    EXPECT_TRUE(placed->hasNetworkInput("input"));
    EXPECT_FALSE(placed->hasNetworkInput("missing_input"));
    std::vector<std::string> networkInputNames = placed->getNetworkInputNames();
    std::set<std::string> inputNames(networkInputNames.begin(), networkInputNames.end());
    EXPECT_EQ(inputNames, (std::set<std::string>{"input", "labels"}));

    ParameterReference weights = fc.getParameterReference("weights");
    BoundParameter boundWeights = placed->resolveParameterReference(weights);
    EXPECT_EQ(boundWeights.getName(), "weights");
    EXPECT_TRUE(boundWeights.isTrainable());
    EXPECT_TRUE(boundWeights.isTrainingEnabled());

    TrainingStep step("generator",
                      {lossRoot},
                      sgd,
                      {weights},
                      3,
                      TrainingStep::GradientClearPolicy::ACCUMULATE,
                      {TrainingInputBinding("input", "z_generator")});
    StepExecutable executable(step, *placed);

    EXPECT_TRUE(executable.isInitialized());
    EXPECT_EQ(executable.getName(), "generator");
    EXPECT_EQ(executable.getRepeatCount(), 3u);
    EXPECT_EQ(executable.getGradientClearPolicy(), TrainingStep::GradientClearPolicy::ACCUMULATE);
    ASSERT_EQ(executable.getLossRoots().size(), 1u);
    ASSERT_EQ(executable.getResolvedLossRoots().size(), 1u);
    EXPECT_EQ(executable.getResolvedLossRoots()[0].getOriginalId(), lossRoot.getOriginalId());
    ASSERT_EQ(executable.getUpdateParameterReferences().size(), 1u);
    ASSERT_EQ(executable.getResolvedUpdateParameters().size(), 1u);
    EXPECT_EQ(executable.getResolvedUpdateParameters()[0].getName(), "weights");
    ASSERT_EQ(executable.getInputBindings().size(), 1u);
    EXPECT_EQ(executable.getInputBindings()[0].getBatchInputName(), "z_generator");
    ASSERT_EQ(executable.getResolvedInputBindings().size(), 2u);
    std::map<std::string, std::string> resolvedInputBindings;
    for (const TrainingInputBinding& binding : executable.getResolvedInputBindings()) {
        resolvedInputBindings[binding.getNetworkInputName()] = binding.getBatchInputName();
    }
    EXPECT_EQ(resolvedInputBindings["input"], "z_generator");
    EXPECT_EQ(resolvedInputBindings["labels"], "labels");
    EXPECT_EQ(executable.getRequiredBatchInputNames(), (std::vector<std::string>{"labels", "z_generator"}));

    nlohmann::json executableJson = executable.architectureJson();
    EXPECT_TRUE(executableJson.at("planned").get<bool>());
    EXPECT_EQ(executableJson.at("resolved_loss_root_count").get<uint64_t>(), 1u);
    EXPECT_EQ(executableJson.at("resolved_update_parameter_count").get<uint64_t>(), 1u);
    EXPECT_EQ(executableJson.at("input_bindings").at(0).at("network_input_name").get<std::string>(), "input");
    ASSERT_EQ(executableJson.at("resolved_input_bindings").size(), 2u);
    EXPECT_EQ(executableJson.at("required_batch_input_names"), (nlohmann::json::array({"labels", "z_generator"})));

    TrainingProgram program({step});
    std::vector<StepExecutable> executables = program.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getName(), "generator");

    EXPECT_THROW(placed->resolveParameterReference(ParameterReference(fc.getId(), "missing")), std::runtime_error);
    EXPECT_THROW(StepExecutable(TrainingStep("bad", {lossRoot}, sgd, {ParameterReference(999999, "weights")}), *placed), std::runtime_error);
    EXPECT_THROW(StepExecutable(TrainingStep("bad_loss", {Tensor(DataType::FP32, {1})}, sgd, {weights}), *placed), std::runtime_error);
    EXPECT_THROW(StepExecutable(TrainingStep("bad_input",
                                             {lossRoot},
                                             sgd,
                                             {weights},
                                             1,
                                             TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
                                             {TrainingInputBinding("missing_input", "batch")}),
                                *placed),
                 std::runtime_error);

    boundWeights.setTrainingEnabled(false);
    EXPECT_THROW(StepExecutable(step, *placed), std::runtime_error);
}
