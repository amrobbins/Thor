#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Parameter/ParameterReference.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Training/ExecutableTrainingPlan.h"
#include "DeepLearning/Api/Training/PhaseGraphConnector.h"
#include "DeepLearning/Api/Training/StepExecutable.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"
#include "DeepLearning/Api/Training/TrainingPhase.h"
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


struct PhaseNetworkFixture {
    std::shared_ptr<Network> network;
    Tensor lossRoot;
    Tensor outputTensor;
};

PhaseNetworkFixture buildPhaseNetwork(const std::string& networkName,
                                      const std::string& inputName,
                                      const std::string& outputName,
                                      bool inputExternal = true,
                                      bool outputExternal = true,
                                      bool withLoss = true) {
    auto network = std::make_shared<Network>(networkName);
    NetworkInput input = NetworkInput::Builder()
                             .network(*network)
                             .name(inputName)
                             .dimensions({1})
                             .dataType(DataType::FP32)
                             .external(inputExternal)
                             .build();
    Tensor outputTensor = input.getFeatureOutput().value();
    Tensor lossRoot;
    if (withLoss) {
        NetworkInput labels =
            NetworkInput::Builder().network(*network).name("labels").dimensions({1}).dataType(DataType::FP32).build();
        MSE loss = MSE::Builder().network(*network).predictions(outputTensor).labels(labels.getFeatureOutput().value()).build();
        lossRoot = loss.getLoss();
        NetworkOutput::Builder().network(*network).name(outputName + "_loss").inputTensor(lossRoot).dataType(DataType::FP32).build();

        std::vector<Tensor> derivedObjectiveRoots = network->getLossRootTensors();
        if (derivedObjectiveRoots.size() != 1) {
            throw std::runtime_error("test phase fixture expected exactly one derived phase graph loss");
        }
        lossRoot = derivedObjectiveRoots[0];
    }
    NetworkOutput::Builder()
        .network(*network)
        .name(outputName)
        .inputTensor(outputTensor)
        .dataType(DataType::FP32)
        .external(outputExternal)
        .build();
    return PhaseNetworkFixture{network, lossRoot, outputTensor};
}

std::shared_ptr<TrainingPhase> makePhase(const std::string& phaseName,
                                         const std::string& inputName = "examples",
                                         const std::string& outputName = "",
                                         bool inputExternal = true,
                                         bool outputExternal = true,
                                         bool withLoss = true,
                                         bool enabled = true,
                                         PhaseNetworkFixture* fixtureOut = nullptr) {
    const std::string resolvedOutputName = outputName.empty() ? phaseName : outputName;
    PhaseNetworkFixture fixture = buildPhaseNetwork(
        phaseName + "_network", inputName, resolvedOutputName, inputExternal, outputExternal, withLoss);
    if (fixtureOut != nullptr) {
        *fixtureOut = fixture;
    }
    return std::make_shared<TrainingPhase>(phaseName, fixture.network, enabled);
}

std::shared_ptr<Network> nonOwningNetworkPtr(Network& network) {
    return std::shared_ptr<Network>(&network, [](Network*) {});
}

std::shared_ptr<TrainingPhase> makeNonOwningPhase(const std::string& phaseName, Network& network, bool enabled = true) {
    return std::make_shared<TrainingPhase>(phaseName, nonOwningNetworkPtr(network), enabled);
}

ComposedPhaseGraph composeActivePhases(const TrainingStep& step) {
    PhaseGraphComposeOptions options;
    options.networkName = "training_program_test_composed_phases";
    options.exposePhaseOutputsAsNetworkOutputs = true;
    return buildComposedPhaseGraphByName(step.getActivePhaseNetworkSpecs(), options);
}

template <typename Fn>
void expectRuntimeErrorContains(Fn&& fn, const std::string& expectedMessageFragment) {
    try {
        fn();
        FAIL() << "Expected std::runtime_error containing: " << expectedMessageFragment;
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(expectedMessageFragment), std::string::npos) << "Actual error: " << error.what();
    } catch (...) {
        FAIL() << "Expected std::runtime_error containing: " << expectedMessageFragment;
    }
}

}  // namespace

TEST(TrainingPhaseApi, ConstructsWithNetworkAndDerivesOutputs) {
    PhaseNetworkFixture fixture = buildPhaseNetwork("daily_phase_network", "examples", "forecast");

    TrainingPhase phase("daily_prediction", fixture.network, true);

    EXPECT_TRUE(phase.isInitialized());
    EXPECT_TRUE(phase.isEnabled());
    EXPECT_NE(phase.getNetwork(), nullptr);
    EXPECT_EQ(phase.getName(), "daily_prediction");
    EXPECT_EQ(phase.getNetwork(), fixture.network);
    ASSERT_EQ(phase.getNetwork()->getLossRootTensors().size(), 1u);
    EXPECT_EQ(phase.getNetwork()->getLossRootTensors()[0], fixture.lossRoot);
    ASSERT_EQ(phase.getOutputs().size(), 2u);
    EXPECT_EQ(phase.getOutputs().at("forecast"), fixture.outputTensor);
    EXPECT_TRUE(phase.getOutputs().count("forecast_loss") != 0);
}

TEST(TrainingPhaseApi, EnableDisableMutatesPhaseState) {
    PhaseNetworkFixture fixture = buildPhaseNetwork("daily_phase_network", "examples", "forecast");
    TrainingPhase phase("daily_prediction", fixture.network);

    EXPECT_TRUE(phase.isEnabled());
    phase.disable();
    EXPECT_FALSE(phase.isEnabled());
    phase.enable();
    EXPECT_TRUE(phase.isEnabled());
    phase.setEnabled(false);
    EXPECT_FALSE(phase.isEnabled());
    phase.setEnabled(true);
    EXPECT_TRUE(phase.isEnabled());
}

TEST(TrainingPhaseApi, AllowsForwardOnlyPhaseWithoutGraphLosses) {
    PhaseNetworkFixture fixture = buildPhaseNetwork("daily_forward_network", "examples", "forecast", true, true, false);
    TrainingPhase phase("daily_forward", fixture.network, true);

    EXPECT_TRUE(phase.isInitialized());
    EXPECT_NE(phase.getNetwork(), nullptr);
    EXPECT_TRUE(phase.getNetwork()->getLossRootTensors().empty());
    ASSERT_EQ(phase.getOutputs().size(), 1u);
    EXPECT_EQ(phase.getOutputs().at("forecast"), fixture.outputTensor);
}

TEST(TrainingPhaseApi, PhasePreservesExternalFalseOutputsForLocalWiring) {
    PhaseNetworkFixture fixture = buildPhaseNetwork("daily_phase_network", "features", "hidden", true, false, false);

    TrainingPhase phase("daily_prediction", fixture.network, false);

    EXPECT_TRUE(phase.isInitialized());
    EXPECT_FALSE(phase.isEnabled());
    EXPECT_NE(phase.getNetwork(), nullptr);
    EXPECT_EQ(phase.getNetwork(), fixture.network);
    EXPECT_TRUE(phase.getNetwork()->getLossRootTensors().empty());
    ASSERT_EQ(phase.getOutputs().size(), 1u);
    EXPECT_EQ(phase.getOutputs().at("hidden"), fixture.outputTensor);

    nlohmann::json phaseJson = phase.architectureJson();
    EXPECT_EQ(phaseJson.at("version").get<std::string>(), "1.1.0");
    EXPECT_EQ(phaseJson.at("network").at("name").get<std::string>(), "daily_phase_network");

    TrainingPhase restored = TrainingPhase::deserialize(phaseJson);
    EXPECT_NE(restored.getNetwork(), nullptr);
    EXPECT_EQ(restored.getNetwork()->getNetworkName(), "daily_phase_network");
    ASSERT_EQ(restored.getOutputs().size(), 1u);
    EXPECT_TRUE(restored.getOutputs().count("hidden") != 0);
    EXPECT_FALSE(restored.isEnabled());
}

TEST(TrainingStepApi, ActivePhaseNetworkSpecsExposeEnabledPhases) {
    auto encoderNetwork = std::make_shared<Network>("encoder_phase_network");
    NetworkInput input = NetworkInput::Builder().network(*encoderNetwork).name("features").dimensions({3}).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*encoderNetwork).name("hidden").inputTensor(input.getFeatureOutput().value()).external(false).build();

    auto headNetwork = std::make_shared<Network>("head_phase_network");
    NetworkInput hidden = NetworkInput::Builder().network(*headNetwork).name("hidden").dimensions({3}).dataType(DataType::FP32).external(false).build();
    NetworkInput labels = NetworkInput::Builder().network(*headNetwork).name("labels").dimensions({3}).dataType(DataType::FP32).build();
    MSE mse = MSE::Builder().network(*headNetwork).predictions(hidden.getFeatureOutput().value()).labels(labels.getFeatureOutput().value()).build();
    NetworkOutput::Builder().network(*headNetwork).name("prediction").inputTensor(hidden.getFeatureOutput().value()).build();
    NetworkOutput::Builder().network(*headNetwork).name("mse_loss").inputTensor(mse.getLoss()).build();

    auto encoderPhase = std::make_shared<TrainingPhase>("encoder", encoderNetwork);
    auto headPhase = std::make_shared<TrainingPhase>("head", headNetwork, false);
    TrainingStep step("two_phase", std::vector<std::shared_ptr<TrainingPhase>>{encoderPhase, headPhase}, nullptr, {});

    std::vector<PhaseGraphNetworkSpec> specs = step.getActivePhaseNetworkSpecs();
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].phaseName, "encoder");
    EXPECT_EQ(specs[0].network, encoderNetwork);
    EXPECT_TRUE(specs[0].active);

    headPhase->enable();
    specs = step.getActivePhaseNetworkSpecs();
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].phaseName, "encoder");
    EXPECT_EQ(specs[1].phaseName, "head");
}

TEST(TrainingPhaseApi, RejectsInvalidConstruction) {
    PhaseNetworkFixture fixture = buildPhaseNetwork("valid_phase_network", "examples", "prediction");

    EXPECT_THROW(TrainingPhase("", fixture.network), std::runtime_error);

    auto unnamedNetwork = std::make_shared<Network>("");
    EXPECT_THROW(TrainingPhase("bad", unnamedNetwork), std::runtime_error);
}

TEST(TrainingPhaseApi, SerializesAndDeserializes) {
    PhaseNetworkFixture fixture = buildPhaseNetwork("daily_phase_network", "examples", "forecast");

    TrainingPhase phase("daily_prediction", fixture.network, false);

    nlohmann::json j = phase.architectureJson();
    EXPECT_EQ(j.at("version").get<std::string>(), "1.1.0");
    EXPECT_EQ(j.at("name").get<std::string>(), "daily_prediction");
    EXPECT_FALSE(j.at("enabled").get<bool>());
    ASSERT_TRUE(j.contains("network"));

    TrainingPhase restored = TrainingPhase::deserialize(j);
    EXPECT_TRUE(restored.isInitialized());
    EXPECT_FALSE(restored.isEnabled());
    EXPECT_NE(restored.getNetwork(), nullptr);
    EXPECT_EQ(restored.getName(), phase.getName());
    EXPECT_EQ(restored.getNetwork()->getNetworkName(), "daily_phase_network");
    ASSERT_EQ(restored.getNetwork()->getLossRootTensors().size(), 1u);
    ASSERT_EQ(restored.getOutputs().size(), 2u);
    EXPECT_TRUE(restored.getOutputs().count("forecast") != 0);
    EXPECT_TRUE(restored.getOutputs().count("forecast_loss") != 0);

    nlohmann::json badVersion = j;
    badVersion["version"] = "0.0.0";
    EXPECT_THROW(TrainingPhase::deserialize(badVersion), std::runtime_error);
}

TEST(TrainingPhaseApi, SerializesAndDeserializesEnabledAndDisabledStateAndRejectsUnsupportedVersion) {
    PhaseNetworkFixture enabledFixture = buildPhaseNetwork("enabled_phase_network", "examples", "prediction");
    PhaseNetworkFixture disabledFixture = buildPhaseNetwork("disabled_phase_network", "examples", "prediction");

    TrainingPhase enabledPhase("enabled_phase", enabledFixture.network, true);
    TrainingPhase disabledPhase("disabled_phase", disabledFixture.network, false);

    nlohmann::json enabledJson = enabledPhase.architectureJson();
    nlohmann::json disabledJson = disabledPhase.architectureJson();
    EXPECT_TRUE(enabledJson.at("enabled").get<bool>());
    EXPECT_FALSE(disabledJson.at("enabled").get<bool>());

    EXPECT_TRUE(TrainingPhase::deserialize(enabledJson).isEnabled());
    EXPECT_FALSE(TrainingPhase::deserialize(disabledJson).isEnabled());

    nlohmann::json missingEnabledJson = enabledJson;
    missingEnabledJson.erase("enabled");
    EXPECT_TRUE(TrainingPhase::deserialize(missingEnabledJson).isEnabled());

    nlohmann::json badVersion = enabledJson;
    badVersion["version"] = "2.0.0";
    expectRuntimeErrorContains([&]() { (void)TrainingPhase::deserialize(badVersion); }, "Unsupported TrainingPhase version: 2.0.0");
}

TEST(TrainingPhaseApi, RejectsInvalidConstructionWithSpecificErrors) {
    PhaseNetworkFixture fixture = buildPhaseNetwork("valid_phase_network", "examples", "prediction");

    expectRuntimeErrorContains([&]() { TrainingPhase("", fixture.network); }, "requires a non-empty name");

    auto unnamedNetwork = std::make_shared<Network>("");
    expectRuntimeErrorContains([&]() { TrainingPhase("bad", unnamedNetwork); }, "network must have a non-empty Network name");
}

TEST(TrainingProgramApi, TrainingStepDerivesObjectiveRootsFromNetworkBackedPhases) {
    PhaseNetworkFixture fixture;
    auto phase = makePhase("daily", "examples", "daily", true, true, true, true, &fixture);
    auto sgd = Sgd::Builder().initialLearningRate(0.01f).build();

    TrainingStep step("daily", std::vector<std::shared_ptr<TrainingPhase>>{phase}, sgd, {});

    EXPECT_TRUE(step.isInitialized());
    EXPECT_TRUE(step.isEnabled());
    ASSERT_EQ(step.getObjectiveRoots().size(), 1u);
    EXPECT_EQ(step.getObjectiveRoots()[0], fixture.lossRoot);
    ASSERT_EQ(step.getActiveObjectiveRoots().size(), 1u);
    EXPECT_EQ(step.getActiveObjectiveRoots()[0], fixture.lossRoot);
    ASSERT_EQ(step.getPhases().size(), 1u);
    EXPECT_EQ(step.getActivePhaseNames(), (std::vector<std::string>{"daily"}));
    ASSERT_EQ(step.getActivePhaseNetworkSpecs().size(), 1u);
}

TEST(TrainingProgramApi, TrainingStepEnableDisableMutatesStepStateAndActiveObjectiveRoots) {
    PhaseNetworkFixture fixture;
    auto phase = makePhase("daily", "examples", "daily", true, true, true, true, &fixture);
    TrainingStep step("daily", std::vector<std::shared_ptr<TrainingPhase>>{phase}, nullptr, {});

    EXPECT_TRUE(step.isEnabled());
    ASSERT_EQ(step.getActiveObjectiveRoots().size(), 1u);
    step.disable();
    EXPECT_FALSE(step.isEnabled());
    EXPECT_TRUE(step.getActiveObjectiveRoots().empty());
    step.enable();
    EXPECT_TRUE(step.isEnabled());
    ASSERT_EQ(step.getActiveObjectiveRoots().size(), 1u);
    step.setEnabled(false);
    EXPECT_FALSE(step.isEnabled());
    step.setEnabled(true);
    EXPECT_TRUE(step.isEnabled());
}

TEST(TrainingProgramApi, TrainingStepActiveObjectiveRootsComeOnlyFromEnabledPhases) {
    PhaseNetworkFixture dailyFixture;
    PhaseNetworkFixture aggregateFixture;
    auto dailyPhase = makePhase("daily_prediction", "examples", "", true, true, true, true, &dailyFixture);
    auto aggregatePhase = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, false, &aggregateFixture);

    TrainingStep step("demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, aggregatePhase}, nullptr, {});

    ASSERT_EQ(step.getObjectiveRoots().size(), 2u);
    std::vector<Tensor> activeRoots = step.getActiveObjectiveRoots();
    ASSERT_EQ(activeRoots.size(), 1u);
    EXPECT_EQ(activeRoots[0], dailyFixture.lossRoot);

    aggregatePhase->enable();
    activeRoots = step.getActiveObjectiveRoots();
    ASSERT_EQ(activeRoots.size(), 2u);
    EXPECT_EQ(activeRoots[0], dailyFixture.lossRoot);
    EXPECT_EQ(activeRoots[1], aggregateFixture.lossRoot);

    dailyPhase->disable();
    activeRoots = step.getActiveObjectiveRoots();
    ASSERT_EQ(activeRoots.size(), 1u);
    EXPECT_EQ(activeRoots[0], aggregateFixture.lossRoot);
}

TEST(TrainingProgramApi, TrainingStepActivePhaseNamesComeOnlyFromEnabledPhasesAndMayIncludeForwardOnlyPhases) {
    PhaseNetworkFixture dailyFixture;
    PhaseNetworkFixture aggregateFixture;
    auto preprocessingPhase = makePhase("feature_preprocessing", "examples", "features", true, false, false);
    auto dailyPhase = makePhase("daily_prediction", "features", "", false, true, true, true, &dailyFixture);
    auto aggregatePhase = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, false, &aggregateFixture);

    TrainingStep step(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{preprocessingPhase, dailyPhase, aggregatePhase}, nullptr, {});

    EXPECT_EQ(step.getActivePhaseNames(), (std::vector<std::string>{"feature_preprocessing", "daily_prediction"}));
    std::vector<Tensor> activeRoots = step.getActiveObjectiveRoots();
    ASSERT_EQ(activeRoots.size(), 1u);
    EXPECT_EQ(activeRoots[0], dailyFixture.lossRoot);

    aggregatePhase->enable();
    EXPECT_EQ(step.getActivePhaseNames(), (std::vector<std::string>{"feature_preprocessing", "daily_prediction", "aggregate_prediction"}));
    activeRoots = step.getActiveObjectiveRoots();
    ASSERT_EQ(activeRoots.size(), 2u);
    EXPECT_EQ(activeRoots[0], dailyFixture.lossRoot);
    EXPECT_EQ(activeRoots[1], aggregateFixture.lossRoot);
}

TEST(TrainingProgramApi, TrainingStepPhaseGraphValidationSkipsDisabledPhasesAndDisabledSteps) {
    PhaseNetworkFixture dailyFixture;
    auto dailyPhase = makePhase("daily_prediction", "examples", "", true, true, true, true, &dailyFixture);
    auto disabledAggregateWithMissingProducer = makePhase("aggregate_prediction", "missing_phase", "", false, true, true, false);
    TrainingStep disabledPhaseStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, disabledAggregateWithMissingProducer}, nullptr, {});
    EXPECT_EQ(disabledPhaseStep.getActivePhaseNames(), (std::vector<std::string>{"daily_prediction"}));
    EXPECT_NO_THROW((void)composeActivePhases(disabledPhaseStep));

    disabledAggregateWithMissingProducer->enable();
    expectRuntimeErrorContains([&]() { (void)composeActivePhases(disabledPhaseStep); },
                               "non-external input 'missing_phase' in phase 'aggregate_prediction' is not satisfied");

    auto disabledDaily = makePhase("daily_prediction", "examples", "", true, true, true, false);
    auto aggregateDependsOnDisabledDaily = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, true);
    TrainingStep disabledStep("demand_forecast",
                              std::vector<std::shared_ptr<TrainingPhase>>{disabledDaily, aggregateDependsOnDisabledDaily},
                              nullptr,
                              {},
                              1,
                              TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
                              {},
                              false);
    EXPECT_TRUE(disabledStep.getActivePhaseNames().empty());
    EXPECT_TRUE(disabledStep.getActiveObjectiveRoots().empty());
}

TEST(TrainingProgramApi, PhaseGraphDependencyErrorsAreSpecificAndSearchable) {
    auto disabledDaily = makePhase("daily_prediction", "examples", "", true, true, true, false);
    auto aggregate = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, true);
    TrainingStep disabledProducerStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{disabledDaily, aggregate}, nullptr, {});
    expectRuntimeErrorContains([&]() { (void)composeActivePhases(disabledProducerStep); },
                               "non-external input 'daily_prediction' in phase 'aggregate_prediction' is not satisfied");

    auto missingProducerPhase = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, true);
    TrainingStep missingProducerStep("demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{missingProducerPhase}, nullptr, {});
    expectRuntimeErrorContains([&]() { (void)composeActivePhases(missingProducerStep); },
                               "non-external input 'daily_prediction' in phase 'aggregate_prediction' is not satisfied");

    auto aggregateFirst = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, true);
    auto dailySecond = makePhase("daily_prediction");
    TrainingStep reorderableStep("demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{aggregateFirst, dailySecond}, nullptr, {});
    ComposedPhaseGraph graph = composeActivePhases(reorderableStep);
    EXPECT_EQ(graph.activePhaseNames, (std::vector<std::string>{"daily_prediction", "aggregate_prediction"}));
}

TEST(TrainingProgramApi, PhaseGraphRejectsInvalidProducerSets) {
    auto disabledDaily = makePhase("daily_prediction", "examples", "", true, true, true, false);
    auto aggregateDependsOnDisabled = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, true);
    TrainingStep disabledProducerStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{disabledDaily, aggregateDependsOnDisabled}, nullptr, {});
    EXPECT_THROW((void)composeActivePhases(disabledProducerStep), std::runtime_error);

    auto aggregateMissingDependency = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, true);
    TrainingStep missingProducerStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{aggregateMissingDependency}, nullptr, {});
    EXPECT_THROW((void)composeActivePhases(missingProducerStep), std::runtime_error);

    auto aggregateForwardDependency = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, true);
    auto dailyLater = makePhase("daily_prediction");
    TrainingStep forwardReferenceStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{aggregateForwardDependency, dailyLater}, nullptr, {});
    EXPECT_NO_THROW((void)composeActivePhases(forwardReferenceStep));
}

TEST(TrainingProgramApi, TrainingStepRejectsInvalidPhaseLists) {
    auto dailyA = makePhase("daily_prediction");
    auto dailyB = makePhase("daily_prediction");

    EXPECT_THROW(TrainingStep("bad", std::vector<std::shared_ptr<TrainingPhase>>{}, nullptr, {}), std::runtime_error);
    EXPECT_THROW(TrainingStep("bad", std::vector<std::shared_ptr<TrainingPhase>>{nullptr}, nullptr, {}), std::runtime_error);
    EXPECT_THROW(TrainingStep("bad", std::vector<std::shared_ptr<TrainingPhase>>{dailyA, dailyB}, nullptr, {}), std::runtime_error);
}

TEST(TrainingProgramApi, TrainingStepSerializesPhaseAwareExecutionView) {
    auto dailyPhase = makePhase("daily_prediction", "input");
    auto aggregatePhase = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, false);
    auto sgd = Sgd::Builder().initialLearningRate(0.01f).build();

    TrainingStep step("demand_forecast",
                      std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, aggregatePhase},
                      sgd,
                      {ParameterReference(123, "weights")},
                      2,
                      TrainingStep::GradientClearPolicy::ACCUMULATE,
                      {TrainingInputBinding("input", "input")},
                      false);

    nlohmann::json j = step.architectureJson();
    EXPECT_EQ(j.at("version").get<std::string>(), "1.2.0");
    EXPECT_EQ(j.at("name").get<std::string>(), "demand_forecast");
    EXPECT_FALSE(j.at("enabled").get<bool>());
    ASSERT_EQ(j.at("phases").size(), 2u);
    EXPECT_EQ(j.at("phases").at(0).at("name").get<std::string>(), "daily_prediction");
    EXPECT_EQ(j.at("phases").at(1).at("name").get<std::string>(), "aggregate_prediction");
    EXPECT_TRUE(j.at("phases").at(0).contains("network"));
    EXPECT_TRUE(j.at("phases").at(1).contains("network"));
    EXPECT_FALSE(j.at("phases").at(1).at("enabled").get<bool>());

    TrainingStep restored = TrainingStep::deserialize(j);
    EXPECT_TRUE(restored.isInitialized());
    EXPECT_FALSE(restored.isEnabled());
    EXPECT_EQ(restored.getName(), step.getName());
    EXPECT_EQ(restored.getRepeatCount(), step.getRepeatCount());
    EXPECT_EQ(restored.getGradientClearPolicy(), step.getGradientClearPolicy());
    ASSERT_EQ(restored.getObjectiveRoots().size(), 2u);
    ASSERT_EQ(restored.getPhases().size(), 2u);
    EXPECT_EQ(restored.getPhases()[0]->getName(), "daily_prediction");
    EXPECT_EQ(restored.getPhases()[1]->getName(), "aggregate_prediction");
    EXPECT_NE(restored.getPhases()[0]->getNetwork(), nullptr);
    EXPECT_NE(restored.getPhases()[1]->getNetwork(), nullptr);
    EXPECT_FALSE(restored.getPhases()[1]->isEnabled());
    EXPECT_TRUE(restored.getActiveObjectiveRoots().empty());
    EXPECT_EQ(restored.getUpdateParameters(), step.getUpdateParameters());
    ASSERT_EQ(restored.getInputBindings().size(), 1u);
    EXPECT_EQ(restored.getInputBindings()[0].getBatchInputName(), "input");
    EXPECT_NE(restored.getOptimizer(), nullptr);
}

TEST(TrainingProgramApi, TrainingStepSerializationPreservesPhaseEnablement) {
    auto dailyPhase = makePhase("daily_prediction");
    auto aggregatePhase = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, false);

    TrainingStep phasedStep("demand_forecast",
                            std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, aggregatePhase},
                            nullptr,
                            {},
                            3,
                            TrainingStep::GradientClearPolicy::ACCUMULATE);
    nlohmann::json phasedJson = phasedStep.architectureJson();
    TrainingStep restoredPhasedStep = TrainingStep::deserialize(phasedJson);
    ASSERT_EQ(restoredPhasedStep.getPhases().size(), 2u);
    EXPECT_TRUE(restoredPhasedStep.getPhases()[0]->isEnabled());
    EXPECT_FALSE(restoredPhasedStep.getPhases()[1]->isEnabled());
    EXPECT_NE(restoredPhasedStep.getPhases()[0]->getNetwork(), nullptr);
    EXPECT_NE(restoredPhasedStep.getPhases()[1]->getNetwork(), nullptr);
    EXPECT_EQ(restoredPhasedStep.getActivePhaseNames(), (std::vector<std::string>{"daily_prediction"}));

    nlohmann::json badVersion = phasedJson;
    badVersion["version"] = "9.9.9";
    expectRuntimeErrorContains([&]() { (void)TrainingStep::deserialize(badVersion); }, "Unsupported TrainingStep version: 9.9.9");
}

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
    auto sgd = Sgd::Builder().initialLearningRate(0.01f).build();
    std::vector<ParameterReference> params{ParameterReference(123, "weights"), ParameterReference(123, "biases")};
    auto phase = makePhase("discriminator");

    TrainingStep step("discriminator",
                      std::vector<std::shared_ptr<TrainingPhase>>{phase},
                      sgd,
                      params,
                      2,
                      TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP);

    EXPECT_TRUE(step.isInitialized());
    EXPECT_EQ(step.getName(), "discriminator");
    EXPECT_EQ(step.getRepeatCount(), 2u);
    EXPECT_TRUE(step.updatesParameter(params[0]));
    EXPECT_FALSE(step.updatesParameter(ParameterReference(999, "weights")));

    nlohmann::json j = step.architectureJson();
    EXPECT_EQ(j.at("version").get<std::string>(), "1.2.0");
    EXPECT_EQ(j.at("name").get<std::string>(), "discriminator");
    EXPECT_EQ(j.at("repeat_count").get<uint32_t>(), 2u);
    ASSERT_EQ(j.at("phases").size(), 1u);
    EXPECT_EQ(j.at("update_parameters").size(), 2u);
    EXPECT_TRUE(j.contains("optimizer"));

    TrainingStep restored = TrainingStep::deserialize(j);
    EXPECT_TRUE(restored.isInitialized());
    EXPECT_EQ(restored.getName(), step.getName());
    EXPECT_EQ(restored.getRepeatCount(), step.getRepeatCount());
    EXPECT_EQ(restored.getGradientClearPolicy(), step.getGradientClearPolicy());
    EXPECT_EQ(restored.getObjectiveRoots().size(), step.getObjectiveRoots().size());
    EXPECT_EQ(restored.getUpdateParameters(), step.getUpdateParameters());
    EXPECT_NE(restored.getOptimizer(), nullptr);
}

TEST(TrainingProgramApi, TrainingStepRejectsDuplicateUpdateParameters) {
    auto sgd = Sgd::Builder().initialLearningRate(0.01f).build();
    ParameterReference weights(123, "weights");
    auto phase = makePhase("bad_phase");

    EXPECT_THROW(TrainingStep("bad", std::vector<std::shared_ptr<TrainingPhase>>{phase}, sgd, {weights, weights}), std::runtime_error);
}

TEST(TrainingProgramApi, TrainingStepAllowsUpdatesWithoutStepOptimizerForPerParameterOptimizers) {
    ParameterReference weights(123, "weights");
    auto phase = makePhase("per_parameter_phase");

    TrainingStep step("per_parameter", std::vector<std::shared_ptr<TrainingPhase>>{phase}, nullptr, {weights});
    EXPECT_TRUE(step.isInitialized());
    EXPECT_EQ(step.getOptimizer(), nullptr);

    nlohmann::json j = step.architectureJson();
    EXPECT_FALSE(j.contains("optimizer"));

    TrainingStep restored = TrainingStep::deserialize(j);
    EXPECT_TRUE(restored.isInitialized());
    EXPECT_EQ(restored.getOptimizer(), nullptr);
    EXPECT_EQ(restored.getUpdateParameters(), step.getUpdateParameters());
}

TEST(TrainingProgramApi, TrainingProgramKeepsOrderedUniqueSteps) {
    auto dSgd = Sgd::Builder().initialLearningRate(0.01f).build();
    auto gSgd = Sgd::Builder().initialLearningRate(0.02f).build();

    auto dStep = std::make_shared<TrainingStep>(
        "discriminator", std::vector<std::shared_ptr<TrainingPhase>>{makePhase("discriminator_phase")}, dSgd, std::vector<ParameterReference>{ParameterReference(1, "weights")});
    auto gStep = std::make_shared<TrainingStep>(
        "generator", std::vector<std::shared_ptr<TrainingPhase>>{makePhase("generator_phase")}, gSgd, std::vector<ParameterReference>{ParameterReference(2, "weights")});

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

TEST(TrainingProgramApi, HoldsTrainingStepsByReference) {
    auto step = std::make_shared<TrainingStep>("daily", std::vector<std::shared_ptr<TrainingPhase>>{makePhase("daily_phase")}, nullptr, std::vector<ParameterReference>{});

    TrainingProgram program(std::vector<std::shared_ptr<TrainingStep>>{step});
    EXPECT_TRUE(program.isInitialized());
    ASSERT_EQ(program.getNumSteps(), 1u);
    EXPECT_TRUE(program.getStep(0).isEnabled());
    EXPECT_EQ(program.getStepReference(0), step);

    step->disable();
    EXPECT_FALSE(program.getStep(0).isEnabled());

    step->enable();
    EXPECT_TRUE(program.getStep(0).isEnabled());

    program.getStep(0).disable();
    EXPECT_FALSE(step->isEnabled());

    program.getStep(0).enable();
    EXPECT_TRUE(step->isEnabled());
}

TEST(TrainingProgramApi, AddStepStoresSharedReferenceAndRejectsNullReference) {
    auto step = std::make_shared<TrainingStep>("daily", std::vector<std::shared_ptr<TrainingPhase>>{makePhase("daily_phase")}, nullptr, std::vector<ParameterReference>{});

    TrainingProgram program;
    EXPECT_THROW(program.addStep(nullptr), std::runtime_error);

    program.addStep(step);
    ASSERT_EQ(program.getNumSteps(), 1u);
    EXPECT_EQ(program.getStepReference(0), step);

    step->disable();
    EXPECT_FALSE(program.getStep(0).isEnabled());

    program.getStepReference(0)->enable();
    EXPECT_TRUE(step->isEnabled());
}

TEST(TrainingProgramApi, DisabledPhasesDoNotValidateInactiveInternalInputsUntilEnabled) {
    PhaseNetworkFixture dailyFixture;
    auto dailyPhase = makePhase("daily_prediction", "examples", "", true, true, true, true, &dailyFixture);
    auto disabledAggregateWithMissingProducer = makePhase("aggregate_prediction", "not_declared_yet", "", false, true, true, false);

    TrainingStep step("demand_forecast",
                      std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, disabledAggregateWithMissingProducer},
                      nullptr,
                      std::vector<ParameterReference>{});

    EXPECT_EQ(step.getActivePhaseNames(), (std::vector<std::string>{"daily_prediction"}));
    std::vector<Tensor> activeRoots = step.getActiveObjectiveRoots();
    ASSERT_EQ(activeRoots.size(), 1u);
    EXPECT_EQ(activeRoots[0], dailyFixture.lossRoot);
    EXPECT_NO_THROW((void)composeActivePhases(step));

    disabledAggregateWithMissingProducer->enable();
    EXPECT_THROW((void)composeActivePhases(step), std::runtime_error);
}

TEST(TrainingProgramApi, PhaseMutationAfterProgramConstructionIsVisibleThroughStep) {
    PhaseNetworkFixture dailyFixture;
    PhaseNetworkFixture aggregateFixture;
    auto dailyPhase = makePhase("daily_prediction", "examples", "", true, true, true, true, &dailyFixture);
    auto aggregatePhase = makePhase("aggregate_prediction", "daily_prediction", "", false, true, true, false, &aggregateFixture);
    auto step = std::make_shared<TrainingStep>("demand_forecast",
                                               std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, aggregatePhase},
                                               nullptr,
                                               std::vector<ParameterReference>{});

    TrainingProgram program(std::vector<std::shared_ptr<TrainingStep>>{step});
    std::vector<Tensor> activeRoots = program.getStep(0).getActiveObjectiveRoots();
    ASSERT_EQ(activeRoots.size(), 1u);
    EXPECT_EQ(activeRoots[0], dailyFixture.lossRoot);

    aggregatePhase->enable();
    activeRoots = program.getStep(0).getActiveObjectiveRoots();
    ASSERT_EQ(activeRoots.size(), 2u);
    EXPECT_EQ(activeRoots[0], dailyFixture.lossRoot);
    EXPECT_EQ(activeRoots[1], aggregateFixture.lossRoot);

    dailyPhase->disable();
    activeRoots = program.getStep(0).getActiveObjectiveRoots();
    ASSERT_EQ(activeRoots.size(), 1u);
    EXPECT_EQ(activeRoots[0], aggregateFixture.lossRoot);
}

TEST(TrainingProgramApi, TrainingProgramRejectsEmptyProgramsAtConstructionSerializationAndCompileTime) {
    EXPECT_THROW(TrainingProgram(std::vector<std::shared_ptr<TrainingStep>>{}), std::runtime_error);

    TrainingProgram program;
    EXPECT_FALSE(program.isInitialized());
    EXPECT_THROW(static_cast<void>(program.architectureJson()), std::runtime_error);

    nlohmann::json emptyProgramJson{{"version", "1.0.0"}, {"steps", nlohmann::json::array()}};
    EXPECT_THROW(TrainingProgram::deserialize(emptyProgramJson), std::runtime_error);
}

TEST(TrainingProgramApi, NetworkResolvesApiSideShapedObjectiveRootsToCanonicalRawLossLayers) {
    Network network("training_program_raw_loss_root_resolution");
    auto sgd = Sgd::Builder().network(network).initialLearningRate(0.01f).build();
    NetworkInput input = NetworkInput::Builder().network(network).name("input").dimensions({1}).dataType(DataType::FP32).build();
    FullyConnected fc = FullyConnected::Builder()
                            .network(network)
                            .featureInput(input.getFeatureOutput().value())
                            .numOutputFeatures(1)
                            .hasBias(false)
                            .weightsInitializer(testInitializer())
                            .noActivation()
                            .build();
    NetworkInput dailyLabels =
        NetworkInput::Builder().network(network).name("daily_labels").dimensions({1}).dataType(DataType::FP32).build();
    NetworkInput aggregateLabels =
        NetworkInput::Builder().network(network).name("aggregate_labels").dimensions({1}).dataType(DataType::FP32).build();
    NetworkInput rawLabels = NetworkInput::Builder().network(network).name("raw_labels").dimensions({1}).dataType(DataType::FP32).build();

    // These losses intentionally use the API-side objects returned by the builders, while Network stores cloned
    // layers internally.  The resolver must match by original tensor id and return the canonical raw loss tensors
    // from the evaluated network graph; Tensor object identity lookup alone is not strong enough here.
    MSE dailyLoss = MSE::Builder()
                        .network(network)
                        .predictions(fc.getFeatureOutput().value())
                        .labels(dailyLabels.getFeatureOutput().value())
                        .reportsBatchLoss()
                        .build();
    MSE aggregateLoss = MSE::Builder()
                            .network(network)
                            .predictions(fc.getFeatureOutput().value())
                            .labels(aggregateLabels.getFeatureOutput().value())
                            .reportsBatchLoss()
                            .build();
    MSE rawLoss = MSE::Builder()
                      .network(network)
                      .predictions(fc.getFeatureOutput().value())
                      .labels(rawLabels.getFeatureOutput().value())
                      .reportsRawLoss()
                      .build();

    NetworkOutput::Builder().network(network).name("daily_loss").inputTensor(dailyLoss.getLoss()).dataType(DataType::FP32).build();
    NetworkOutput::Builder()
        .network(network)
        .name("aggregate_loss")
        .inputTensor(aggregateLoss.getLoss())
        .dataType(DataType::FP32)
        .build();
    NetworkOutput::Builder().network(network).name("raw_loss").inputTensor(rawLoss.getLoss()).dataType(DataType::FP32).build();

    std::vector<Tensor> dailyRaw = network.getRawLossTensorsForTrainingRoots({dailyLoss.getLoss()});
    ASSERT_EQ(dailyRaw.size(), 1u);
    EXPECT_NE(dailyRaw[0].getOriginalId(), dailyLoss.getLoss().getOriginalId())
        << "Batch-shaped graph losses must resolve to the underlying raw physical loss tensor, not remain on the shaper output.";

    std::vector<Tensor> aggregateRaw = network.getRawLossTensorsForTrainingRoots({aggregateLoss.getLoss()});
    ASSERT_EQ(aggregateRaw.size(), 1u);
    EXPECT_NE(aggregateRaw[0].getOriginalId(), aggregateLoss.getLoss().getOriginalId());
    EXPECT_NE(aggregateRaw[0].getOriginalId(), dailyRaw[0].getOriginalId());

    std::vector<Tensor> rawRoot = network.getRawLossTensorsForTrainingRoots({rawLoss.getLoss()});
    ASSERT_EQ(rawRoot.size(), 1u);
    EXPECT_EQ(rawRoot[0].getOriginalId(), rawLoss.getLoss().getOriginalId())
        << "A raw graph loss is already the physical backward seed and should resolve to itself.";

    std::vector<Tensor> mixedRaw =
        network.getRawLossTensorsForTrainingRoots({dailyLoss.getLoss(), aggregateLoss.getLoss(), rawLoss.getLoss()});
    ASSERT_EQ(mixedRaw.size(), 3u);
    EXPECT_EQ(mixedRaw[0].getOriginalId(), dailyRaw[0].getOriginalId());
    EXPECT_EQ(mixedRaw[1].getOriginalId(), aggregateRaw[0].getOriginalId());
    EXPECT_EQ(mixedRaw[2].getOriginalId(), rawRoot[0].getOriginalId());

    std::vector<Tensor> duplicateRaw = network.getRawLossTensorsForTrainingRoots({dailyLoss.getLoss(), dailyLoss.getLoss()});
    ASSERT_EQ(duplicateRaw.size(), 1u);
    EXPECT_EQ(duplicateRaw[0].getOriginalId(), dailyRaw[0].getOriginalId());

    auto phase = makeNonOwningPhase("demand_forecast_phase", network);
    auto step = std::make_shared<TrainingStep>(
        "demand_forecast",
        std::vector<std::shared_ptr<TrainingPhase>>{phase},
        nullptr,
        std::vector<ParameterReference>{});
    TrainingProgram program(std::vector<std::shared_ptr<TrainingStep>>{step});

    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placed = network.place(/*batchSize=*/2, initDoneEvents, /*inferenceOnly=*/false);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    std::vector<StepExecutable> executables = program.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getActivePhaseNames(), (std::vector<std::string>{"demand_forecast_phase"}));
    ASSERT_EQ(executables[0].getObjectiveRoots().size(), 3u);
    EXPECT_EQ(executables[0].getObjectiveRoots()[0].getOriginalId(), dailyRaw[0].getOriginalId());
    EXPECT_EQ(executables[0].getObjectiveRoots()[1].getOriginalId(), aggregateRaw[0].getOriginalId());
    EXPECT_EQ(executables[0].getObjectiveRoots()[2].getOriginalId(), rawRoot[0].getOriginalId());

    const std::vector<Tensor>& resolvedPhaseRoots = executables[0].getResolvedObjectiveRoots();
    ASSERT_EQ(resolvedPhaseRoots.size(), 3u);
    EXPECT_EQ(resolvedPhaseRoots[0].getOriginalId(), dailyRaw[0].getOriginalId());
    EXPECT_EQ(resolvedPhaseRoots[1].getOriginalId(), aggregateRaw[0].getOriginalId());
    EXPECT_EQ(resolvedPhaseRoots[2].getOriginalId(), rawRoot[0].getOriginalId());

    expectRuntimeErrorContains(
        [&]() { (void)network.getRawLossTensorsForTrainingRoots({fc.getFeatureOutput().value()}); },
        "does not resolve to any physical loss layer");
}

TEST(TrainingProgramApi, TrainingProgramRejectsOutOfRangeAccessAndUnsupportedVersion) {
    auto step = std::make_shared<TrainingStep>("daily", std::vector<std::shared_ptr<TrainingPhase>>{makePhase("daily_phase")}, nullptr, std::vector<ParameterReference>{});
    TrainingProgram program(std::vector<std::shared_ptr<TrainingStep>>{step});

    EXPECT_THROW(static_cast<void>(program.getStep(1)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(program.getStepReference(1)), std::runtime_error);

    nlohmann::json programJson = program.architectureJson();
    programJson["version"] = "2.0.0";
    expectRuntimeErrorContains([&]() { (void)TrainingProgram::deserialize(programJson); }, "Unsupported TrainingProgram version: 2.0.0");
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

    auto discriminatorNetwork = std::make_shared<Network>("discriminator_phase_network");
    NetworkInput realImages = NetworkInput::Builder()
                                  .network(*discriminatorNetwork)
                                  .name("real_images")
                                  .dimensions({1})
                                  .dataType(DataType::FP32)
                                  .build();
    NetworkInput z = NetworkInput::Builder()
                         .network(*discriminatorNetwork)
                         .name("z")
                         .dimensions({1})
                         .dataType(DataType::FP32)
                         .build();
    MSE discriminatorLoss = MSE::Builder()
                                .network(*discriminatorNetwork)
                                .predictions(realImages.getFeatureOutput().value())
                                .labels(z.getFeatureOutput().value())
                                .build();
    NetworkOutput::Builder()
        .network(*discriminatorNetwork)
        .name("discriminator_prediction")
        .inputTensor(realImages.getFeatureOutput().value())
        .dataType(DataType::FP32)
        .build();
    NetworkOutput::Builder()
        .network(*discriminatorNetwork)
        .name("discriminator_loss")
        .inputTensor(discriminatorLoss.getLoss())
        .dataType(DataType::FP32)
        .build();
    auto phase = std::make_shared<TrainingPhase>("discriminator_phase", discriminatorNetwork);
    TrainingStep step("discriminator",
                      std::vector<std::shared_ptr<TrainingPhase>>{phase},
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
                              std::vector<std::shared_ptr<TrainingPhase>>{phase},
                              nullptr,
                              {},
                              1,
                              TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
                              {TrainingInputBinding("real_images", "a"), TrainingInputBinding("real_images", "b")}),
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
    MSE mse = MSE::Builder().network(network).predictions(fc.getFeatureOutput().value()).labels(labels.getFeatureOutput().value()).build();
    NetworkOutput::Builder().network(network).name("scores").inputTensor(fc.getFeatureOutput().value()).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(network).name("mse_loss").inputTensor(mse.getLoss()).dataType(DataType::FP32).build();
    auto generatorPhase = makeNonOwningPhase("generator_phase", network);
    const Tensor lossRoot = generatorPhase->getNetwork()->getLossRootTensors()[0];

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
                      std::vector<std::shared_ptr<TrainingPhase>>{generatorPhase},
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
    ASSERT_EQ(executable.getObjectiveRoots().size(), 1u);
    ASSERT_EQ(executable.getResolvedObjectiveRoots().size(), 1u);
    EXPECT_EQ(executable.getResolvedObjectiveRoots()[0].getOriginalId(), lossRoot.getOriginalId());
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
    EXPECT_EQ(executableJson.at("resolved_objective_root_count").get<uint64_t>(), 1u);
    EXPECT_EQ(executableJson.at("resolved_update_parameter_count").get<uint64_t>(), 1u);
    EXPECT_EQ(executableJson.at("input_bindings").at(0).at("network_input_name").get<std::string>(), "input");
    ASSERT_EQ(executableJson.at("resolved_input_bindings").size(), 2u);
    EXPECT_EQ(executableJson.at("required_batch_input_names"), (nlohmann::json::array({"labels", "z_generator"})));

    auto stepRef = std::make_shared<TrainingStep>(step);
    TrainingProgram program(std::vector<std::shared_ptr<TrainingStep>>{stepRef});
    std::vector<StepExecutable> executables = program.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getName(), "generator");

    PhaseNetworkFixture dailyPhaseFixture;
    auto dailyPhase = makePhase("daily_prediction", "input", "daily_prediction", true, false, true, true, &dailyPhaseFixture);
    PhaseNetworkFixture aggregatePhaseFixture;
    auto aggregatePhase = makePhase(
        "aggregate_prediction", "daily_prediction", "aggregate_prediction", false, true, true, false, &aggregatePhaseFixture);
    auto phasedStep = std::make_shared<TrainingStep>(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, aggregatePhase}, sgd, std::vector<ParameterReference>{});

    ASSERT_EQ(phasedStep->getActiveObjectiveRoots().size(), 1u);
    EXPECT_EQ(phasedStep->getActivePhaseNames(), (std::vector<std::string>{"daily_prediction"}));
    EXPECT_EQ(phasedStep->getActiveObjectiveRoots()[0], dailyPhaseFixture.lossRoot);
    nlohmann::json phasedJson = phasedStep->architectureJson();
    EXPECT_EQ(phasedJson.at("phases").at(0).at("network").at("name").get<std::string>(), "daily_prediction_network");
    EXPECT_FALSE(phasedJson.at("phases").at(1).at("enabled").get<bool>());

    aggregatePhase->enable();
    ASSERT_EQ(phasedStep->getActiveObjectiveRoots().size(), 2u);
    EXPECT_EQ(phasedStep->getActivePhaseNames(), (std::vector<std::string>{"daily_prediction", "aggregate_prediction"}));
    EXPECT_EQ(phasedStep->getActiveObjectiveRoots()[0], dailyPhaseFixture.lossRoot);
    EXPECT_EQ(phasedStep->getActiveObjectiveRoots()[1], aggregatePhaseFixture.lossRoot);
    EXPECT_NO_THROW((void)composeActivePhases(*phasedStep));

    dailyPhase->disable();
    EXPECT_THROW((void)composeActivePhases(*phasedStep), std::runtime_error);

    aggregatePhase->disable();
    EXPECT_THROW((void)composeActivePhases(*phasedStep), std::runtime_error);
    dailyPhase->enable();

    ParameterReference biases = fc.getParameterReference("biases");
    auto disabledStep = std::make_shared<TrainingStep>(
        "disabled", std::vector<std::shared_ptr<TrainingPhase>>{generatorPhase}, sgd, std::vector<ParameterReference>{weights, biases});
    auto enabledStep = std::make_shared<TrainingStep>(
        "enabled", std::vector<std::shared_ptr<TrainingPhase>>{generatorPhase}, sgd, std::vector<ParameterReference>{weights, biases});
    disabledStep->disable();
    TrainingProgram referenceProgram(std::vector<std::shared_ptr<TrainingStep>>{disabledStep, enabledStep});
    executables = referenceProgram.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getName(), "enabled");

    disabledStep->enable();
    enabledStep->disable();
    executables = referenceProgram.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getName(), "disabled");

    disabledStep->disable();
    EXPECT_THROW(static_cast<void>(referenceProgram.compile(*placed)), std::runtime_error);

    auto skippedBadDependencyPhase = makePhase("aggregate_prediction", "missing_daily", "aggregate_prediction", false, true, true, true);
    auto skippedBadDependencyStep = std::make_shared<TrainingStep>(
        "skipped_bad_dependency", std::vector<std::shared_ptr<TrainingPhase>>{skippedBadDependencyPhase}, sgd, std::vector<ParameterReference>{weights});
    skippedBadDependencyStep->disable();
    auto validReferenceStep = std::make_shared<TrainingStep>(
        "valid_reference", std::vector<std::shared_ptr<TrainingPhase>>{generatorPhase}, sgd, std::vector<ParameterReference>{weights});
    TrainingProgram skipDisabledInvalidProgram(std::vector<std::shared_ptr<TrainingStep>>{skippedBadDependencyStep, validReferenceStep});
    executables = skipDisabledInvalidProgram.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getName(), "valid_reference");

    skippedBadDependencyStep->enable();
    expectRuntimeErrorContains([&]() { (void)composeActivePhases(*skippedBadDependencyStep); },
                               "non-external input 'missing_daily'");

    ExecutableTrainingPlan plan = ExecutableTrainingPlan::compile(program, *placed);
    EXPECT_TRUE(plan.isInitialized());
    EXPECT_EQ(plan.getNumSteps(), 1u);
    EXPECT_EQ(plan.getTotalStepRepeatsPerIteration(), 3u);
    EXPECT_EQ(plan.getRequiredBatchInputNames(), (std::vector<std::string>{"labels", "z_generator"}));
    EXPECT_EQ(plan.getStep(0).getName(), "generator");
    EXPECT_THROW(plan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences()), std::runtime_error);

    TrainingStep nativeStep("native",
                            std::vector<std::shared_ptr<TrainingPhase>>{generatorPhase},
                            sgd,
                            {weights, biases},
                            1,
                            TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
                            {TrainingInputBinding("input", "z_generator")});
    auto nativeStepRef = std::make_shared<TrainingStep>(nativeStep);
    ExecutableTrainingPlan nativePlan =
        ExecutableTrainingPlan::compile(TrainingProgram(std::vector<std::shared_ptr<TrainingStep>>{nativeStepRef}), *placed);
    EXPECT_NO_THROW(nativePlan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences()));

    TrainingStep perParameterOptimizerStep("per_parameter", std::vector<std::shared_ptr<TrainingPhase>>{generatorPhase}, nullptr, {weights, biases});
    auto perParameterOptimizerStepRef = std::make_shared<TrainingStep>(perParameterOptimizerStep);
    ExecutableTrainingPlan perParameterOptimizerPlan =
        ExecutableTrainingPlan::compile(TrainingProgram(std::vector<std::shared_ptr<TrainingStep>>{perParameterOptimizerStepRef}), *placed);
    EXPECT_EQ(perParameterOptimizerPlan.getStep(0).getOptimizer(), nullptr);
    EXPECT_NO_THROW(perParameterOptimizerPlan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences()));

    TrainingStep singleStep("native_single_step", std::vector<std::shared_ptr<TrainingPhase>>{generatorPhase}, sgd, {weights, biases});
    auto singleStepRef = std::make_shared<TrainingStep>(singleStep);
    ExecutableTrainingPlan singleStepPlan =
        ExecutableTrainingPlan::compile(TrainingProgram(std::vector<std::shared_ptr<TrainingStep>>{singleStepRef}), *placed);
    EXPECT_NO_THROW(singleStepPlan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences()));
    nlohmann::json planJson = singleStepPlan.architectureJson();
    EXPECT_EQ(planJson.at("version").get<std::string>(), "1.0.0");
    EXPECT_EQ(planJson.at("step_count").get<uint64_t>(), 1u);
    EXPECT_EQ(planJson.at("total_step_repeats_per_iteration").get<uint64_t>(), 1u);

    EXPECT_THROW(placed->resolveParameterReference(ParameterReference(fc.getId(), "missing")), std::runtime_error);
    EXPECT_THROW(StepExecutable(TrainingStep("bad",
                                             std::vector<std::shared_ptr<TrainingPhase>>{generatorPhase},
                                             sgd,
                                             {ParameterReference(999999, "weights")}),
                                *placed),
                 std::runtime_error);
    auto foreignPhase = makePhase("foreign_phase");
    EXPECT_THROW(StepExecutable(TrainingStep("bad_loss", std::vector<std::shared_ptr<TrainingPhase>>{foreignPhase}, sgd, {weights}), *placed),
                 std::runtime_error);
    EXPECT_THROW(StepExecutable(TrainingStep("bad_input",
                                             std::vector<std::shared_ptr<TrainingPhase>>{generatorPhase},
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
