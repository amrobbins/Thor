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

TEST(TrainingPhaseApi, ConstructsWithNameLossRootsOutputsAndDependencies) {
    Tensor dailyLoss(DataType::FP32, {1});
    Tensor dailyForecast(DataType::FP32, {100});
    Tensor dailyQuantile(DataType::FP32, {100});

    TrainingPhase phase(
        "daily_prediction", {dailyLoss}, {{"forecast", dailyForecast}, {"quantile_high", dailyQuantile}}, {"feature_preprocessing"}, true);

    EXPECT_TRUE(phase.isInitialized());
    EXPECT_TRUE(phase.isEnabled());
    EXPECT_EQ(phase.getName(), "daily_prediction");
    ASSERT_EQ(phase.getLossRoots().size(), 1u);
    EXPECT_EQ(phase.getLossRoots()[0], dailyLoss);
    ASSERT_EQ(phase.getOutputs().size(), 2u);
    EXPECT_EQ(phase.getOutputs().at("forecast"), dailyForecast);
    EXPECT_EQ(phase.getOutputs().at("quantile_high"), dailyQuantile);
    ASSERT_EQ(phase.getDependsOn().size(), 1u);
    EXPECT_EQ(phase.getDependsOn()[0], "feature_preprocessing");
}

TEST(TrainingPhaseApi, EnableDisableMutatesPhaseState) {
    Tensor loss(DataType::FP32, {1});
    TrainingPhase phase("daily_prediction", {loss});

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

TEST(TrainingPhaseApi, AllowsForwardOnlyPhaseWithoutLossRoots) {
    Tensor forecast(DataType::FP32, {100});
    TrainingPhase phase("daily_forward", {}, {{"forecast", forecast}}, {}, true);

    EXPECT_TRUE(phase.isInitialized());
    EXPECT_TRUE(phase.getLossRoots().empty());
    ASSERT_EQ(phase.getOutputs().size(), 1u);
    EXPECT_EQ(phase.getOutputs().at("forecast"), forecast);
}

TEST(TrainingPhaseApi, RejectsInvalidConstruction) {
    Tensor loss(DataType::FP32, {1});
    Tensor output(DataType::FP32, {100});
    Tensor uninitialized;

    EXPECT_THROW(TrainingPhase("", {loss}), std::runtime_error);
    EXPECT_THROW(TrainingPhase("bad", {uninitialized}), std::runtime_error);
    EXPECT_THROW(TrainingPhase("bad", {}, {{"", output}}), std::runtime_error);
    EXPECT_THROW(TrainingPhase("bad", {}, {{"forecast", uninitialized}}), std::runtime_error);
    EXPECT_THROW(TrainingPhase("bad", {}, {}, {""}), std::runtime_error);
    EXPECT_THROW(TrainingPhase("daily_prediction", {}, {}, {"daily_prediction"}), std::runtime_error);
    EXPECT_THROW(TrainingPhase("aggregate_prediction", {}, {}, {"daily_prediction", "daily_prediction"}), std::runtime_error);
}

TEST(TrainingPhaseApi, SerializesAndDeserializes) {
    Tensor dailyLoss(DataType::FP32, {1});
    Tensor dailyForecast(DataType::FP32, {100});
    Tensor dailyQuantile(DataType::FP32, {100});

    TrainingPhase phase(
        "daily_prediction", {dailyLoss}, {{"forecast", dailyForecast}, {"quantile_high", dailyQuantile}}, {"feature_preprocessing"}, false);

    nlohmann::json j = phase.architectureJson();
    EXPECT_EQ(j.at("version").get<std::string>(), "1.0.0");
    EXPECT_EQ(j.at("name").get<std::string>(), "daily_prediction");
    EXPECT_FALSE(j.at("enabled").get<bool>());
    ASSERT_EQ(j.at("loss_roots").size(), 1u);
    ASSERT_EQ(j.at("outputs").size(), 2u);
    ASSERT_EQ(j.at("depends_on").size(), 1u);
    EXPECT_EQ(j.at("depends_on").at(0).get<std::string>(), "feature_preprocessing");

    TrainingPhase restored = TrainingPhase::deserialize(j);
    EXPECT_TRUE(restored.isInitialized());
    EXPECT_FALSE(restored.isEnabled());
    EXPECT_EQ(restored.getName(), phase.getName());
    ASSERT_EQ(restored.getLossRoots().size(), 1u);
    EXPECT_EQ(restored.getLossRoots()[0].getOriginalId(), dailyLoss.getOriginalId());
    ASSERT_EQ(restored.getOutputs().size(), 2u);
    EXPECT_EQ(restored.getOutputs().at("forecast").getOriginalId(), dailyForecast.getOriginalId());
    EXPECT_EQ(restored.getOutputs().at("quantile_high").getOriginalId(), dailyQuantile.getOriginalId());
    ASSERT_EQ(restored.getDependsOn().size(), 1u);
    EXPECT_EQ(restored.getDependsOn()[0], "feature_preprocessing");

    nlohmann::json badVersion = j;
    badVersion["version"] = "0.0.0";
    EXPECT_THROW(TrainingPhase::deserialize(badVersion), std::runtime_error);
}

TEST(TrainingPhaseApi, SerializesAndDeserializesEnabledAndDisabledStateAndRejectsUnsupportedVersion) {
    Tensor loss(DataType::FP32, {1});

    TrainingPhase enabledPhase("enabled_phase", {loss}, {}, {}, true);
    TrainingPhase disabledPhase("disabled_phase", {loss}, {}, {}, false);

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
    Tensor loss(DataType::FP32, {1});
    Tensor output(DataType::FP32, {100});
    Tensor uninitialized;

    expectRuntimeErrorContains([&]() { TrainingPhase("", {loss}); }, "requires a non-empty name");
    expectRuntimeErrorContains([&]() { TrainingPhase("bad", {uninitialized}); }, "loss roots must all be initialized");
    expectRuntimeErrorContains([&]() { TrainingPhase("bad", {}, {{"", output}}); }, "output names must be non-empty");
    expectRuntimeErrorContains([&]() { TrainingPhase("bad", {}, {{"forecast", uninitialized}}); }, "outputs must all be initialized");
    expectRuntimeErrorContains([&]() { TrainingPhase("bad", {}, {}, {""}); }, "dependency names must be non-empty");
    expectRuntimeErrorContains([&]() { TrainingPhase("daily_prediction", {}, {}, {"daily_prediction"}); }, "cannot depend on itself");
    expectRuntimeErrorContains([&]() { TrainingPhase("aggregate_prediction", {}, {}, {"daily_prediction", "daily_prediction"}); },
                               "contains duplicate dependency 'daily_prediction'");
}

TEST(TrainingProgramApi, TrainingStepLegacyLossRootsNormalizeToSinglePhase) {
    Tensor loss(DataType::FP32, {1});
    auto sgd = Sgd::Builder().initialLearningRate(0.01f).build();

    TrainingStep step("daily", {loss}, sgd, {});

    EXPECT_TRUE(step.isInitialized());
    EXPECT_TRUE(step.isEnabled());
    ASSERT_EQ(step.getLossRoots().size(), 1u);
    EXPECT_EQ(step.getLossRoots()[0], loss);
    ASSERT_EQ(step.getActiveLossRoots().size(), 1u);
    EXPECT_EQ(step.getActiveLossRoots()[0], loss);
    ASSERT_EQ(step.getPhases().size(), 1u);
    EXPECT_TRUE(step.getPhases()[0]->isInitialized());
    EXPECT_TRUE(step.getPhases()[0]->isEnabled());
    EXPECT_EQ(step.getPhases()[0]->getName(), "daily_phase");
    ASSERT_EQ(step.getPhases()[0]->getLossRoots().size(), 1u);
    EXPECT_EQ(step.getPhases()[0]->getLossRoots()[0], loss);
}

TEST(TrainingProgramApi, TrainingStepEnableDisableMutatesStepStateAndActiveLossRoots) {
    Tensor loss(DataType::FP32, {1});
    TrainingStep step("daily", {loss}, nullptr, {});

    EXPECT_TRUE(step.isEnabled());
    ASSERT_EQ(step.getActiveLossRoots().size(), 1u);
    step.disable();
    EXPECT_FALSE(step.isEnabled());
    EXPECT_TRUE(step.getActiveLossRoots().empty());
    step.enable();
    EXPECT_TRUE(step.isEnabled());
    ASSERT_EQ(step.getActiveLossRoots().size(), 1u);
    step.setEnabled(false);
    EXPECT_FALSE(step.isEnabled());
    step.setEnabled(true);
    EXPECT_TRUE(step.isEnabled());
}

TEST(TrainingProgramApi, TrainingStepActiveLossRootsComeOnlyFromEnabledPhases) {
    Tensor dailyLoss(DataType::FP32, {1});
    Tensor aggregateLoss(DataType::FP32, {1});
    auto dailyPhase = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{dailyLoss});
    auto aggregatePhase = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                          std::vector<Tensor>{aggregateLoss},
                                                          std::map<std::string, Tensor>{},
                                                          std::vector<std::string>{"daily_prediction"},
                                                          false);

    TrainingStep step("demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, aggregatePhase}, nullptr, {});

    ASSERT_EQ(step.getLossRoots().size(), 2u);
    std::vector<Tensor> activeRoots = step.getActiveLossRoots();
    ASSERT_EQ(activeRoots.size(), 1u);
    EXPECT_EQ(activeRoots[0], dailyLoss);

    aggregatePhase->enable();
    activeRoots = step.getActiveLossRoots();
    ASSERT_EQ(activeRoots.size(), 2u);
    EXPECT_EQ(activeRoots[0], dailyLoss);
    EXPECT_EQ(activeRoots[1], aggregateLoss);

    dailyPhase->disable();
    EXPECT_THROW(step.getActiveLossRoots(), std::runtime_error);
}

TEST(TrainingProgramApi, TrainingStepActivePhaseNamesComeOnlyFromEnabledPhasesAndMayIncludeForwardOnlyPhases) {
    Tensor preprocessedFeatures(DataType::FP32, {16});
    Tensor dailyLoss(DataType::FP32, {1});
    Tensor aggregateLoss(DataType::FP32, {1});

    auto preprocessingPhase = std::make_shared<TrainingPhase>(
        "feature_preprocessing", std::vector<Tensor>{}, std::map<std::string, Tensor>{{"features", preprocessedFeatures}});
    auto dailyPhase = std::make_shared<TrainingPhase>("daily_prediction",
                                                      std::vector<Tensor>{dailyLoss},
                                                      std::map<std::string, Tensor>{},
                                                      std::vector<std::string>{"feature_preprocessing"});
    auto aggregatePhase = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                          std::vector<Tensor>{aggregateLoss},
                                                          std::map<std::string, Tensor>{},
                                                          std::vector<std::string>{"daily_prediction"},
                                                          false);

    TrainingStep step(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{preprocessingPhase, dailyPhase, aggregatePhase}, nullptr, {});

    EXPECT_EQ(step.getActivePhaseNames(), (std::vector<std::string>{"feature_preprocessing", "daily_prediction"}));
    std::vector<Tensor> activeRoots = step.getActiveLossRoots();
    ASSERT_EQ(activeRoots.size(), 1u);
    EXPECT_EQ(activeRoots[0], dailyLoss);

    aggregatePhase->enable();
    EXPECT_EQ(step.getActivePhaseNames(), (std::vector<std::string>{"feature_preprocessing", "daily_prediction", "aggregate_prediction"}));
    activeRoots = step.getActiveLossRoots();
    ASSERT_EQ(activeRoots.size(), 2u);
    EXPECT_EQ(activeRoots[0], dailyLoss);
    EXPECT_EQ(activeRoots[1], aggregateLoss);
}

TEST(TrainingProgramApi, TrainingStepDependencyValidationSkipsDisabledPhasesAndDisabledSteps) {
    Tensor dailyLoss(DataType::FP32, {1});
    Tensor aggregateLoss(DataType::FP32, {1});

    auto dailyPhase = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{dailyLoss});
    auto disabledAggregateWithMissingDependency = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                                                  std::vector<Tensor>{aggregateLoss},
                                                                                  std::map<std::string, Tensor>{},
                                                                                  std::vector<std::string>{"missing_phase"},
                                                                                  false);
    TrainingStep disabledPhaseStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, disabledAggregateWithMissingDependency}, nullptr, {});
    EXPECT_NO_THROW(disabledPhaseStep.validateEnabledPhaseDependencies());
    EXPECT_EQ(disabledPhaseStep.getActivePhaseNames(), (std::vector<std::string>{"daily_prediction"}));

    disabledAggregateWithMissingDependency->enable();
    expectRuntimeErrorContains([&]() { disabledPhaseStep.validateEnabledPhaseDependencies(); },
                               "enabled phase 'aggregate_prediction' depends on missing phase 'missing_phase'");

    auto disabledDaily = std::make_shared<TrainingPhase>(
        "daily_prediction", std::vector<Tensor>{dailyLoss}, std::map<std::string, Tensor>{}, std::vector<std::string>{}, false);
    auto aggregateDependsOnDisabledDaily = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                                           std::vector<Tensor>{aggregateLoss},
                                                                           std::map<std::string, Tensor>{},
                                                                           std::vector<std::string>{"daily_prediction"},
                                                                           true);
    TrainingStep disabledStep("demand_forecast",
                              std::vector<std::shared_ptr<TrainingPhase>>{disabledDaily, aggregateDependsOnDisabledDaily},
                              nullptr,
                              {},
                              1,
                              TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
                              {},
                              false);
    EXPECT_NO_THROW(disabledStep.validateEnabledPhaseDependencies());
    EXPECT_TRUE(disabledStep.getActivePhaseNames().empty());
    EXPECT_TRUE(disabledStep.getActiveLossRoots().empty());
}

TEST(TrainingProgramApi, TrainingStepDependencyErrorsAreSpecificAndSearchable) {
    Tensor loss(DataType::FP32, {1});

    auto disabledDaily = std::make_shared<TrainingPhase>(
        "daily_prediction", std::vector<Tensor>{loss}, std::map<std::string, Tensor>{}, std::vector<std::string>{}, false);
    auto aggregateDependsOnDisabled = std::make_shared<TrainingPhase>(
        "aggregate_prediction", std::vector<Tensor>{loss}, std::map<std::string, Tensor>{}, std::vector<std::string>{"daily_prediction"});
    TrainingStep disabledDependencyStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{disabledDaily, aggregateDependsOnDisabled}, nullptr, {});
    expectRuntimeErrorContains(
        [&]() { disabledDependencyStep.validateEnabledPhaseDependencies(); },
        "TrainingStep 'demand_forecast' enabled phase 'aggregate_prediction' depends on disabled phase 'daily_prediction'.");

    auto missingDependencyPhase = std::make_shared<TrainingPhase>(
        "aggregate_prediction", std::vector<Tensor>{loss}, std::map<std::string, Tensor>{}, std::vector<std::string>{"daily_prediction"});
    TrainingStep missingDependencyStep("demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{missingDependencyPhase}, nullptr, {});
    expectRuntimeErrorContains([&]() { missingDependencyStep.validateEnabledPhaseDependencies(); },
                               "enabled phase 'aggregate_prediction' depends on missing phase 'daily_prediction'");

    auto aggregateForwardDependency = std::make_shared<TrainingPhase>(
        "aggregate_prediction", std::vector<Tensor>{loss}, std::map<std::string, Tensor>{}, std::vector<std::string>{"daily_prediction"});
    auto dailyLater = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{loss});
    TrainingStep forwardDependencyStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{aggregateForwardDependency, dailyLater}, nullptr, {});
    expectRuntimeErrorContains([&]() { forwardDependencyStep.validateEnabledPhaseDependencies(); },
                               "but that dependency does not appear earlier in the step");
}

TEST(TrainingProgramApi, TrainingStepRejectsInvalidPhaseDependencies) {
    Tensor loss(DataType::FP32, {1});

    auto disabledDaily = std::make_shared<TrainingPhase>(
        "daily_prediction", std::vector<Tensor>{loss}, std::map<std::string, Tensor>{}, std::vector<std::string>{}, false);
    auto aggregateDependsOnDisabled = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                                      std::vector<Tensor>{loss},
                                                                      std::map<std::string, Tensor>{},
                                                                      std::vector<std::string>{"daily_prediction"},
                                                                      true);
    TrainingStep disabledDependencyStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{disabledDaily, aggregateDependsOnDisabled}, nullptr, {});
    EXPECT_THROW(disabledDependencyStep.validateEnabledPhaseDependencies(), std::runtime_error);
    EXPECT_THROW(disabledDependencyStep.getActiveLossRoots(), std::runtime_error);

    auto aggregateMissingDependency = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                                      std::vector<Tensor>{loss},
                                                                      std::map<std::string, Tensor>{},
                                                                      std::vector<std::string>{"daily_prediction"},
                                                                      true);
    TrainingStep missingDependencyStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{aggregateMissingDependency}, nullptr, {});
    EXPECT_THROW(missingDependencyStep.validateEnabledPhaseDependencies(), std::runtime_error);

    auto aggregateForwardDependency = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                                      std::vector<Tensor>{loss},
                                                                      std::map<std::string, Tensor>{},
                                                                      std::vector<std::string>{"daily_prediction"},
                                                                      true);
    auto dailyLater = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{loss});
    TrainingStep forwardDependencyStep(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{aggregateForwardDependency, dailyLater}, nullptr, {});
    EXPECT_THROW(forwardDependencyStep.validateEnabledPhaseDependencies(), std::runtime_error);
}

TEST(TrainingProgramApi, TrainingStepRejectsInvalidPhaseLists) {
    Tensor loss(DataType::FP32, {1});
    auto dailyA = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{loss});
    auto dailyB = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{loss});

    EXPECT_THROW(TrainingStep("bad", std::vector<std::shared_ptr<TrainingPhase>>{}, nullptr, {}), std::runtime_error);
    EXPECT_THROW(TrainingStep("bad", std::vector<std::shared_ptr<TrainingPhase>>{nullptr}, nullptr, {}), std::runtime_error);
    EXPECT_THROW(TrainingStep("bad", std::vector<std::shared_ptr<TrainingPhase>>{dailyA, dailyB}, nullptr, {}), std::runtime_error);
}

TEST(TrainingProgramApi, TrainingStepSerializesPhaseAwareExecutionView) {
    Tensor dailyLoss(DataType::FP32, {1});
    Tensor aggregateLoss(DataType::FP32, {1});
    auto dailyPhase = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{dailyLoss});
    auto aggregatePhase = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                          std::vector<Tensor>{aggregateLoss},
                                                          std::map<std::string, Tensor>{},
                                                          std::vector<std::string>{"daily_prediction"},
                                                          false);
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
    EXPECT_EQ(j.at("version").get<std::string>(), "1.1.0");
    EXPECT_EQ(j.at("name").get<std::string>(), "demand_forecast");
    EXPECT_FALSE(j.at("enabled").get<bool>());
    ASSERT_EQ(j.at("loss_roots").size(), 2u);
    ASSERT_EQ(j.at("phases").size(), 2u);
    EXPECT_EQ(j.at("phases").at(0).at("name").get<std::string>(), "daily_prediction");
    EXPECT_EQ(j.at("phases").at(1).at("name").get<std::string>(), "aggregate_prediction");
    EXPECT_FALSE(j.at("phases").at(1).at("enabled").get<bool>());

    TrainingStep restored = TrainingStep::deserialize(j);
    EXPECT_TRUE(restored.isInitialized());
    EXPECT_FALSE(restored.isEnabled());
    EXPECT_EQ(restored.getName(), step.getName());
    EXPECT_EQ(restored.getRepeatCount(), step.getRepeatCount());
    EXPECT_EQ(restored.getGradientClearPolicy(), step.getGradientClearPolicy());
    ASSERT_EQ(restored.getLossRoots().size(), 2u);
    ASSERT_EQ(restored.getPhases().size(), 2u);
    EXPECT_EQ(restored.getPhases()[0]->getName(), "daily_prediction");
    EXPECT_EQ(restored.getPhases()[1]->getName(), "aggregate_prediction");
    EXPECT_FALSE(restored.getPhases()[1]->isEnabled());
    EXPECT_TRUE(restored.getActiveLossRoots().empty());
    EXPECT_EQ(restored.getUpdateParameters(), step.getUpdateParameters());
    ASSERT_EQ(restored.getInputBindings().size(), 1u);
    EXPECT_EQ(restored.getInputBindings()[0].getBatchInputName(), "input");
    EXPECT_NE(restored.getOptimizer(), nullptr);
}

TEST(TrainingProgramApi, TrainingStepSerializationPreservesPhaseEnablementAndLegacyJsonStillDeserializes) {
    Tensor dailyLoss(DataType::FP32, {1});
    Tensor aggregateLoss(DataType::FP32, {1});
    auto dailyPhase = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{dailyLoss});
    auto aggregatePhase = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                          std::vector<Tensor>{aggregateLoss},
                                                          std::map<std::string, Tensor>{},
                                                          std::vector<std::string>{"daily_prediction"},
                                                          false);

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
    EXPECT_EQ(restoredPhasedStep.getPhases()[1]->getDependsOn(), (std::vector<std::string>{"daily_prediction"}));
    EXPECT_EQ(restoredPhasedStep.getActivePhaseNames(), (std::vector<std::string>{"daily_prediction"}));

    nlohmann::json legacyJson = phasedJson;
    legacyJson["version"] = "1.0.0";
    legacyJson.erase("phases");
    TrainingStep restoredLegacyStep = TrainingStep::deserialize(legacyJson);
    ASSERT_EQ(restoredLegacyStep.getPhases().size(), 1u);
    EXPECT_EQ(restoredLegacyStep.getPhases()[0]->getName(), "demand_forecast_phase");
    EXPECT_EQ(restoredLegacyStep.getActivePhaseNames(), (std::vector<std::string>{"demand_forecast_phase"}));
    ASSERT_EQ(restoredLegacyStep.getActiveLossRoots().size(), 2u);

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
    EXPECT_EQ(j.at("version").get<std::string>(), "1.1.0");
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

TEST(TrainingProgramApi, TrainingStepAllowsUpdatesWithoutStepOptimizerForPerParameterOptimizers) {
    Tensor loss(DataType::FP32, {1});
    ParameterReference weights(123, "weights");

    TrainingStep step("per_parameter", {loss}, nullptr, {weights});
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
    Tensor dLoss(DataType::FP32, {1});
    Tensor gLoss(DataType::FP32, {1});
    auto dSgd = Sgd::Builder().initialLearningRate(0.01f).build();
    auto gSgd = Sgd::Builder().initialLearningRate(0.02f).build();

    auto dStep = std::make_shared<TrainingStep>(
        "discriminator", std::vector<Tensor>{dLoss}, dSgd, std::vector<ParameterReference>{ParameterReference(1, "weights")});
    auto gStep = std::make_shared<TrainingStep>(
        "generator", std::vector<Tensor>{gLoss}, gSgd, std::vector<ParameterReference>{ParameterReference(2, "weights")});

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
    Tensor loss(DataType::FP32, {1});
    auto step = std::make_shared<TrainingStep>("daily", std::vector<Tensor>{loss}, nullptr, std::vector<ParameterReference>{});

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
    Tensor loss(DataType::FP32, {1});
    auto step = std::make_shared<TrainingStep>("daily", std::vector<Tensor>{loss}, nullptr, std::vector<ParameterReference>{});

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

TEST(TrainingProgramApi, DisabledPhasesDoNotValidateInactiveDependenciesUntilEnabled) {
    Tensor dailyLoss(DataType::FP32, {1});
    Tensor aggregateLoss(DataType::FP32, {1});

    auto dailyPhase = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{dailyLoss});
    auto disabledAggregateWithMissingDependency = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                                                  std::vector<Tensor>{aggregateLoss},
                                                                                  std::map<std::string, Tensor>{},
                                                                                  std::vector<std::string>{"not_declared_yet"},
                                                                                  false);

    TrainingStep step("demand_forecast",
                      std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, disabledAggregateWithMissingDependency},
                      nullptr,
                      std::vector<ParameterReference>{});

    EXPECT_NO_THROW(step.validateEnabledPhaseDependencies());
    EXPECT_EQ(step.getActivePhaseNames(), (std::vector<std::string>{"daily_prediction"}));
    std::vector<Tensor> activeRoots = step.getActiveLossRoots();
    ASSERT_EQ(activeRoots.size(), 1u);
    EXPECT_EQ(activeRoots[0], dailyLoss);

    disabledAggregateWithMissingDependency->enable();
    EXPECT_THROW(step.validateEnabledPhaseDependencies(), std::runtime_error);
    EXPECT_THROW(step.getActiveLossRoots(), std::runtime_error);
}

TEST(TrainingProgramApi, PhaseMutationAfterProgramConstructionIsVisibleThroughStep) {
    Tensor dailyLoss(DataType::FP32, {1});
    Tensor aggregateLoss(DataType::FP32, {1});
    auto dailyPhase = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{dailyLoss});
    auto aggregatePhase = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                          std::vector<Tensor>{aggregateLoss},
                                                          std::map<std::string, Tensor>{},
                                                          std::vector<std::string>{"daily_prediction"},
                                                          false);
    auto step = std::make_shared<TrainingStep>("demand_forecast",
                                               std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, aggregatePhase},
                                               nullptr,
                                               std::vector<ParameterReference>{});

    TrainingProgram program(std::vector<std::shared_ptr<TrainingStep>>{step});
    std::vector<Tensor> activeRoots = program.getStep(0).getActiveLossRoots();
    ASSERT_EQ(activeRoots.size(), 1u);
    EXPECT_EQ(activeRoots[0], dailyLoss);

    aggregatePhase->enable();
    activeRoots = program.getStep(0).getActiveLossRoots();
    ASSERT_EQ(activeRoots.size(), 2u);
    EXPECT_EQ(activeRoots[0], dailyLoss);
    EXPECT_EQ(activeRoots[1], aggregateLoss);

    dailyPhase->disable();
    EXPECT_THROW(program.getStep(0).getActiveLossRoots(), std::runtime_error);
}

TEST(TrainingProgramApi, TrainingProgramRejectsEmptyProgramsAtConstructionSerializationAndCompileTime) {
    EXPECT_THROW(TrainingProgram(std::vector<std::shared_ptr<TrainingStep>>{}), std::runtime_error);

    TrainingProgram program;
    EXPECT_FALSE(program.isInitialized());
    EXPECT_THROW(program.architectureJson(), std::runtime_error);

    nlohmann::json emptyProgramJson{{"version", "1.0.0"}, {"steps", nlohmann::json::array()}};
    EXPECT_THROW(TrainingProgram::deserialize(emptyProgramJson), std::runtime_error);
}

TEST(TrainingProgramApi, NetworkResolvesApiSideShapedLossRootsToCanonicalRawLossLayers) {
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
        << "Batch-shaped loss roots must resolve to the underlying raw physical loss tensor, not remain on the shaper output.";

    std::vector<Tensor> aggregateRaw = network.getRawLossTensorsForTrainingRoots({aggregateLoss.getLoss()});
    ASSERT_EQ(aggregateRaw.size(), 1u);
    EXPECT_NE(aggregateRaw[0].getOriginalId(), aggregateLoss.getLoss().getOriginalId());
    EXPECT_NE(aggregateRaw[0].getOriginalId(), dailyRaw[0].getOriginalId());

    std::vector<Tensor> rawRoot = network.getRawLossTensorsForTrainingRoots({rawLoss.getLoss()});
    ASSERT_EQ(rawRoot.size(), 1u);
    EXPECT_EQ(rawRoot[0].getOriginalId(), rawLoss.getLoss().getOriginalId())
        << "A raw loss root is already the physical backward seed and should resolve to itself.";

    std::vector<Tensor> mixedRaw =
        network.getRawLossTensorsForTrainingRoots({dailyLoss.getLoss(), aggregateLoss.getLoss(), rawLoss.getLoss()});
    ASSERT_EQ(mixedRaw.size(), 3u);
    EXPECT_EQ(mixedRaw[0].getOriginalId(), dailyRaw[0].getOriginalId());
    EXPECT_EQ(mixedRaw[1].getOriginalId(), aggregateRaw[0].getOriginalId());
    EXPECT_EQ(mixedRaw[2].getOriginalId(), rawRoot[0].getOriginalId());

    std::vector<Tensor> duplicateRaw = network.getRawLossTensorsForTrainingRoots({dailyLoss.getLoss(), dailyLoss.getLoss()});
    ASSERT_EQ(duplicateRaw.size(), 1u);
    EXPECT_EQ(duplicateRaw[0].getOriginalId(), dailyRaw[0].getOriginalId());

    auto dailyPhase = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{dailyLoss.getLoss()});
    auto aggregatePhase = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                          std::vector<Tensor>{aggregateLoss.getLoss()},
                                                          std::map<std::string, Tensor>{},
                                                          std::vector<std::string>{"daily_prediction"},
                                                          true);
    auto rawPhase = std::make_shared<TrainingPhase>("raw_auxiliary", std::vector<Tensor>{rawLoss.getLoss()});
    auto step = std::make_shared<TrainingStep>(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, aggregatePhase, rawPhase}, nullptr, std::vector<ParameterReference>{});
    TrainingProgram program(std::vector<std::shared_ptr<TrainingStep>>{step});

    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placed = network.place(/*batchSize=*/2, initDoneEvents, /*inferenceOnly=*/false);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    std::vector<StepExecutable> executables = program.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getActivePhaseNames(),
              (std::vector<std::string>{"daily_prediction", "aggregate_prediction", "raw_auxiliary"}));
    ASSERT_EQ(executables[0].getLossRoots().size(), 3u);
    EXPECT_EQ(executables[0].getLossRoots()[0].getOriginalId(), dailyLoss.getLoss().getOriginalId());
    EXPECT_EQ(executables[0].getLossRoots()[1].getOriginalId(), aggregateLoss.getLoss().getOriginalId());
    EXPECT_EQ(executables[0].getLossRoots()[2].getOriginalId(), rawLoss.getLoss().getOriginalId());

    const std::vector<Tensor>& resolvedPhaseRoots = executables[0].getResolvedLossRoots();
    ASSERT_EQ(resolvedPhaseRoots.size(), 3u);
    EXPECT_EQ(resolvedPhaseRoots[0].getOriginalId(), dailyLoss.getLoss().getOriginalId());
    EXPECT_EQ(resolvedPhaseRoots[1].getOriginalId(), aggregateLoss.getLoss().getOriginalId());
    EXPECT_EQ(resolvedPhaseRoots[2].getOriginalId(), rawLoss.getLoss().getOriginalId());

    expectRuntimeErrorContains(
        [&]() { (void)network.getRawLossTensorsForTrainingRoots({fc.getFeatureOutput().value()}); },
        "does not resolve to any physical loss layer");
}

TEST(TrainingProgramApi, TrainingProgramRejectsOutOfRangeAccessAndUnsupportedVersion) {
    Tensor loss(DataType::FP32, {1});
    auto step = std::make_shared<TrainingStep>("daily", std::vector<Tensor>{loss}, nullptr, std::vector<ParameterReference>{});
    TrainingProgram program(std::vector<std::shared_ptr<TrainingStep>>{step});

    EXPECT_THROW(static_cast<void>(program.getStep(1)), std::runtime_error);
    EXPECT_THROW(program.getStepReference(1), std::runtime_error);

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
    NetworkOutput::Builder()
        .network(network)
        .name("labels_out")
        .inputTensor(labels.getFeatureOutput().value())
        .dataType(DataType::FP32)
        .build();

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

    auto stepRef = std::make_shared<TrainingStep>(step);
    TrainingProgram program(std::vector<std::shared_ptr<TrainingStep>>{stepRef});
    std::vector<StepExecutable> executables = program.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getName(), "generator");

    const Tensor aggregateLossRoot = labels.getFeatureOutput().value();
    auto dailyPhase = std::make_shared<TrainingPhase>("daily_prediction", std::vector<Tensor>{lossRoot});
    auto aggregatePhase = std::make_shared<TrainingPhase>("aggregate_prediction",
                                                          std::vector<Tensor>{aggregateLossRoot},
                                                          std::map<std::string, Tensor>{},
                                                          std::vector<std::string>{"daily_prediction"},
                                                          false);
    auto phasedStep = std::make_shared<TrainingStep>(
        "demand_forecast", std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, aggregatePhase}, sgd, std::vector<ParameterReference>{});
    TrainingProgram phasedProgram(std::vector<std::shared_ptr<TrainingStep>>{phasedStep});

    executables = phasedProgram.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getName(), "demand_forecast");
    EXPECT_EQ(executables[0].getActivePhaseNames(), (std::vector<std::string>{"daily_prediction"}));
    ASSERT_EQ(executables[0].getLossRoots().size(), 1u);
    EXPECT_EQ(executables[0].getLossRoots()[0].getOriginalId(), lossRoot.getOriginalId());
    EXPECT_EQ(executables[0].getUpdateParameterReferences().size(), network.getTrainableParameterReferences().size());
    nlohmann::json phasedJson = executables[0].architectureJson();
    EXPECT_EQ(phasedJson.at("active_phase_names"), (nlohmann::json::array({"daily_prediction"})));

    aggregatePhase->enable();
    executables = phasedProgram.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getActivePhaseNames(), (std::vector<std::string>{"daily_prediction", "aggregate_prediction"}));
    ASSERT_EQ(executables[0].getLossRoots().size(), 2u);
    EXPECT_EQ(executables[0].getLossRoots()[0].getOriginalId(), lossRoot.getOriginalId());
    EXPECT_EQ(executables[0].getLossRoots()[1].getOriginalId(), aggregateLossRoot.getOriginalId());

    dailyPhase->disable();
    EXPECT_THROW(phasedProgram.compile(*placed), std::runtime_error);

    aggregatePhase->disable();
    EXPECT_THROW(phasedProgram.compile(*placed), std::runtime_error);
    dailyPhase->enable();

    ParameterReference biases = fc.getParameterReference("biases");
    auto disabledStep =
        std::make_shared<TrainingStep>("disabled", std::vector<Tensor>{lossRoot}, sgd, std::vector<ParameterReference>{weights, biases});
    auto enabledStep =
        std::make_shared<TrainingStep>("enabled", std::vector<Tensor>{lossRoot}, sgd, std::vector<ParameterReference>{weights, biases});
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
    EXPECT_THROW(referenceProgram.compile(*placed), std::runtime_error);

    auto skippedBadDependencyStep = std::make_shared<TrainingStep>(
        "skipped_bad_dependency",
        std::vector<std::shared_ptr<TrainingPhase>>{std::make_shared<TrainingPhase>("aggregate_prediction",
                                                                                    std::vector<Tensor>{lossRoot},
                                                                                    std::map<std::string, Tensor>{},
                                                                                    std::vector<std::string>{"missing_daily"},
                                                                                    true)},
        sgd,
        std::vector<ParameterReference>{weights});
    skippedBadDependencyStep->disable();
    auto validReferenceStep =
        std::make_shared<TrainingStep>("valid_reference", std::vector<Tensor>{lossRoot}, sgd, std::vector<ParameterReference>{weights});
    TrainingProgram skipDisabledInvalidProgram(std::vector<std::shared_ptr<TrainingStep>>{skippedBadDependencyStep, validReferenceStep});
    executables = skipDisabledInvalidProgram.compile(*placed);
    ASSERT_EQ(executables.size(), 1u);
    EXPECT_EQ(executables[0].getName(), "valid_reference");

    skippedBadDependencyStep->enable();
    expectRuntimeErrorContains([&]() { (void)skipDisabledInvalidProgram.compile(*placed); },
                               "enabled phase 'aggregate_prediction' depends on missing phase 'missing_daily'");

    ExecutableTrainingPlan plan = ExecutableTrainingPlan::compile(program, *placed);
    EXPECT_TRUE(plan.isInitialized());
    EXPECT_EQ(plan.getNumSteps(), 1u);
    EXPECT_EQ(plan.getTotalStepRepeatsPerIteration(), 3u);
    EXPECT_EQ(plan.getRequiredBatchInputNames(), (std::vector<std::string>{"labels", "z_generator"}));
    EXPECT_EQ(plan.getStep(0).getName(), "generator");
    EXPECT_THROW(plan.assertLegacyLocalExecutorCompatible(), std::runtime_error);
    EXPECT_THROW(plan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences()), std::runtime_error);

    TrainingStep nativeStep("native",
                            {lossRoot},
                            sgd,
                            {weights, biases},
                            1,
                            TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
                            {TrainingInputBinding("input", "z_generator")});
    auto nativeStepRef = std::make_shared<TrainingStep>(nativeStep);
    ExecutableTrainingPlan nativePlan =
        ExecutableTrainingPlan::compile(TrainingProgram(std::vector<std::shared_ptr<TrainingStep>>{nativeStepRef}), *placed);
    EXPECT_NO_THROW(nativePlan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences()));
    EXPECT_THROW(nativePlan.assertLegacyLocalExecutorCompatible(), std::runtime_error);

    TrainingStep perParameterOptimizerStep("per_parameter", {lossRoot}, nullptr, {weights, biases});
    auto perParameterOptimizerStepRef = std::make_shared<TrainingStep>(perParameterOptimizerStep);
    ExecutableTrainingPlan perParameterOptimizerPlan =
        ExecutableTrainingPlan::compile(TrainingProgram(std::vector<std::shared_ptr<TrainingStep>>{perParameterOptimizerStepRef}), *placed);
    EXPECT_EQ(perParameterOptimizerPlan.getStep(0).getOptimizer(), nullptr);
    EXPECT_NO_THROW(perParameterOptimizerPlan.assertLegacyLocalExecutorCompatible());
    EXPECT_NO_THROW(perParameterOptimizerPlan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences()));

    TrainingStep legacyStep("legacy", {lossRoot}, sgd, {weights, biases});
    auto legacyStepRef = std::make_shared<TrainingStep>(legacyStep);
    ExecutableTrainingPlan legacyPlan =
        ExecutableTrainingPlan::compile(TrainingProgram(std::vector<std::shared_ptr<TrainingStep>>{legacyStepRef}), *placed);
    EXPECT_NO_THROW(legacyPlan.assertLegacyLocalExecutorCompatible());
    EXPECT_NO_THROW(legacyPlan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences()));
    nlohmann::json planJson = legacyPlan.architectureJson();
    EXPECT_EQ(planJson.at("version").get<std::string>(), "1.0.0");
    EXPECT_EQ(planJson.at("step_count").get<uint64_t>(), 1u);
    EXPECT_EQ(planJson.at("total_step_repeats_per_iteration").get<uint64_t>(), 1u);

    EXPECT_THROW(placed->resolveParameterReference(ParameterReference(fc.getId(), "missing")), std::runtime_error);
    EXPECT_THROW(StepExecutable(TrainingStep("bad", {lossRoot}, sgd, {ParameterReference(999999, "weights")}), *placed),
                 std::runtime_error);
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
