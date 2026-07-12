from __future__ import annotations

import thor


EXPECTED_DATA_API = {
    "AffineWindowedTensorChunk",
    "BatchPolicy",
    "BatchSession",
    "ConstantPad",
    "DatasetAccessPolicy",
    "DatasetField",
    "DatasetFieldKind",
    "DatasetId",
    "DatasetLayout",
    "DatasetSchema",
    "DatasetSplitManifest",
    "DatasetWriter",
    "DeviceDatasetStorage",
    "ExampleIndexRange",
    "ExampleIndexSet",
    "FileDataset",
    "NamedDataset",
    "NumpyDataset",
    "StratificationMode",
    "StratifiedFold",
    "StratifiedHoldoutKFoldManifest",
    "StratifiedKFoldManifest",
    "StratifiedSplit",
    "StratifiedSplitter",
    "StratifiedTrainValidationTestSplit",
    "TensorLayout",
    "TrainingData",
    "WindowedTensorChunk",
    "WindowedTensorLayout",
    "WindowedTensorSourceLayout",
}

EXPECTED_TRAINING_API = {
    "DatasetInputBindings",
    "DeviceDatasetStorageReport",
    "EarlyCompletionPolicy",
    "EarlyCompletionRule",
    "GradientClearPolicy",
    "RestartCondition",
    "RestartPolicy",
    "StepExecutable",
    "Trainer",
    "TrainerFitOptions",
    "TrainingEarlyCompletionPolicy",
    "TrainingEnsembleMemberResult",
    "TrainingEnsembleResult",
    "TrainingEventPhase",
    "TrainingInputBinding",
    "TrainingNamedMetricResult",
    "TrainingPhase",
    "TrainingProgram",
    "TrainingRestartCondition",
    "TrainingRestartPolicy",
    "TrainingRunCompletionReason",
    "TrainingRunInputSignature",
    "TrainingRunOutputSignature",
    "TrainingRunResult",
    "TrainingRuns",
    "TrainingRunsEarlyCompletionPolicy",
    "TrainingRunsEarlyCompletionRule",
    "TrainingRunsFailurePolicy",
    "TrainingRunsRestartCondition",
    "TrainingRunsRestartPolicy",
    "TrainingRunsResult",
    "TrainingRunStatus",
    "TrainingStatsSnapshot",
    "TrainingStep",
    "make_k_fold_run_specs",
    "training_runs_from_k_fold_split",
}


def _module_public_attributes(module: object) -> set[str]:
    return {name for name in vars(module) if not name.startswith("_")}


def test_data_namespace_matches_exact_allowlist():
    assert set(thor.data.__all__) == EXPECTED_DATA_API
    assert _module_public_attributes(thor.data) == EXPECTED_DATA_API
    assert set(dir(thor.data)) == EXPECTED_DATA_API


def test_training_namespace_matches_exact_allowlist():
    assert set(thor.training.__all__) == EXPECTED_TRAINING_API
    assert _module_public_attributes(thor.training) == EXPECTED_TRAINING_API
    assert set(dir(thor.training)) == EXPECTED_TRAINING_API


def test_data_implementation_types_do_not_leak_through_training_namespace():
    assert EXPECTED_DATA_API.isdisjoint(EXPECTED_TRAINING_API)
    for name in EXPECTED_DATA_API:
        assert not hasattr(thor.training, name)
