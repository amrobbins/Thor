from __future__ import annotations

import pytest

import thor


class _FakeTrainer:
    def __init__(self, name: str):
        self.name = name


def test_make_k_fold_run_specs_builds_training_runs_entries_from_holdout_manifest():
    keys = [f"product_{i}" for i in range(12)]
    splitter = thor.data.StratifiedSplitter(keys, list(range(12)), mode="quantile", num_bins=4, seed=17)
    split = splitter.holdout_plus_k_fold(test_size=2, k=3)
    calls = []

    def make_trainer(*, fold, run_name, ensemble_group, test_keys, test_groups):
        calls.append((fold.fold_index, run_name, ensemble_group, tuple(test_keys), tuple(test_groups)))
        return _FakeTrainer(run_name)

    specs = thor.training.make_k_fold_run_specs(
        split,
        make_trainer=make_trainer,
        ensemble_group="brand_demand_cv3",
        run_name_template="brand_a_fold_{fold_index}",
        ensemble_weight=lambda *, fold, **_: fold.fold_index + 1,
    )

    assert [spec[0] for spec in specs] == ["brand_a_fold_0", "brand_a_fold_1", "brand_a_fold_2"]
    assert [spec[2] for spec in specs] == ["brand_demand_cv3"] * 3
    assert [spec[3] for spec in specs] == [1.0, 2.0, 3.0]
    assert [spec[1].name for spec in specs] == ["brand_a_fold_0", "brand_a_fold_1", "brand_a_fold_2"]
    assert all(call[2] == "brand_demand_cv3" for call in calls)
    assert all(call[3] == split.test_keys for call in calls)
    assert all(call[4] == split.test_groups for call in calls)


def test_make_k_fold_run_specs_supports_simple_positional_fold_factory():
    keys = [f"row_{i}" for i in range(6)]
    manifest = thor.data.StratifiedSplitter(keys, [0, 0, 1, 1, 2, 2], mode="categorical", seed=3).k_fold(3)

    specs = thor.training.make_k_fold_run_specs(
        manifest,
        make_trainer=lambda fold: _FakeTrainer(f"trainer_{fold.fold_index}"),
        ensemble_group="plain_cv3",
    )

    assert [(run_name, trainer.name, group, weight) for run_name, trainer, group, weight in specs] == [
        ("fold_0", "trainer_0", "plain_cv3", 1.0),
        ("fold_1", "trainer_1", "plain_cv3", 1.0),
        ("fold_2", "trainer_2", "plain_cv3", 1.0),
    ]


def test_make_k_fold_run_specs_rejects_duplicate_run_names_and_bad_weights():
    manifest = thor.data.StratifiedSplitter(["a", "b", "c", "d"], [0.0, 1.0, 2.0, 3.0], seed=5).k_fold(2)

    with pytest.raises(ValueError, match="duplicate run name"):
        thor.training.make_k_fold_run_specs(
            manifest,
            make_trainer=lambda fold: _FakeTrainer(str(fold.fold_index)),
            ensemble_group="cv2",
            run_name_template="same_name",
        )

    with pytest.raises(ValueError, match="positive"):
        thor.training.make_k_fold_run_specs(
            manifest,
            make_trainer=lambda fold: _FakeTrainer(str(fold.fold_index)),
            ensemble_group="cv2",
            ensemble_weight=0.0,
        )

    with pytest.raises(ValueError, match="ensemble_group"):
        thor.training.make_k_fold_run_specs(
            manifest,
            make_trainer=lambda fold: _FakeTrainer(str(fold.fold_index)),
            ensemble_group="",
        )
