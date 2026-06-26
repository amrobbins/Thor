import pytest
import thor


def test_training_runs_bad_trainer_type_has_clear_error():
    with pytest.raises(TypeError) as exc_info:
        thor.training.TrainingRuns([("fold_0", object(), "demand_cv5")])

    message = str(exc_info.value)
    assert "TrainingRuns runs entry[1]" in message
    assert "thor.training.Trainer" in message
    assert "got object" in message
    assert "cast" not in message.lower()


def test_training_program_bad_steps_type_has_clear_error():
    with pytest.raises(TypeError) as exc_info:
        thor.training.TrainingProgram(steps=object())

    message = str(exc_info.value)
    assert "TrainingProgram.__new__() argument 'steps'" in message
    assert "sequence of thor.training.TrainingStep objects or None" in message
    assert "got object" in message
    assert "cast" not in message.lower()
