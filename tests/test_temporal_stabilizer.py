from src.inference.temporal_stabilizer import (
    TemporalStabilizer,
)


def test_initial_prediction_requires_two_matching_predictions():

    stabilizer = TemporalStabilizer(
        history_size=5,
        initial_min_votes=2,
        transition_min_votes=3,
    )

    prediction, confidence = (
        stabilizer.update(
            "61. Summer",
            0.60,
        )
    )

    assert prediction == "Uncertain"
    assert confidence == 0.0

    prediction, confidence = (
        stabilizer.update(
            "61. Summer",
            0.62,
        )
    )

    assert prediction == "61. Summer"
    assert confidence > 0.0


def test_high_confidence_prediction_can_initialize_immediately():

    stabilizer = TemporalStabilizer(
        history_size=5,
        initial_min_votes=2,
        transition_min_votes=3,
        high_confidence_threshold=0.90,
    )

    prediction, confidence = (
        stabilizer.update(
            "63. Winter",
            0.99,
        )
    )

    assert prediction == "63. Winter"
    assert confidence == 0.99


def test_uncertain_prediction_does_not_cause_flicker():

    stabilizer = TemporalStabilizer(
        history_size=5,
        initial_min_votes=2,
        transition_min_votes=3,
    )

    stabilizer.update(
        "61. Summer",
        0.60,
    )

    stabilizer.update(
        "61. Summer",
        0.62,
    )

    prediction, confidence = (
        stabilizer.update(
            "Uncertain",
            0.0,
        )
    )

    assert prediction == "61. Summer"
    assert confidence > 0.0


def test_two_new_predictions_do_not_change_stable_class():

    stabilizer = TemporalStabilizer(
        history_size=5,
        initial_min_votes=2,
        transition_min_votes=3,
    )

    stabilizer.update(
        "61. Summer",
        0.70,
    )

    stabilizer.update(
        "61. Summer",
        0.72,
    )

    stabilizer.update(
        "61. Summer",
        0.74,
    )

    stabilizer.update(
        "62. Spring",
        0.60,
    )

    prediction, confidence = (
        stabilizer.update(
            "62. Spring",
            0.62,
        )
    )

    assert prediction == "61. Summer"
    assert confidence > 0.0


def test_three_persistent_new_predictions_change_class():

    stabilizer = TemporalStabilizer(
        history_size=5,
        initial_min_votes=2,
        transition_min_votes=3,
    )

    stabilizer.update(
        "61. Summer",
        0.70,
    )

    stabilizer.update(
        "61. Summer",
        0.72,
    )

    stabilizer.update(
        "61. Summer",
        0.74,
    )

    stabilizer.update(
        "62. Spring",
        0.60,
    )

    stabilizer.update(
        "62. Spring",
        0.62,
    )

    prediction, confidence = (
        stabilizer.update(
            "62. Spring",
            0.64,
        )
    )

    assert prediction == "62. Spring"
    assert confidence > 0.0


def test_no_hand_resets_prediction():

    stabilizer = TemporalStabilizer(
        history_size=5,
        initial_min_votes=2,
        transition_min_votes=3,
    )

    stabilizer.update(
        "64. Fall",
        0.70,
    )

    stabilizer.update(
        "64. Fall",
        0.72,
    )

    prediction, confidence = (
        stabilizer.update(
            "Uncertain",
            0.0,
            hand_detected=False,
        )
    )

    assert prediction == "Uncertain"
    assert confidence == 0.0

    prediction, confidence = (
        stabilizer.update(
            "64. Fall",
            0.70,
        )
    )

    assert prediction == "Uncertain"

    prediction, confidence = (
        stabilizer.update(
            "64. Fall",
            0.72,
        )
    )

    assert prediction == "64. Fall"
    assert confidence > 0.0