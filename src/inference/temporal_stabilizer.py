from collections import deque


class TemporalStabilizer:
    """
    Stabilize realtime predictions using a short
    temporal prediction history.

    The stabilizer uses different requirements for:

    1. Establishing the initial prediction.
    2. Changing an already established prediction.

    This prevents startup delay while requiring stronger
    evidence before changing an established class.
    """

    def __init__(
        self,
        history_size=5,
        initial_min_votes=2,
        transition_min_votes=3,
        high_confidence_threshold=0.90,
    ):
        if history_size < 1:
            raise ValueError(
                "history_size must be at least 1."
            )

        if initial_min_votes < 1:
            raise ValueError(
                "initial_min_votes must be at least 1."
            )

        if transition_min_votes < 1:
            raise ValueError(
                "transition_min_votes must be at least 1."
            )

        if initial_min_votes > history_size:
            raise ValueError(
                "initial_min_votes cannot exceed history_size."
            )

        if transition_min_votes > history_size:
            raise ValueError(
                "transition_min_votes cannot exceed history_size."
            )

        if not 0.0 <= high_confidence_threshold <= 1.0:
            raise ValueError(
                "high_confidence_threshold must be between 0 and 1."
            )

        self.history_size = history_size
        self.initial_min_votes = initial_min_votes
        self.transition_min_votes = transition_min_votes
        self.high_confidence_threshold = (
            high_confidence_threshold
        )

        self.history = deque(
            maxlen=history_size
        )

        self.stable_prediction = None

    def reset(self):
        """
        Reset the temporal state.
        """

        self.history.clear()
        self.stable_prediction = None

    def update(
        self,
        prediction,
        confidence,
        hand_detected=True,
    ):
        """
        Update the temporal state.

        Parameters
        ----------
        prediction : str
            Raw model prediction.

        confidence : float
            Raw model confidence.

        hand_detected : bool
            Whether a hand is currently detected.

        Returns
        -------
        tuple
            (stable_prediction, stable_confidence)
        """

        # --------------------------------------------------
        # No hand
        # --------------------------------------------------

        if not hand_detected:

            self.reset()

            return (
                "Uncertain",
                0.0,
            )

        # --------------------------------------------------
        # Store current prediction
        # --------------------------------------------------

        self.history.append(
            {
                "prediction": prediction,
                "confidence": confidence,
            }
        )

        # --------------------------------------------------
        # High-confidence startup
        # --------------------------------------------------

        if self.stable_prediction is None:

            if (
                prediction != "Uncertain"
                and confidence
                >= self.high_confidence_threshold
            ):
                self.stable_prediction = prediction

            else:

                # Count predictions for each recognized class.
                class_counts = {}

                for item in self.history:

                    label = item["prediction"]

                    if label == "Uncertain":
                        continue

                    class_counts[label] = (
                        class_counts.get(label, 0) + 1
                    )

                if not class_counts:

                    return (
                        "Uncertain",
                        0.0,
                    )

                candidate = max(
                    class_counts,
                    key=class_counts.get,
                )

                candidate_votes = (
                    class_counts[candidate]
                )

                if (
                    candidate_votes
                    >= self.initial_min_votes
                ):
                    self.stable_prediction = (
                        candidate
                    )

                else:

                    return (
                        "Uncertain",
                        0.0,
                    )

        # --------------------------------------------------
        # Existing stable prediction
        # --------------------------------------------------

        else:

            class_counts = {}

            for item in self.history:

                label = item["prediction"]

                if label == "Uncertain":
                    continue

                class_counts[label] = (
                    class_counts.get(label, 0) + 1
                )

            if class_counts:

                candidate = max(
                    class_counts,
                    key=class_counts.get,
                )

                candidate_votes = (
                    class_counts[candidate]
                )

                # Require stronger evidence before
                # changing an established prediction.

                if (
                    candidate
                    != self.stable_prediction
                    and candidate_votes
                    >= self.transition_min_votes
                ):

                    self.stable_prediction = (
                        candidate
                    )

        # --------------------------------------------------
        # Calculate confidence of stable prediction
        # --------------------------------------------------

        matching_confidences = [
            item["confidence"]
            for item in self.history
            if (
                item["prediction"]
                == self.stable_prediction
            )
        ]

        if matching_confidences:

            stable_confidence = (
                sum(matching_confidences)
                / len(matching_confidences)
            )

        else:

            stable_confidence = 0.0

        return (
            self.stable_prediction,
            stable_confidence,
        )