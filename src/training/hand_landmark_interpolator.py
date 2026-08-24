import numpy as np


class HandLandmarkInterpolator:

    POSE_FEATURES = 33 * 4
    HAND_LANDMARKS = 21
    FEATURES_PER_HAND = 3

    LEFT_HAND_START = POSE_FEATURES

    RIGHT_HAND_START = (
        POSE_FEATURES
        + HAND_LANDMARKS * FEATURES_PER_HAND
    )

    HAND_FEATURES = (
        HAND_LANDMARKS
        * FEATURES_PER_HAND
    )

    def __init__(self, max_gap=5):

        if max_gap < 1:
            raise ValueError(
                "max_gap must be >= 1"
            )

        self.max_gap = max_gap

    def _is_missing(self, hand):

        return np.all(
            hand == 0.0,
            axis=1,
        )

    def _interpolate_hand(
        self,
        hand,
    ):

        hand = hand.copy()

        missing = self._is_missing(hand)

        frame_count = len(hand)

        index = 0

        while index < frame_count:

            if not missing[index]:

                index += 1
                continue

            start = index

            while (
                index < frame_count
                and missing[index]
            ):
                index += 1

            end = index - 1

            gap_length = (
                end - start + 1
            )

            # Only interpolate short gaps.

            if gap_length > self.max_gap:
                continue

            # Need a valid frame on both sides.

            previous_index = start - 1
            next_index = end + 1

            if previous_index < 0:
                continue

            if next_index >= frame_count:
                continue

            if missing[previous_index]:
                continue

            if missing[next_index]:
                continue

            previous = hand[
                previous_index
            ]

            next_frame = hand[
                next_index
            ]

            for frame_index in range(
                start,
                end + 1,
            ):

                alpha = (
                    frame_index
                    - previous_index
                ) / (
                    next_index
                    - previous_index
                )

                hand[frame_index] = (
                    previous
                    + alpha
                    * (
                        next_frame
                        - previous
                    )
                )

        return hand

    def interpolate(self, landmarks):

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32,
        ).copy()

        if landmarks.ndim != 2:

            raise ValueError(
                "Expected landmarks with shape "
                "(frames, features)"
            )

        expected_features = (
            self.POSE_FEATURES
            + 2 * self.HAND_FEATURES
        )

        if landmarks.shape[1] != expected_features:

            raise ValueError(
                f"Expected {expected_features} "
                f"features, found "
                f"{landmarks.shape[1]}"
            )

        # ---------------- Left hand ----------------

        left_end = (
            self.LEFT_HAND_START
            + self.HAND_FEATURES
        )

        left_hand = landmarks[
            :,
            self.LEFT_HAND_START:left_end,
        ].reshape(
            -1,
            self.HAND_LANDMARKS,
            self.FEATURES_PER_HAND,
        )

        # One vector per frame.

        left_hand = left_hand.reshape(
            landmarks.shape[0],
            self.HAND_FEATURES,
        )

        left_hand = self._interpolate_hand(
            left_hand
        )

        landmarks[
            :,
            self.LEFT_HAND_START:left_end,
        ] = left_hand

        # ---------------- Right hand ----------------

        right_end = (
            self.RIGHT_HAND_START
            + self.HAND_FEATURES
        )

        right_hand = landmarks[
            :,
            self.RIGHT_HAND_START:right_end,
        ]

        right_hand = self._interpolate_hand(
            right_hand
        )

        landmarks[
            :,
            self.RIGHT_HAND_START:right_end,
        ] = right_hand

        return landmarks