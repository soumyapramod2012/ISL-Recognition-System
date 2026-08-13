import numpy as np


class LandmarkNormalizer:

    POSE_LANDMARKS = 33
    FEATURES_PER_POSE = 4
    FEATURES_PER_HAND = 3
    HAND_LANDMARKS = 21

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def normalize(self, landmarks):

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32,
        ).copy()

        if landmarks.ndim != 2:
            raise ValueError(
                "Expected landmarks with shape "
                "(frames, features)"
            )

        if landmarks.shape[1] != 258:
            raise ValueError(
                f"Expected 258 features, "
                f"found {landmarks.shape[1]}"
            )

        for frame_index in range(
            landmarks.shape[0]
        ):

            frame = landmarks[frame_index]

            left_offset = (
                self.LEFT_SHOULDER
                * self.FEATURES_PER_POSE
            )

            right_offset = (
                self.RIGHT_SHOULDER
                * self.FEATURES_PER_POSE
            )

            left_shoulder = frame[
                left_offset:left_offset + 3
            ]

            right_shoulder = frame[
                right_offset:right_offset + 3
            ]

            # Skip normalization if either
            # shoulder is missing.

            if (
                np.allclose(left_shoulder, 0.0)
                or np.allclose(right_shoulder, 0.0)
            ):
                continue

            center = (
                left_shoulder + right_shoulder
            ) / 2.0

            scale = np.linalg.norm(
                right_shoulder - left_shoulder
            )

            # Prevent division by zero.

            if scale < 1e-6:
                continue

            # ---------------- Pose ----------------

            for landmark_index in range(
                self.POSE_LANDMARKS
            ):

                offset = (
                    landmark_index
                    * self.FEATURES_PER_POSE
                )

                coordinates = frame[
                    offset:offset + 3
                ]

                # Don't transform completely
                # missing landmarks.

                if np.allclose(
                    coordinates,
                    0.0,
                ):
                    continue

                frame[
                    offset:offset + 3
                ] = (
                    coordinates - center
                ) / scale

            # ---------------- Left Hand ----------------

            left_hand_start = (
                self.POSE_LANDMARKS
                * self.FEATURES_PER_POSE
            )

            left_hand_end = (
                left_hand_start
                + self.HAND_LANDMARKS
                * self.FEATURES_PER_HAND
            )

            left_hand = frame[
                left_hand_start:left_hand_end
            ]

            for index in range(
                self.HAND_LANDMARKS
            ):

                offset = (
                    index
                    * self.FEATURES_PER_HAND
                )

                coordinates = left_hand[
                    offset:offset + 3
                ]

                if np.allclose(
                    coordinates,
                    0.0,
                ):
                    continue

                left_hand[
                    offset:offset + 3
                ] = (
                    coordinates - center
                ) / scale

            # ---------------- Right Hand ----------------

            right_hand_start = left_hand_end

            right_hand_end = (
                right_hand_start
                + self.HAND_LANDMARKS
                * self.FEATURES_PER_HAND
            )

            right_hand = frame[
                right_hand_start:right_hand_end
            ]

            for index in range(
                self.HAND_LANDMARKS
            ):

                offset = (
                    index
                    * self.FEATURES_PER_HAND
                )

                coordinates = right_hand[
                    offset:offset + 3
                ]

                if np.allclose(
                    coordinates,
                    0.0,
                ):
                    continue

                right_hand[
                    offset:offset + 3
                ] = (
                    coordinates - center
                ) / scale

        return landmarks