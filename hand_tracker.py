"""
hand_tracker.py
---------------
Handles webcam capture and MediaPipe hand landmark detection.

Uses the NEW MediaPipe Tasks API (mediapipe >= 0.10):
    mediapipe.tasks.python.vision.HandLandmarker

The old mp.solutions.hands API was removed in mediapipe 0.10.x+.

Landmark format (Tasks API)
----------------------------
get_frame_and_hands() returns:
    hand_list : list[list[NormalizedLandmark]]

Each inner list has exactly 21 NormalizedLandmark objects, indexed 0–20.
Access coordinates as:   landmark.x   landmark.y   landmark.z
All values are normalised to [0.0, 1.0] relative to the frame dimensions.

This is the same logical layout as the old API but WITHOUT the wrapper object:
    Old:  hand_landmarks.landmark[i].x
    New:  landmarks[i].x              (landmarks = hand_list[0])
"""

import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Hand skeleton connection pairs (standard MediaPipe 21-point topology) ─────
# Each tuple is (landmark_index_A, landmark_index_B).
# Drawn as lines on the debug overlay.
HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17),
]

# Colours for the drawn skeleton (BGR)
_LANDMARK_COLOUR    = (0, 255, 128)   # green dots
_CONNECTION_COLOUR  = (255, 255, 255) # white lines
_LANDMARK_RADIUS    = 4
_CONNECTION_THICKNESS = 2

# Default path to the downloaded .task model file.
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task",
)


def _draw_landmarks(frame, landmarks) -> None:
    """
    Draw the 21 hand landmarks and their connecting bones onto *frame* in-place.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (modified in-place).
    landmarks : list[NormalizedLandmark]
        21-element list returned by HandLandmarker.
    """
    h, w = frame.shape[:2]

    # Convert normalised [0,1] coords → pixel coords once.
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Draw connections (bones) first so dots appear on top.
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], _CONNECTION_COLOUR, _CONNECTION_THICKNESS, cv2.LINE_AA)

    # Draw landmark dots.
    for px, py in pts:
        cv2.circle(frame, (px, py), _LANDMARK_RADIUS, _LANDMARK_COLOUR, -1, cv2.LINE_AA)


class HandTracker:
    """
    Wraps the MediaPipe Tasks HandLandmarker and an OpenCV VideoCapture.

    Usage
    -----
        tracker = HandTracker()
        while True:
            frame, hands = tracker.get_frame_and_hands()
            if frame is None:
                break
            # hands is a list[list[NormalizedLandmark]]
            # hands[0][8].y  → y-coord of index fingertip of first hand
        tracker.release()
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_path: str = _DEFAULT_MODEL_PATH,
        max_num_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        presence_confidence: float = 0.5,
    ):
        """
        Parameters
        ----------
        camera_index : int
            Which webcam to open (0 = default system webcam).
        model_path : str
            Absolute or relative path to hand_landmarker.task.
        max_num_hands : int
            Maximum simultaneous hands to detect. Keep at 1 for this app.
        detection_confidence : float
            Minimum confidence for the initial detection stage (0–1).
        tracking_confidence : float
            Minimum confidence for tracking between frames (0–1).
        presence_confidence : float
            Minimum confidence that a hand is actually present (0–1).
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download it with:\n"
                "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task "
                "-OutFile hand_landmarker.task"
            )

        # ── Build the HandLandmarker (Tasks API) ─────────────────────────────
        base_options = mp_python.BaseOptions(model_asset_path=model_path)

        # VIDEO mode: the landmarker uses temporal smoothing across frames,
        # which gives steadier landmarks than IMAGE mode on a live feed.
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # ── Camera setup ─────────────────────────────────────────────────────
        # Note: cv2.CAP_DSHOW cannot be used with index-based capture on all
        # OpenCV builds (you may see "can't be used to capture by index").
        # We use the default backend and let OpenCV resolve it (typically MSMF
        # on Windows). If your camera still fails, run:
        #   python find_camera.py
        # to identify the correct index.
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {camera_index}.\n"
                "Run  python find_camera.py  to list working camera indices,\n"
                "then set CAMERA_INDEX in main.py to the correct value."
            )

        # MSMF (Windows Media Foundation) cameras often need a few reads to
        # warm up — the first frame(s) come back empty even though isOpened()
        # returns True.  We discard up to 5 warm-up frames before declaring
        # the camera ready.  Without this, main.py exits immediately with
        # "Camera read failed" on some systems.
        _warmed_up = False
        for _ in range(5):
            ok, _ = self.cap.read()
            if ok:
                _warmed_up = True
                break
        if not _warmed_up:
            self.cap.release()
            raise RuntimeError(
                f"Webcam at index {camera_index} opened but returned no frames.\n"
                "Run  python find_camera.py  to verify which index is working."
            )

        # Timestamp for VIDEO mode (must be monotonically increasing in ms).
        self._start_time_ms = int(time.time() * 1000)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_frame_and_hands(self):
        """
        Capture one frame and run hand landmark detection on it.

        Returns
        -------
        frame : np.ndarray | None
            BGR frame with hand skeleton drawn on it.
            Returns None if the camera read failed.
        hand_list : list[list[NormalizedLandmark]]
            One entry per detected hand (up to max_num_hands).
            Each entry is a list of 21 NormalizedLandmark objects.
            Empty list when no hands are detected.

        Landmark access example
        -----------------------
            frame, hands = tracker.get_frame_and_hands()
            if hands:
                index_tip_y = hands[0][8].y   # index finger tip, y coord
        """
        success, frame = self.cap.read()
        if not success or frame is None:
            return None, []

        # Mirror the image so it feels like looking in a mirror.
        frame = cv2.flip(frame, 1)

        # Build a monotonically increasing timestamp (ms) for VIDEO mode.
        timestamp_ms = int(time.time() * 1000) - self._start_time_ms

        # Convert BGR → RGB for MediaPipe.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in a MediaPipe Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run detection for this video frame.
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        # result.hand_landmarks : list[list[NormalizedLandmark]]
        hand_list = result.hand_landmarks  # may be empty

        # Draw skeleton on the debug frame.
        for landmarks in hand_list:
            _draw_landmarks(frame, landmarks)

        return frame, hand_list

    def release(self) -> None:
        """Release the webcam and MediaPipe resources."""
        self.cap.release()
        self._landmarker.close()


# ── Standalone smoke-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Standalone HandTracker test — press 'q' to quit.\n")
    tracker = HandTracker()
    print("HandTracker initialised OK.\n")

    while True:
        frame, hands = tracker.get_frame_and_hands()
        if frame is None:
            print("Camera read failed.")
            break

        # Print landmark counts to the terminal.
        if hands:
            counts = [len(h) for h in hands]
            print(f"Hands detected: {len(hands)}  |  Landmarks per hand: {counts}", end="\r")
        else:
            print("No hands detected.                          ", end="\r")

        cv2.imshow("HandTracker – standalone test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nQuitting.")
            break

    tracker.release()
    cv2.destroyAllWindows()
