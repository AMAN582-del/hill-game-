"""
gesture_classifier.py
---------------------
Converts a MediaPipe hand-landmark set into a high-level gesture string.

Supported gestures
------------------
  "gas"     – open palm (all 4 fingers extended)
  "brake"   – closed fist (all 4 fingers curled)
  "neutral" – no hand detected, or ambiguous pose

Finger-curl math (explained)
-----------------------------
MediaPipe numbers the 21 landmarks 0–20.  Each finger has 4 joints:
  MCP (knuckle at palm)  → PIP (middle joint) → DIP → TIP

For our purposes we compare two points per finger:
  • TIP  – the very tip of the finger
  • PIP  – the "lower" middle joint (two steps below the tip)

Both coordinates are normalised to [0, 1] where:
  • (0, 0) = top-left of the video frame
  • (1, 1) = bottom-right of the video frame

Because the Y-axis increases downward:
  • When a finger is EXTENDED, the tip is *above* the PIP joint →
      tip.y  <  pip.y   (smaller Y = higher on screen)
  • When a finger is CURLED, the tip is *below* or level with the PIP →
      tip.y  >=  pip.y

The thumb is trickier because it bends sideways, so we compare
tip.x to the IP joint's x instead (left/right rather than up/down).
We skip the thumb for the gas/brake classification to keep it robust.

Landmark indices used
---------------------
Index  Finger     Part
  4    Thumb      TIP
  3    Thumb      IP  (interphalangeal – the only knuckle on thumb)
  8    Index      TIP
  6    Index      PIP
 12    Middle     TIP
 10    Middle     PIP
 16    Ring       TIP
 14    Ring       PIP
 20    Pinky      TIP
 18    Pinky      PIP
"""

from __future__ import annotations
from typing import Literal

# Type alias for clarity
Gesture = Literal["gas", "brake", "neutral"]

# How many of the 4 non-thumb fingers must agree to call a gesture.
# 4/4 = strict; lower = more permissive. Start with 4.
_FINGERS_REQUIRED = 4


def classify(hand_landmarks) -> Gesture:
    """
    Parameters
    ----------
    hand_landmarks : list[NormalizedLandmark]  (MediaPipe Tasks API)
        A plain list of 21 NormalizedLandmark objects for a single detected hand,
        as returned by HandLandmarker (hand_list[0] from HandTracker).
        Pass None to get "neutral".

        Access coordinates as:  hand_landmarks[i].x / .y / .z
        (The old mp.solutions API wrapped these in a NormalizedLandmarkList with
        a .landmark accessor; the Tasks API drops that wrapper.)

    Returns
    -------
    Gesture
        "gas", "brake", or "neutral".
    """
    if hand_landmarks is None:
        return "neutral"

    # Tasks API: hand_landmarks IS already the list of landmarks.
    # (Old API was: lm = hand_landmarks.landmark)
    lm = hand_landmarks  # lm[i].x / lm[i].y / lm[i].z

    # ------------------------------------------------------------------
    # Check each non-thumb finger for extension vs. curl.
    # ------------------------------------------------------------------
    # (tip_index, pip_index) pairs for Index, Middle, Ring, Pinky
    finger_pairs = [
        (8,  6),   # Index finger:  TIP vs PIP
        (12, 10),  # Middle finger: TIP vs PIP
        (16, 14),  # Ring finger:   TIP vs PIP
        (20, 18),  # Pinky finger:  TIP vs PIP
    ]

    extended_count = 0
    curled_count = 0

    for tip_idx, pip_idx in finger_pairs:
        tip_y = lm[tip_idx].y
        pip_y = lm[pip_idx].y

        # Y increases downward.
        # tip above pip  → extended  (tip.y < pip.y)
        # tip below pip  → curled    (tip.y >= pip.y)
        #
        # We add a small tolerance band (+/- CURL_TOLERANCE) so that
        # fingers hovering roughly horizontal are not mis-classified.
        # Increase CURL_TOLERANCE (e.g. 0.02) if you get too many neutral
        # readings; decrease it for a stricter classifier.
        CURL_TOLERANCE = 0.0  # set to 0 for a hard threshold

        if tip_y < pip_y - CURL_TOLERANCE:
            extended_count += 1
        elif tip_y > pip_y + CURL_TOLERANCE:
            curled_count += 1
        # else: ambiguous – counts as neither

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    if extended_count >= _FINGERS_REQUIRED:
        return "gas"
    if curled_count >= _FINGERS_REQUIRED:
        return "brake"

    # At least one finger is ambiguous – call it neutral to avoid
    # accidental key presses.
    return "neutral"


# ------------------------------------------------------------------
# Gesture smoother
# ------------------------------------------------------------------

class GestureSmoother:
    """
    Requires a gesture to be seen for `threshold` consecutive frames
    before it is promoted to the "active" gesture.

    This prevents single-frame noise from triggering key presses.
    Increase `threshold` if you see flickering; decrease it for
    faster response.
    """

    def __init__(self, threshold: int = 4):
        self.threshold = threshold
        self._candidate: Gesture = "neutral"
        self._count: int = 0
        self.active: Gesture = "neutral"   # the last committed gesture

    def update(self, raw: Gesture) -> Gesture:
        """
        Feed in the raw gesture for this frame and get back the
        smoothed (committed) gesture.
        """
        if raw == self._candidate:
            self._count += 1
        else:
            # Gesture changed – restart the counter for the new candidate.
            self._candidate = raw
            self._count = 1

        if self._count >= self.threshold:
            self.active = self._candidate

        return self.active
