"""
input_controller.py
-------------------
Translates gesture strings into Windows key-press events via pydirectinput.

Why pydirectinput instead of pyautogui?
----------------------------------------
pyautogui uses SendInput with KEYEVENTF_UNICODE, which many games and
Android emulators (BlueStacks, LDPlayer, etc.) ignore because they read
hardware scan-codes rather than virtual keys.  pydirectinput sends
DirectInput-compatible scan-code events, which those applications DO
register.

Key mapping
-----------
  "gas"     → RIGHT arrow key held down (accelerate in Hill Climb Racing)
  "brake"   → LEFT  arrow key held down (brake / reverse)
  "neutral" → both keys released

Only keyDown / keyUp transitions are sent – not every frame – to avoid
flooding the game's input queue.
"""

from __future__ import annotations
from typing import Literal
import pydirectinput
import pygetwindow as gw

# Type alias (mirrors gesture_classifier.Gesture)
Gesture = Literal["gas", "brake", "neutral"]

# Mapping from gesture name to the key that should be held.
GESTURE_KEY_MAP: dict[str, str] = {
    "gas":   "right",   # Right arrow = accelerator
    "brake": "left",    # Left arrow  = brake / reverse
}


class InputController:
    """
    Manages key-down / key-up state so that:
      1. A key is pressed at most once per gesture transition.
      2. The previous key is released before a new one is pressed.
      3. Both keys are released when gesture is "neutral".
    """

    def __init__(self):
        # Track which key (if any) is currently held down.
        self._held_key: str | None = None

        # pydirectinput adds a 0.1 s pause after every call by default.
        # That pause would cap us at ~10 FPS, so we disable it.
        pydirectinput.PAUSE = 0.0

        print("[DEBUG] InputController initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, gesture: Gesture) -> None:
        """
        Update key state based on the current smoothed gesture.

        Call once per frame (or per gesture change); the method is
        idempotent – calling it with the same gesture multiple times
        in a row does nothing after the first call.

        Parameters
        ----------
        gesture : Gesture
            The smoothed gesture from GestureSmoother.active.
        """
        # --- Unconditional call trace: proves apply() is reached at all ---
        print(f"[DEBUG] apply() called | gesture={gesture!r} | held_key={self._held_key!r}")

        desired_key = GESTURE_KEY_MAP.get(gesture)  # None if "neutral"

        if desired_key == self._held_key:
            return  # No state change – do nothing

        # --- Release the currently held key (if any) ---
        if self._held_key is not None:
            pydirectinput.keyUp(self._held_key)
            print(f"[DEBUG] keyUp('{self._held_key}') fired")
            self._held_key = None

        # --- Press the new key (if the gesture requires one) ---
        if desired_key is not None:
            # Check exactly which window is focused right before we send the key.
            try:
                active = gw.getActiveWindow()
                active_title = active.title if active else "NONE"
            except Exception as e:
                active_title = f"ERROR getting active window: {e}"
            print(f"[DEBUG] Active window right before keyDown: {active_title}")

            pydirectinput.keyDown(desired_key)
            self._held_key = desired_key
            print(f"[DEBUG] keyDown('{desired_key}') fired")

    def release_all(self) -> None:
        """
        Safety method: release any held key unconditionally.
        Call this when the application is shutting down so the game
        doesn't get stuck with a key permanently held down.
        """
        if self._held_key is not None:
            pydirectinput.keyUp(self._held_key)
            print(f"[DEBUG] release_all(): keyUp('{self._held_key}') fired")
            self._held_key = None

    @property
    def held_key(self) -> str | None:
        """The key currently being held, or None."""
        return self._held_key