"""
main.py
-------
Entry point.  Ties together HandTracker, GestureSmoother, and InputController
in the main capture-and-control loop.

Run with:
    python main.py

Press 'q' in the debug window (or Ctrl-C in the terminal) to quit cleanly.
"""

import time
import cv2

from hand_tracker import HandTracker
from gesture_classifier import classify, GestureSmoother
from input_controller import InputController

# ── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX       = 0      # 0 = default webcam; change if you have multiple
SMOOTHER_THRESHOLD = 4      # frames a gesture must persist before activating
WINDOW_NAME        = "Hill Climb Racing – Gesture Controller"

# Colour constants (BGR) for the on-screen text
COLOUR_GAS     = (0,   220,  80)   # green
COLOUR_BRAKE   = (0,    80, 220)   # red-orange
COLOUR_NEUTRAL = (200, 200, 200)   # grey
COLOUR_FPS     = (255, 255,   0)   # yellow
# ─────────────────────────────────────────────────────────────────────────────


def gesture_colour(gesture: str) -> tuple[int, int, int]:
    """Return the BGR colour for a given gesture string."""
    return {
        "gas":     COLOUR_GAS,
        "brake":   COLOUR_BRAKE,
        "neutral": COLOUR_NEUTRAL,
    }.get(gesture, COLOUR_NEUTRAL)


def draw_overlay(frame, gesture: str, fps: float) -> None:
    """
    Draw gesture label and FPS counter onto the frame in-place.

    Parameters
    ----------
    frame   : BGR numpy array (modified in-place)
    gesture : current smoothed gesture string
    fps     : frames per second to display
    """
    h, w = frame.shape[:2]

    # ── Gesture label (large, bottom-left) ───────────────────────────────────
    label = f"Gesture: {gesture.upper()}"
    colour = gesture_colour(gesture)
    cv2.putText(
        frame, label,
        org=(20, h - 30),           # 30 px from bottom
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.2,
        color=colour,
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    # ── FPS counter (small, top-right) ───────────────────────────────────────
    fps_text = f"FPS: {fps:.1f}"
    (text_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(
        frame, fps_text,
        org=(w - text_w - 15, 30),  # 15 px from right edge, 30 from top
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=COLOUR_FPS,
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    # ── Key-hint bar (small, top-left) ───────────────────────────────────────
    hints = [
        "Open palm  = GAS (→)",
        "Closed fist = BRAKE (←)",
        "q = quit",
    ]
    for i, hint in enumerate(hints):
        cv2.putText(
            frame, hint,
            org=(10, 25 + i * 22),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            color=(180, 180, 180),
            thickness=1,
            lineType=cv2.LINE_AA,
        )


def main():
    tracker   = HandTracker(camera_index=CAMERA_INDEX)
    smoother  = GestureSmoother(threshold=SMOOTHER_THRESHOLD)
    controller = InputController()

    print("=" * 55)
    print(" Hill Climb Racing – Gesture Controller")
    print("=" * 55)
    print(f"  Camera index  : {CAMERA_INDEX}")
    print(f"  Smoother      : {SMOOTHER_THRESHOLD} frames")
    print(f"  Gas key       : RIGHT arrow")
    print(f"  Brake key     : LEFT arrow")
    print()
    print("  Focus the game window, then wave your hand at the camera.")
    print("  Press 'q' in the debug window to quit.")
    print("=" * 55)

    # For FPS calculation
    prev_time = time.perf_counter()

    try:
        while True:
            # ── Capture + detect ─────────────────────────────────────────────
            frame, hand_list = tracker.get_frame_and_hands()
            if frame is None:
                print("ERROR: Camera read failed. Exiting.")
                break

            # ── Classify ─────────────────────────────────────────────────────
            # Use the first detected hand; ignore the rest.
            raw_gesture = classify(hand_list[0] if hand_list else None)

            # ── Smooth ───────────────────────────────────────────────────────
            smoothed_gesture = smoother.update(raw_gesture)

            # ── Send keys ────────────────────────────────────────────────────
            controller.apply(smoothed_gesture)

            # ── FPS ──────────────────────────────────────────────────────────
            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            # ── Debug overlay ────────────────────────────────────────────────
            draw_overlay(frame, smoothed_gesture, fps)

            cv2.imshow(WINDOW_NAME, frame)

            # ── Quit on 'q' ──────────────────────────────────────────────────
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit requested.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Always release keys and clean up – even if an exception occurred.
        controller.release_all()
        tracker.release()
        cv2.destroyAllWindows()
        print("Cleaned up. Goodbye!")


if __name__ == "__main__":
    main()
