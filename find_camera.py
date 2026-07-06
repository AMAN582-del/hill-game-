"""
find_camera.py
--------------
Diagnostic script: probe camera indices 0-3 with both the DirectShow
(CAP_DSHOW) and default backends, then show a live preview of any camera
that successfully reads a frame.

Run with:
    python find_camera.py

The script will print a table like:
    Index  Backend    isOpened  Frame OK
    0      CAP_DSHOW  Yes       Yes   ← use this one
    1      CAP_DSHOW  No        -
    2      default    Yes       No    ← opens but gives empty frames
    ...

Use the winning index and set CAMERA_INDEX in main.py accordingly.
"""

import cv2

MAX_INDEX = 4   # probe 0, 1, 2, 3

# OpenCV on Windows may not support index-based capture with every backend.
# We test the default backend (lets OpenCV pick) and CAP_MSMF explicitly.
# CAP_DSHOW is NOT used here because some OpenCV builds report
# "can't be used to capture by index" when tried directly.
BACKENDS = [
    ("default", None),              # OpenCV auto-selects backend
    ("MSMF",    cv2.CAP_MSMF),     # Microsoft Media Foundation
]


def probe_cameras() -> list[dict]:
    """Try every (index, backend) combination and return a summary list."""
    results = []
    for idx in range(MAX_INDEX):
        for backend_name, backend_flag in BACKENDS:
            cap = (
                cv2.VideoCapture(idx, backend_flag)
                if backend_flag is not None
                else cv2.VideoCapture(idx)
            )
            opened = cap.isOpened()
            frame_ok = False
            if opened:
                ret, frame = cap.read()
                frame_ok = ret and frame is not None and frame.size > 0
            cap.release()
            results.append(
                {
                    "index":   idx,
                    "backend": backend_name,
                    "opened":  opened,
                    "frame":   frame_ok,
                }
            )
    return results


def print_table(results: list[dict]) -> None:
    header = f"{'Index':>5}  {'Backend':<10}  {'isOpened':<10}  {'Frame OK':<10}  {'Status'}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        opened_s = "Yes" if r["opened"] else "No"
        frame_s  = "Yes" if r["frame"]  else ("No" if r["opened"] else "-")
        if r["frame"]:
            status = "[OK] WORKING -- use this index"
        elif r["opened"]:
            status = "Opens but no frame (MSMF issue)"
        else:
            status = "Not found"
        print(f"{r['index']:>5}  {r['backend']:<10}  {opened_s:<10}  {frame_s:<10}  {status}")
    print()


def live_preview(results: list[dict]) -> None:
    """Show a live preview of the first fully working camera found."""
    working = [r for r in results if r["frame"]]
    if not working:
        print("No working cameras found. Check that your webcam is connected and not in use by another app.")
        return

    best = working[0]
    idx  = best["index"]
    flag = cv2.CAP_DSHOW if best["backend"] == "CAP_DSHOW" else None

    print(f"Opening live preview: index={idx}, backend={best['backend']}")
    print("Press 'q' to quit the preview.\n")

    cap = cv2.VideoCapture(idx, flag) if flag else cv2.VideoCapture(idx)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Preview: frame read failed.")
            break
        cv2.putText(
            frame,
            f"Index={idx}  Backend={best['backend']}  (press q to quit)",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA,
        )
        cv2.imshow("Camera preview – find_camera.py", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Probing cameras 0-3 …")
    results = probe_cameras()
    print_table(results)

    working = [r for r in results if r["frame"]]
    if working:
        best = working[0]
        print(f"Recommended: set CAMERA_INDEX = {best['index']} in main.py")
        print(f"             backend '{best['backend']}' produced a valid frame.")
        print()
        live_preview(results)
    else:
        print("No working cameras found.")
        print("Tips:")
        print("  • Make sure no other app (Teams, Zoom, OBS …) is using the webcam.")
        print("  • Try plugging in an external USB webcam.")
        print("  • On some laptops, the built-in camera is index 1 or 2.")
