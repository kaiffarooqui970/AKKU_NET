"""Check that OpenCV can open a window (needed for the Stage 5 live demo).

Usage:
    python scripts/check_opencv_gui.py

It checks three things:
  1. only one OpenCV package is installed (two builds overwrite each other's files)
  2. the OpenCV build has a GUI backend (headless builds have none)
  3. a test window actually opens; it closes after 3 s or on any key
"""
import sys
from importlib.metadata import PackageNotFoundError, version

import cv2
import numpy as np

OPENCV_PACKAGES = [
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
]


def installed_opencv_packages() -> list[str]:
    found = []
    for name in OPENCV_PACKAGES:
        try:
            found.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            pass
    return found


def gui_backend() -> str:
    """The 'GUI:' line from OpenCV's build info, e.g. 'COCOA' on macOS, 'NONE' if headless."""
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", 1)[1].strip()
    return "NONE"


def main() -> int:
    packages = installed_opencv_packages()
    print(f"OpenCV packages installed: {packages}")
    if len(packages) != 1:
        print("FAIL: exactly one OpenCV package should be installed. Fix with:\n"
              "  pip uninstall -y " + " ".join(OPENCV_PACKAGES) + "\n"
              "  pip install -r requirements.txt")
        return 1

    backend = gui_backend()
    print(f"cv2 {cv2.__version__}, GUI backend: {backend}")
    if backend == "NONE":
        print("FAIL: this OpenCV build has no GUI support (headless).")
        return 1

    img = np.full((200, 520, 3), 40, dtype=np.uint8)
    cv2.putText(img, "AkkuNet: OpenCV window works", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    try:
        cv2.imshow("AkkuNet OpenCV check", img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # macOS needs one more event-loop tick to actually close the window
    except cv2.error as e:
        print(f"FAIL: could not open a window: {e}")
        return 1

    print("OK: window opened and closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
