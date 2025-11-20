import time
import os
from datetime import datetime
import cv2
import numpy as np
import dxcam

SAVE_DIR = "Data_Sources\Forest"
INTERVAL = .25  # seconds between screenshots

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    camera = dxcam.create()
    camera.start(target_fps=60)

    print("DXCAM screenshot script started.")
    print("Press CTRL+C to stop.\n")

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                continue

            # BGR for OpenCV
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(SAVE_DIR, f"{timestamp}.png")

            cv2.imwrite(path, img)
            print(f"Saved: {path}")

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        camera.stop()
        print("\nStopped. Screenshots saved in:", SAVE_DIR)


if __name__ == "__main__":
    main()
