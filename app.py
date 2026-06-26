import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import av

class HandTracker(VideoProcessorBase):
    def __init__(self):
        self.canvas = None
        self.prev_x = 0
        self.prev_y = 0
        self.clear_counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        fh, fw, _ = img.shape

        if self.canvas is None:
            self.canvas = np.zeros((fh, fw, 3), dtype=np.uint8)

        black = np.zeros((fh, fw, 3), dtype=np.uint8)

        # Detect skin color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 20, 70])
        upper1 = np.array([20, 255, 255])
        lower2 = np.array([170, 20, 70])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.dilate(mask, None, iterations=3)
        mask = cv2.erode(mask, None, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > 3000:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    hull = cv2.convexHull(c, returnPoints=False)
                    if len(hull) > 3:
                        defects = cv2.convexityDefects(c, hull)
                        fingers = 0
                        if defects is not None:
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                if d / 256 > 20:
                                    fingers += 1

                        # Draw fingertip dot
                        cv2.circle(black, (cx, cy), 15,
                                   (0, 255, 0), cv2.FILLED)

                        # 0-1 defects = 1-2 fingers = DRAW
                        if fingers <= 1:
                            if self.prev_x == 0 and self.prev_y == 0:
                                self.prev_x, self.prev_y = cx, cy
                            cv2.line(self.canvas,
                                     (self.prev_x, self.prev_y),
                                     (cx, cy), (0, 255, 0), 5)
                            self.prev_x, self.prev_y = cx, cy
                            cv2.putText(black, "Drawing...",
                                        (10, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 2)

                        # 2-3 defects = pause
                        elif fingers <= 3:
                            self.prev_x, self.prev_y = 0, 0
                            cv2.putText(black, "Paused",
                                        (10, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 200, 255), 2)

                        # 4+ defects = clear
                        else:
                            self.clear_counter += 1
                            cv2.putText(black, "Hold to Clear...",
                                        (10, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2)
                            if self.clear_counter > 10:
                                self.canvas = np.zeros(
                                    (fh, fw, 3), dtype=np.uint8)
                                self.prev_x, self.prev_y = 0, 0
                                self.clear_counter = 0
        else:
            self.prev_x, self.prev_y = 0, 0
            cv2.putText(black, "Show your hand...",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (150, 150, 150), 2)

        black = cv2.add(black, self.canvas)
        cv2.putText(black,
                    "1-2 fingers=Draw | 3 fingers=Pause | 5 fingers=Clear",
                    (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (150, 150, 150), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(black, format="bgr24")


st.set_page_config(page_title="Air Writing", page_icon="✏️")
st.title("Air Writing - Hand Gesture Drawing")
st.markdown("""
**How to use:**
- ☝️ **1-2 fingers** = Draw
- 🤟 **3 fingers** = Pause
- 🖐️ **5 fingers** = Clear screen
""")

webrtc_streamer(
    key="hand-tracking",
    video_processor_factory=HandTracker,
    media_stream_constraints={"video": True, "audio": False},
)
