import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import av

class HandDrawer(VideoProcessorBase):
    def __init__(self):
        self.canvas = None
        self.prev_x = 0
        self.prev_y = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        fh, fw, _ = img.shape

        if self.canvas is None:
            self.canvas = np.zeros((fh, fw, 3), dtype=np.uint8)

        black = np.zeros((fh, fw, 3), dtype=np.uint8)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 1000:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = cx, cy

                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (cx, cy), (0, 255, 0), 5)
                    self.prev_x, self.prev_y = cx, cy
                    cv2.circle(black, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        else:
            self.prev_x, self.prev_y = 0, 0

        black = cv2.add(black, self.canvas)

        cv2.putText(black, "Show RED object to draw | Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (150, 150, 150), 2)

        return av.VideoFrame.from_ndarray(black, format="bgr24")

st.set_page_config(page_title="Air Drawing", page_icon="✏️")
st.title("Air Drawing App")
st.markdown("""
**How to use:**
- Show a **RED object** (pen cap, red marker) to camera to draw!
- Move it around to draw on screen
""")

webrtc_streamer(
    key="hand-drawing",
    video_processor_factory=HandDrawer,
    media_stream_constraints={"video": True, "audio": False},
)
