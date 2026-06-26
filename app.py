import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import av
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker(VideoProcessorBase):
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path='hand_landmarker.task'
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.canvas = None
        self.prev_points = {}
        self.clear_counter = 0
        self.FINGER_TIPS = [4, 8, 12, 16, 20]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        fh, fw, _ = img.shape

        if self.canvas is None:
            self.canvas = np.zeros((fh, fw, 3), dtype=np.uint8)

        black = np.zeros((fh, fw, 3), dtype=np.uint8)
        black = cv2.add(black, self.canvas)

        cv2.putText(black, "1 finger=Draw | 2 fingers=Pause | 5 fingers=Clear",
                    (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (150, 150, 150), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(black, format="bgr24")

st.set_page_config(page_title="Air Writing", page_icon="✏️")
st.title("Air Writing - Hand Gesture Drawing")
st.markdown("""
- ☝️ **1 finger** = Draw
- ✌️ **2 fingers** = Pause
- 🖐️ **5 fingers** = Clear
""")

webrtc_streamer(
    key="hand-tracking",
    video_processor_factory=HandTracker,
    media_stream_constraints={"video": True, "audio": False},
)
