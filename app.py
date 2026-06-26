import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import av
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class HandTracker(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.canvas = None
        self.prev_points = {}
        self.clear_counter = 0
        self.FINGER_TIPS = [4, 8, 12, 16, 20]

    def fingers_up(self, hand_landmarks, hand_label):
        lm = hand_landmarks.landmark
        fingers = []
        if hand_label == "Right":
            fingers.append(1 if lm[4].x < lm[3].x else 0)
        else:
            fingers.append(1 if lm[4].x > lm[3].x else 0)
        for tip in self.FINGER_TIPS[1:]:
            fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
        return fingers

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        fh, fw, _ = img.shape

        if self.canvas is None:
            self.canvas = np.zeros((fh, fw, 3), dtype=np.uint8)

        black = np.zeros((fh, fw, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for idx in range(min(len(results.multi_hand_landmarks), 2)):
                hand_lms = results.multi_hand_landmarks[idx]
                hand_label = results.multi_handedness[idx].classification[0].label

                if idx not in self.prev_points:
                    self.prev_points[idx] = (0, 0)

                mp_draw.draw_landmarks(black, hand_lms, mp_hands.HAND_CONNECTIONS)

                f = self.fingers_up(hand_lms, hand_label)
                total = sum(f)
                index_tip = hand_lms.landmark[8]
                cx = int(index_tip.x * fw)
                cy = int(index_tip.y * fh)
                px, py = self.prev_points[idx]
                color = (0, 255, 0) if hand_label == "Right" else (255, 100, 0)

                if f[1] == 1 and f[2] == 0 and f[3] == 0 and f[4] == 0:
                    if px == 0 and py == 0:
                        self.prev_points[idx] = (cx, cy)
                    else:
                        cv2.line(self.canvas, (px, py), (cx, cy), color, 5)
                        self.prev_points[idx] = (cx, cy)
                    cv2.circle(black, (cx, cy), 10, color, cv2.FILLED)
                    cv2.putText(black, f"{hand_label} Drawing...",
                                (10, 80 + idx * 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                elif f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0:
                    self.prev_points[idx] = (0, 0)
                    cv2.putText(black, f"{hand_label} Paused",
                                (10, 80 + idx * 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
                elif total >= 4:
                    self.clear_counter += 1
                    if self.clear_counter > 10:
                        self.canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
                        self.prev_points = {}
                        self.clear_counter = 0
                    cv2.putText(black, "Hold to Clear...",
                                (10, 80 + idx * 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    self.clear_counter = 0
                    self.prev_points[idx] = (0, 0)

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
