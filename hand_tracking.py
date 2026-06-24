import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas = None
prev_points = {}
FINGER_TIPS = [4, 8, 12, 16, 20]
clear_counter = 0

def fingers_up(hand_landmarks, hand_label):
    lm = hand_landmarks.landmark
    fingers = []
    if hand_label == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)
    for tip in FINGER_TIPS[1:]:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    return fingers

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    fh, fw, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((fh, fw, 3), dtype=np.uint8)

    black = np.zeros((fh, fw, 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        total_hands = len(results.multi_hand_landmarks)

        for idx in range(min(total_hands, 2)):
            hand_lms = results.multi_hand_landmarks[idx]
            hand_label = results.multi_handedness[idx].classification[0].label

            if idx not in prev_points:
                prev_points[idx] = (0, 0)

            mp_draw.draw_landmarks(black, hand_lms, mp_hands.HAND_CONNECTIONS)

            f = fingers_up(hand_lms, hand_label)
            total = sum(f)

            index_tip = hand_lms.landmark[8]
            cx = int(index_tip.x * fw)
            cy = int(index_tip.y * fh)

            px, py = prev_points[idx]

            color = (0, 255, 0) if hand_label == "Right" else (255, 100, 0)

            # 1 finger = DRAW
            if f[1] == 1 and f[2] == 0 and f[3] == 0 and f[4] == 0:
                if px == 0 and py == 0:
                    prev_points[idx] = (cx, cy)
                else:
                    cv2.line(canvas, (px, py), (cx, cy), color, 5)
                    prev_points[idx] = (cx, cy)
                cv2.circle(black, (cx, cy), 10, color, cv2.FILLED)
                cv2.putText(black, f"{hand_label} Drawing...",
                            (10, 80 + idx * 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 2 fingers = PAUSE
            elif f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0:
                prev_points[idx] = (0, 0)
                cv2.putText(black, f"{hand_label} Paused",
                            (10, 80 + idx * 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            # 5 fingers = CLEAR
            elif total >= 4:
                clear_counter += 1
                cv2.putText(black, f"Hold to Clear... {clear_counter}",
                            (10, 80 + idx * 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if clear_counter > 10:
                    canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
                    prev_points = {}
                    clear_counter = 0
            else:
                clear_counter = 0
                prev_points[idx] = (0, 0)

    black = cv2.add(black, canvas)

    cv2.putText(black, "1 finger=Draw | 2 fingers=Pause | 5 fingers=Clear | Q=Quit",
                (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (150, 150, 150), 1, cv2.LINE_AA)

    cv2.putText(black, "Right Hand=Green | Left Hand=Blue",
                (10, fh - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imshow("Air Writing - 2 Hands", black)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
