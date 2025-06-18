import cv2
import mediapipe as mp
import math
import numpy as np
import screen_brightness_control as sbc

# Initialize hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Start video capture
vidObj = cv2.VideoCapture(0)
vidObj.set(3, 1280)  # Set width
vidObj.set(4, 720)   # Set height

# Brightness limits
minBrightness, maxBrightness = 10, 100  # Adjust limits as needed

while True:
    _, frame = vidObj.read()
    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            if len(lmList) > 8:
                x1, y1 = lmList[4][1], lmList[4][2]   # Thumb Tip
                x2, y2 = lmList[8][1], lmList[8][2]   # Index Finger Tip
                distance = math.hypot(x2 - x1, y2 - y1)

                # Map distance to brightness range
                brightness = np.interp(distance, [30, 200], [minBrightness, maxBrightness])
                sbc.set_brightness(int(brightness))

                # Draw line between fingers
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)

                # Display brightness level
                cv2.putText(frame, f'Brightness: {int(brightness)}%', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    # Show frame
    cv2.imshow("Brightness Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vidObj.release()
cv2.destroyAllWindows()
