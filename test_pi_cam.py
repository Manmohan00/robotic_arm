# source gesture_env/bin/activate

from picamera2 import Picamera2
import mediapipe as mp
import cv2

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)})
picam2.configure(config)
picam2.start()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)

try:
    while True:
        frame = picam2.capture_array()  # RGB888, shape (480, 640, 3)
        print(frame.shape, frame.dtype)

        # DO NOT convert to BGR for MediaPipe
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            print("Hand detected!")
        else:
            print("No hands")
        # If you want to display:
        cv2.imshow("Frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    hands.close()
    cv2.destroyAllWindows()

