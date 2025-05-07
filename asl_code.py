import cv2
import time
import mediapipe as mp
import numpy as np

try:
    import arm_code  # Comment this out if not available
except ImportError:
    class arm_code:
        @staticmethod
        def move(finger, position):
            print(f"[SIMULATION] Moving finger {finger} to position {position}")

# Configuration
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CONFIDENCE_THRESHOLD = 0.85
INVERT_CAMERA = True
FRAME_SKIP = 2

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

asl_letters = ['A', 'B', 'E', 'F', 'H', 'I', 'L', 'W', 'X', 'Y']

def recognize_asl(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]  # Thumb to pinky
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    
    fingers = []

    # Thumb: tip is to the left of the IP joint (right hand)
    thumb_up = landmarks[4][0] < landmarks[3][0]
    fingers.append(thumb_up)

    # Other fingers: tip above PIP
    for i in range(1, 5):
        fingers.append(landmarks[tip_ids[i]][1] < landmarks[tip_ids[i] - 2][1])

    # A: Thumb up, all other fingers down
    if fingers == [True, False, False, False, False]:
        return "A"
    
    # B: All fingers up except thumb
    elif fingers == [False, True, True, True, True]:
        return "B"

    # F: Index down slightly, others up
    index_down = landmarks[8][1] > landmarks[6][1]
    middle_up = landmarks[12][1] < landmarks[10][1]
    ring_up = landmarks[16][1] < landmarks[14][1]
    pinky_up = landmarks[20][1] < landmarks[18][1]

    if index_down and thumb_up and middle_up and ring_up and pinky_up:
        return "F"
    
    # H: Index + middle up, others down
    if fingers == [False, True, True, False, False]:
        return "H"
    
    # I: Only pinky up
    if fingers == [False, False, False, False, True]:
        return "I"
    
    # L: Thumb and index up
    if fingers == [True, True, False, False, False]:
        return "L"
    
    # W: Index + middle + ring up
    if fingers == [False, True, True, True, False]:
        return "W"
    
    # X: Index slightly bent (tip below DIP but above PIP)
    index_tip_y = landmarks[8][1]
    index_dip_y = landmarks[7][1]
    index_pip_y = landmarks[6][1]
    index_slightly_bent = index_tip_y > index_dip_y and index_tip_y < index_pip_y
    if index_slightly_bent and not fingers[0] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "X"
    
    # Y: Thumb and pinky up
    if fingers == [True, False, False, False, True]:
        return "Y"

    # E: All fingers slightly bent (tip below PIP but above MCP)
    def slightly_bent(tip, pip, mcp):
        return landmarks[tip][1] > landmarks[pip][1] and landmarks[tip][1] < landmarks[mcp][1]
    
    if (not fingers[0] and
        all(slightly_bent(t, t-2, t-3) for t in tip_ids[1:])):
        return "E"

    return None


# Command mapping to arm positions 
def send_asl_command(letter):
    print(f"[ASL] Recognized: {letter}")
    try:
        if letter == 'A':
            for f in range(5): arm_code.move(finger=f, position=0)
        elif letter == 'B':
            for f in range(1, 5): arm_code.move(finger=f, position=2)
            arm_code.move(finger=0, position=0)
        elif letter == 'F':
            for f in range(5): arm_code.move(finger=f, position=2)
            arm_code.move(finger=1, position=0)  # index folded
        elif letter == 'H':
            arm_code.move(finger=1, position=2)
            arm_code.move(finger=2, position=2)
            for f in [0, 3, 4]: arm_code.move(finger=f, position=0)
        elif letter == 'I':
            arm_code.move(finger=0, position=0)
            for f in range(1, 4): arm_code.move(finger=f, position=0)
            arm_code.move(finger=4, position=2)
        elif letter == 'L':
            arm_code.move(finger=0, position=2)
            arm_code.move(finger=1, position=2)
            for f in [2, 3, 4]: arm_code.move(finger=f, position=0)
        elif letter == 'W':
            for f in [1, 2, 3]: arm_code.move(finger=f, position=2)
            for f in [0, 4]: arm_code.move(finger=f, position=0)
        elif letter == 'X':
            arm_code.move(finger=1, position=2)
            for f in [0, 2, 3, 4]: arm_code.move(finger=f, position=0)
        elif letter == 'Y':
            arm_code.move(finger=0, position=2)
            arm_code.move(finger=4, position=2)
            for f in [1, 2, 3]: arm_code.move(finger=f, position=0)
    except Exception as e:
        print(f"[ERROR] Failed to send ASL command: {e}")

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        last_letter = None
        last_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if INVERT_CAMERA:
                frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Recognize ASL letter from landmarks
                    letter = recognize_asl(hand_landmarks)

                    # Only trigger on new gesture or after 2 seconds
                    if letter and (letter != last_letter or time.time() - last_time > 2):
                        send_asl_command(letter)
                        last_letter = letter
                        last_time = time.time()

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    cv2.putText(frame, f"ASL: {letter}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("ASL Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
