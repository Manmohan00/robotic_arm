import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO
import arm_code  # Your Maestro-based move() function

# --- CONFIGURATION ---
SIMULATION_MODE = False  # Set True for testing without hardware
TRIGGER_TIME = 2         # Seconds object must be visible to trigger gesture

# Map YOLO class names to ASL gestures
OBJECT_TO_ASL = {
    "bottle": "FIST",
    "book": "HIGHFIVE",
    "cup": "PEACE",
    "person": "ROCK",  # Example, add more as needed
}

# Map gesture name to finger positions for one hand
GESTURE_TO_POSITIONS = {
    'FIST':      [0, 0, 0, 0, 0],  # All closed
    'ROCK':      [2, 2, 0, 0, 2],  # Rock sign
    'PEACE':     [0, 2, 2, 0, 0],  # V sign
    'HIGHFIVE':  [2, 2, 2, 2, 2],  # All open
}

def send_command(gesture):
    """Send gesture command to bionic arm using arm_code.move"""
    print('Sending command to arm:', gesture)
    try:
        if gesture in GESTURE_TO_POSITIONS:
            for finger, pos in enumerate(GESTURE_TO_POSITIONS[gesture]):
                arm_code.move(finger, pos, SIMULATION_MODE)
        return True
    except Exception as e:
        print(f"[BIONIC ARM HARDWARE] Error sending command: {e}")
        return False

def main():
    # Initialize camera
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    # Load YOLO model (use yolov8n.pt or yolo11n.pt as available)
    model = YOLO("yolov8n.pt")

    last_detected = None
    visible_since = 0
    cooldown_until = 0

    print("System started. Press 'q' to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            results = model(frame)
            detected_class = None

            # Find first relevant object (by confidence)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    print("Detected:", model.names[cls], "Confidence:", conf)
                    class_name = model.names[cls]
                    if class_name in OBJECT_TO_ASL and conf > 0.5:
                        detected_class = class_name
                        break
                if detected_class:
                    break

            now = time.time()
            # Context-aware filtering and 2–3s trigger logic
            if detected_class:
                if detected_class == last_detected:
                    if visible_since == 0:
                        visible_since = now
                    # Only trigger if visible for TRIGGER_TIME and not in cooldown
                    if now - visible_since > TRIGGER_TIME and now > cooldown_until:
                        gesture = OBJECT_TO_ASL[detected_class]
                        send_command(gesture)
                        cooldown_until = now + 3  # 3s cooldown to avoid retrigger
                else:
                    last_detected = detected_class
                    visible_since = now
            else:
                last_detected = None
                visible_since = 0

            # Draw info on frame
            text = f"Detected: {detected_class if detected_class else 'None'}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("YOLO ASL Bionic Arm", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
