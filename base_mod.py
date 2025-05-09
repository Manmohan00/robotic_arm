# The system consists of four main components:
# Hand Gesture Recognizer - Uses computer vision to detect and classify hand gestures
# Arduino Interface - Communicates with an Arduino microcontroller
# Voice Control - Provides speech recognition and text-to-speech capabilities
# Main Control System - Integrates all components and manages the overall flow

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'xcb'
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import time
import numpy as np
import tflite_runtime.interpreter as tflite
# import pyttsx3
import threading
# import speech_recognition as sr
from collections import deque
import arm_code
from picamera2 import Picamera2
from libcamera import Transform

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# ================== CONFIGURATION ==================
SIMULATION_MODE = False  # Enable if hardware isn't available
FRAME_WIDTH = 1920        # Camera resolution width 640
FRAME_HEIGHT = 1080       # Camera resolution height 480
CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence for accepting gestures
FRAME_SKIP = 2 
TARGET_FPS = 15           # Process every nth frame (performance optimization)
INVERT_CAMERA = True     # Mirror camera view for more intuitive interaction

# ================== HAND GESTURE RECOGNIZER ==================
class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.target_gestures = ['fist', 'rock', 'peace', 'highfive']
        self.prediction_queue = deque(maxlen=5)
        self.current_gesture = None
        self.confidence = 0

    def preprocess_landmarks(self, landmarks):
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]
        
        wrist = points[0]
        thumb_cmc = points[1]
        index_mcp = points[5]
        middle_mcp = points[9]
        ring_mcp = points[13]
        pinky_mcp = points[17]
        
        thumb_extended = np.linalg.norm(thumb_tip - thumb_cmc) > 0.1
        index_extended = np.linalg.norm(index_tip - index_mcp) > 0.1
        middle_extended = np.linalg.norm(middle_tip - middle_mcp) > 0.1
        ring_extended = np.linalg.norm(ring_tip - ring_mcp) > 0.1
        pinky_extended = np.linalg.norm(pinky_tip - pinky_mcp) > 0.1
        
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'fist', 0.9
        elif index_extended and not middle_extended and not ring_extended and pinky_extended:
            return 'rock', 0.85
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return 'peace', 0.9
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            return 'highfive', 0.9
        else:
            return None, 0

    def predict_gesture(self, frame, draw=True):
        results = self.hands.process(frame)
        gesture = None
        confidence = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                gesture, confidence = self.preprocess_landmarks(hand_landmarks.landmark)
                self.prediction_queue.append((gesture, confidence))
        else:
            self.prediction_queue.clear()
            self.current_gesture = None
            self.confidence = 0
            return None, 0, frame

        if len(self.prediction_queue) == self.prediction_queue.maxlen:
            gesture_counts = {}
            total_confidence = {}
            
            for g, conf in self.prediction_queue:
                if g is not None:
                    gesture_counts[g] = gesture_counts.get(g, 0) + 1
                    total_confidence[g] = total_confidence.get(g, 0) + conf
            
            if gesture_counts:
                most_common = max(gesture_counts.items(), key=lambda x: x[1])
                if most_common[1] >= 3:
                    avg_confidence = total_confidence[most_common[0]] / most_common[1]
                    if avg_confidence > CONFIDENCE_THRESHOLD:
                        self.current_gesture = most_common[0]
                        self.confidence = avg_confidence
        
        return self.current_gesture, self.confidence, frame

# ================== MAIN CONTROL SYSTEM ==================
class GestureControlSystem:
    def __init__(self):
        self.running = True
        self.frame_count = 0
        self.last_gesture_time = 0
        self.last_gesture = None
        
        # Initialize camera with stable configuration
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            transform=Transform(hflip=INVERT_CAMERA, vflip=False)
        )
        self.picam2.configure(config)
        
        # Set camera controls for stable frame rate
        self.picam2.set_controls({
            "FrameDurationLimits": (int(1e6/TARGET_FPS), int(1e6/TARGET_FPS)),
            "AwbEnable": True,
            "AeEnable": True,
            "AeExposureMode": 1,  # Normal exposure
            "AeMeteringMode": 0,  # Center-weighted
            "NoiseReductionMode": 2,
            "AwbMode": 0,  # Grey world white balance
            "ExposureTime": 100000,
            "ColourGains": (0.0, 0.0),  # Experiment with values
        })
           #self.picam2.set_controls({
           # "FrameDurationLimits": (int(1e6/TARGET_FPS), int(1e6/TARGET_FPS)),
           # "AwbEnable": True,
            #"AeEnable": True,
            #"AeExposureMode": 1,  # Normal exposure
           # "AeMeteringMode": 0,  # Center-weighted
           # "NoiseReductionMode": 2,
           # "AwbMode": 0,  # Grey world white balance
            #"ColourGains": (1.8, 1.2),  # Experiment with values
            #"AeFlickerMode": 2,  # 50Hz anti-flicker
           # "ExposureTime": 10000,
          #  "AeExposureMode": controls.AeExposureModeEnum.Short,
           # "FrameDurationLimits": (10000, 10000)  # Sync with exposure
       # })
        
        self.picam2.start()
        self.recognizer = HandGestureRecognizer()

        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()

    def handle_gesture_command(self, gesture):
        responses = {
            'fist': ("Fist bump!", "FIST"),
            'rock': ("Rock on!", "ROCK"),
            'peace': ("Peace!", "PEACE"),
            'highfive': ("High five!", "HIGHFIVE")
        }
        
        current_time = time.time()
        if gesture in responses and (gesture != self.last_gesture or current_time - self.last_gesture_time > 2.0):
            text, cmd = responses[gesture]
            self.send_command(cmd)
            self.last_gesture = gesture
            self.last_gesture_time = current_time
            print(f"[ACTION] {gesture} detected - Command: {cmd}")

    def send_command(self, command):
        print('Sending command to arm:', command)
        try:
            positions = {
                'FIST': [0,0,0,0,0],
                'ROCK': [2,2,0,0,2],
                'PEACE': [0,2,2,0,0],
                'HIGHFIVE': [2,2,2,2,2]
            }
            
            if command in positions:
                for finger, pos in enumerate(positions[command]):
                    arm_code.move(finger, pos, SIMULATION_MODE)
            return True
        except Exception as e:
            print(f"[BIONIC ARM HARDWARE] Error sending command: {e}")
            return False

    def calculate_fps(self):
        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_queue.append(fps)
        return np.mean(self.fps_queue)

    def main_loop(self):
        print("[SYSTEM] Starting gesture recognition system")
        window_name = 'Hand Gesture Control'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_interval = 1.0 / TARGET_FPS
        last_frame_time = time.time()

        try:
            while self.running:
                current_time = time.time()
                elapsed = current_time - last_frame_time

                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

                last_frame_time = current_time
 
                frame = self.picam2.capture_array()

                if INVERT_CAMERA:
                    frame = cv2.flip(frame, 1)

                self.frame_count += 1
                if self.frame_count % FRAME_SKIP != 0:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        self.running = False
                    continue

                gesture, confidence, processed_frame = self.recognizer.predict_gesture(frame)

                if gesture and confidence > CONFIDENCE_THRESHOLD:
                    self.handle_gesture_command(gesture)

                if gesture:
                    cv2.putText(
                        processed_frame, 
                        f"{gesture} ({confidence:.2f})", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255), 
                        2
                    )

                fps = self.calculate_fps()
                cv2.putText(
                    processed_frame,
                    f"FPS: {fps:.1f}",
                    (FRAME_WIDTH - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(window_name, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

        except Exception as e:
            print(f"[ERROR] Exception in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        print("[SYSTEM] Cleaning up resources...")
        self.running = False
        self.picam2.stop()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        print("[SYSTEM] System shut down")

if __name__ == "__main__":
    try:
        system = GestureControlSystem()
        system.main_loop()
    except KeyboardInterrupt:
        print("\n[USER] Program stopped by user")
    except Exception as e:
        print(f"[ERROR] {e}")
