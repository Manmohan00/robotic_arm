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
import threading
from collections import deque
import arm_code
from picamera2 import Picamera2
from libcamera import Transform
import pyttsx3
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# ================== CONFIGURATION ==================
SIMULATION_MODE = True  # Enable if hardware isn't available
FRAME_WIDTH = 1920        # Camera resolution width 640
FRAME_HEIGHT = 1080       # Camera resolution height 480
CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence for accepting gestures
FRAME_SKIP = 2 
TARGET_FPS = 15           # Process every nth frame (performance optimization)
INVERT_CAMERA = True     # Mirror camera view for more intuitive interaction

# ========== VOICE CONTROL CONFIGURATION ==========
VOICE_MODEL_PATH = "models/vosk-model-small-en-us-0.15"
VOICE_COMMANDS = ["fist", "rock", "peace", "highfive"]
tts_engine = pyttsx3.init()
recognizer_model = Model(VOICE_MODEL_PATH)
recognizer = KaldiRecognizer(recognizer_model, 16000)

def speak(text):
    print(f"speaking ... {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()
    
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

    
class ASLRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.last_letter = None
        self.last_time = 0
        self.asl_letters = ['A', 'B', 'E', 'F', 'H', 'I', 'L', 'W', 'X', 'Y']

    def recognize_asl(self, hand_landmarks):
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

    def predict_letter(self, frame, draw=True):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)
        letter = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                letter = self.recognize_asl(hand_landmarks)
                if draw:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return letter, frame

    def send_asl_command(self, letter):
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

class HandMimickingSystem:
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
        
        # Define finger joint indices in MediaPipe
        self.finger_tips = [4, 8, 12, 16, 20]  # THUMB, INDEX, MIDDLE, RING, PINKY
        self.position_history = deque(maxlen=3)  # For smoothing
        self.has_hand = False
    
    def process_hand(self, frame, draw=True):
        """Process frame to extract precise finger positions for mimicking"""
        results = self.hands.process(frame)
        control_signals = [0, 0, 0, 0, 0]  # Default position
        self.has_hand = False
        
        if results.multi_hand_landmarks:
            self.has_hand = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            if draw:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
            
            # Extract finger extension values
            landmarks = hand_landmarks.landmark
            palm_pos = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
            
            # Calculate finger extensions (0=closed, 1=mid, 2=extended)
            for i, tip_idx in enumerate(self.finger_tips):
                finger_tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
                # Determine base joint for each finger
                base_idx = tip_idx - (4 if i > 0 else 3)
                base_pos = np.array([landmarks[base_idx].x, landmarks[base_idx].y, landmarks[base_idx].z])
                
                # Calculate extension ratio
                extension = np.linalg.norm(finger_tip - base_pos) / np.linalg.norm(base_pos - palm_pos)
                
                # Map to control values (0-2)
                if extension < 0.5:
                    control_signals[i] = 0  # Closed
                elif extension < 0.8:
                    control_signals[i] = 1  # Mid-position
                else:
                    control_signals[i] = 2  # Fully extended
            
            # Store history for potential smoothing
            self.position_history.append(control_signals)
            
            # Apply smoothing if enough history exists
            if len(self.position_history) >= 3:
                control_signals = self._smooth_signals()
        
        return control_signals, self.has_hand, frame
    
    def _smooth_signals(self):
        """Apply smoothing to reduce jitter"""
        smoothed = [0, 0, 0, 0, 0]
        for i in range(5):
            # Weighted average (more weight to recent positions)
            values = [pos[i] for pos in self.position_history]
            smoothed[i] = int(round(sum(values) / len(values)))
        return smoothed
    
    def send_mimicking_commands(self, control_signals, simulation_mode=True):
        """Send control signals to robotic hand"""
        try:
            for finger, position in enumerate(control_signals):
                arm_code.move(finger, position, simulation_mode)
            return True
        except Exception as e:
            print(f"[HAND MIMICKING] Error: {e}")
            return False


class VoiceRecognizer(threading.Thread):
    def __init__(self, callback):
        super().__init__(daemon=True)
        self.callback = callback
        self._stop_event = threading.Event()
        self.q = queue.Queue()

    def callback_stream(self, indata, frames, time, status):
        if status:
            print("Mic status:", status)
        print("Received audio data")  # For debugging
        self.q.put(bytes(indata))

    def stop(self):
        self._stop_event.set()

    def run(self):
        print("...Voice control started...")
        speak("Voice control is ready.")
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=self.callback_stream):
            while not self._stop_event.is_set():
                try:
                    data = self.q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").lower().strip()
                    if text:  # Print any recognized text, even if not a command
                        print(f"[VOICE RECOGNIZED] {text}")
                    if text in VOICE_COMMANDS:
                        self.callback(text)


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
             "AwbMode": 2,  # Grey world white balance
            "ColourGains": (1.8, 1.2),  # Experiment with values
        })
        
        self.picam2.start()
        self.recognizer = HandGestureRecognizer()
        self.mimicking_system = HandMimickingSystem()
        self.mimicking_mode = False
        self.asl_recognizer = ASLRecognizer()
        self.asl_mode = False
        
        self.voice_thread = None
        self.voice_mode = True
        
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

    def handle_voice_command(self, command):
        gesture = command
        responses = {
            'fist': ("Fist bump!", "FIST"),
            'rock': ("Rock on!", "ROCK"),
            'peace': ("Peace!", "PEACE"),
            'highfive': ("High five!", "HIGHFIVE")
        }
        if gesture in responses:
            text, cmd = responses[gesture]
            speak(text)
            self.send_command(cmd)
            print(f"[VOICE ACTION] {gesture} triggered - Command: {cmd}")

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

    def start_voice_mode(self):
        if self.voice_thread is None or not self.voice_thread.is_alive():
            self.voice_thread = VoiceRecognizer(self.handle_voice_command)
            self.voice_thread.start()
            print("[SYSTEM] Voice mode started")

    def stop_voice_mode(self):
        if self.voice_thread is not None:
            self.voice_thread.stop()
            self.voice_thread.join()
            self.voice_thread = None
            print("[SYSTEM] Voice mode stopped")

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

                # Mode exclusivity: only one can be active
                if self.voice_mode:
                    # Show a message indicating voice mode is active
                    blank = np.zeros_like(frame)
                    cv2.putText(blank, "VOICE MODE ACTIVE", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.imshow(window_name, blank)
                elif self.asl_mode:
                    letter, processed_frame = self.asl_recognizer.predict_letter(frame)
                    if letter and (letter != self.asl_recognizer.last_letter or time.time() - self.asl_recognizer.last_time > 2):
                        self.asl_recognizer.send_asl_command(letter)
                        self.asl_recognizer.last_letter = letter
                        self.asl_recognizer.last_time = time.time()
                    cv2.putText(processed_frame, f"ASL: {letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(window_name, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                elif self.mimicking_mode:
                    signals, has_hand, processed_frame = self.mimicking_system.process_hand(frame)
                    if has_hand:
                        self.handle_mimicking(signals)
                    cv2.putText(processed_frame, f"Mimicking: {signals}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(window_name, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                else:
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

                # Handle mode switching
                key = cv2.waitKey(1) & 0xFF
                if key == ord('v'):
                    # Toggle voice mode
                    if not self.voice_mode:
                        self.voice_mode = True
                        self.asl_mode = False
                        self.mimicking_mode = False
                        self.start_voice_mode()
                        print("[SYSTEM] Voice mode activated")
                    else:
                        self.voice_mode = False
                        self.stop_voice_mode()
                        print("[SYSTEM] Voice mode deactivated")
                if key == ord('m'):
                    if not self.mimicking_mode:
                        self.mimicking_mode = True
                        self.asl_mode = False
                        if self.voice_mode:
                            self.voice_mode = False
                            self.stop_voice_mode()
                        print("[SYSTEM] Mimicking mode activated")
                    else:
                        self.mimicking_mode = False
                        print("[SYSTEM] Mimicking mode deactivated")
                if key == ord('a'):
                    if not self.asl_mode:
                        self.asl_mode = True
                        self.mimicking_mode = False
                        if self.voice_mode:
                            self.voice_mode = False
                            self.stop_voice_mode()
                        print("[SYSTEM] ASL mode activated")
                    else:
                        self.asl_mode = False
                        print("[SYSTEM] ASL mode deactivated")
                if key == ord('q'):
                    self.running = False

        except Exception as e:
            print(f"[ERROR] Exception in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        print("[SYSTEM] Cleaning up resources...")
        self.running = False
        self.stop_voice_mode()
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
