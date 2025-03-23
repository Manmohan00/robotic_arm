# The system consists of four main components:

# Hand Gesture Recognizer - Uses computer vision to detect and classify hand gestures

# Arduino Interface - Communicates with an Arduino microcontroller

# Voice Control - Provides speech recognition and text-to-speech capabilities

# Main Control System - Integrates all components and manages the overall flow


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import mediapipe as mp
import serial.tools.list_ports
import time
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import speech_recognition as sr
from collections import deque
import arm_code

# ================== CONFIGURATION ==================
SIMULATION_MODE = False  # Enable if Arduino hardware isn't available
FRAME_WIDTH = 640        # Camera resolution width
FRAME_HEIGHT = 480       # Camera resolution height
CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence for accepting gestures
FRAME_SKIP = 2           # Process every nth frame (performance optimization)
INVERT_CAMERA = True     # Mirror camera view for more intuitive interaction


# ================== HAND GESTURE RECOGNIZER ==================
class HandGestureRecognizer:
    
    # This recognizer uses MediaPipe's hand tracking to identify key points on the hand, then applies a custom classification algorithm to determine which gesture is being performed.

    # The key method preprocess_landmarks() analyzes finger positions to identify gestures:

    #     Fist: All fingers closed

    #     Rock: Index and pinky extended, others closed

    #     Peace: Index and middle fingers extended

    #     High Five: All fingers extended

    # The system uses temporal smoothing with a queue to stabilize predictions, only recognizing a gesture when it's consistently detected over multiple frames.
    
    def __init__(self):
        # Initialize MediaPipe hand tracking components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # For video processing
            max_num_hands=1,          # Only track one hand for simplicity
            model_complexity=0,       # Faster processing with lower complexity
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Target gestures to recognize
        self.target_gestures = ['fist', 'rock', 'peace', 'highfive']
        
        # Queue for smoothing predictions (prevents flickering)
        self.prediction_queue = deque(maxlen=5)
        self.current_gesture = None
        self.confidence = 0
    
    def preprocess_landmarks(self, landmarks):
        """Extract hand features for gesture classification"""
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Calculate features (simplified version)
        # 1. Fingers folded or extended (using relative positions)
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
        
        # Normalize tips by distance from MCP joints
        thumb_extended = np.linalg.norm(thumb_tip - thumb_cmc) > 0.1
        index_extended = np.linalg.norm(index_tip - index_mcp) > 0.1
        middle_extended = np.linalg.norm(middle_tip - middle_mcp) > 0.1
        ring_extended = np.linalg.norm(ring_tip - ring_mcp) > 0.1
        pinky_extended = np.linalg.norm(pinky_tip - pinky_mcp) > 0.1
        
        # Simple gesture classification based on finger extension
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
        """Detect hand and predict gesture"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        gesture = None
        confidence = 0
        
        # Draw landmarks if hands detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Predict gesture from landmarks
                gesture, confidence = self.preprocess_landmarks(hand_landmarks.landmark)
                
                # Add to queue for smoothing
                self.prediction_queue.append((gesture, confidence))
        else:
            # Clear queue when no hand detected
            self.prediction_queue.clear()
            self.current_gesture = None
            self.confidence = 0
            return None, 0, frame
        
        # Process prediction queue for stability
        if len(self.prediction_queue) == self.prediction_queue.maxlen:
            # Count occurrences of each gesture
            gesture_counts = {}
            total_confidence = {}
            
            for g, conf in self.prediction_queue:
                if g is not None:
                    gesture_counts[g] = gesture_counts.get(g, 0) + 1
                    total_confidence[g] = total_confidence.get(g, 0) + conf
            
            # Find most common gesture
            if gesture_counts:
                most_common = max(gesture_counts.items(), key=lambda x: x[1])
                
                # Only update if it appears in majority of frames with good confidence
                if most_common[1] >= 3:  # At least 3 out of 5 frames
                    avg_confidence = total_confidence[most_common[0]] / most_common[1]
                    
                    if avg_confidence > CONFIDENCE_THRESHOLD:
                        self.current_gesture = most_common[0]
                        self.confidence = avg_confidence
        
        return self.current_gesture, self.confidence, frame

# ================== COMMUNICATION WITH ARDUINO ==================
# class ArduinoInterface:
#     # This component handles communication with an Arduino microcontroller:

#     # Automatically detects and connects to Arduino over serial port

#     # Sends gesture commands as text strings (e.g., "FIST", "ROCK")

#     # Provides a simulation mode for testing without hardware

#     # The Arduino code (provided as comments) configures specific pins for each gesture and lights up the appropriate pin when a command is received.
    
#     def __init__(self, simulation_mode=False):
#         self.simulation_mode = simulation_mode
#         self.serial_port = None
#         self.connected = False
        
#         if not self.simulation_mode:
#             self.connect()
            
#     def connect(self):
#         """Connect to Arduino via serial port"""
#         try:
#             ports = list(serial.tools.list_ports.comports())
#             for port in ports:
#                 print('port.description ==>', port.description)
#                 print('port ==>', port)
#                 if port.description == 'COM6' or port.description == 'COM3':
#                     # self.serial_port = serial.Serial(port.device, 115200, timeout=0.1)
#                     print(f"[ARDUINO] Connected to {port.device}")
#                     self.connected = True
                    
#                     # Wait for Arduino to initialize
#                     time.sleep(2)
#                     return
            
#             print("[ARDUINO] No Arduino found - using simulation mode")
#         except Exception as e:
#             print(f"[ARDUINO] Connection error: {e}")
    
#     def send_command(self, command):
#         """Send command to Arduino"""
#         if self.simulation_mode:
#             print(f"[SIM] Sending command: {command}")
#             return True
        
#         if not self.connected or not self.serial_port:
#             return False
        
#         # arm_code.move(finger=3,position=2)
        
#         print('sending command to arm ....', command)
        
#         try:
#             if command == 'FIST':
#                 arm_code.move(finger=0,position=0)
#                 arm_code.move(finger=1,position=0)
#                 arm_code.move(finger=2,position=0)
#                 arm_code.move(finger=3,position=0)
#                 arm_code.move(finger=4,position=0)
#             elif command == 'ROCK':
#                 arm_code.move(finger=0,position=1)
#                 arm_code.move(finger=1,position=1)
#                 arm_code.move(finger=2,position=0)
#                 arm_code.move(finger=3,position=0)
#                 arm_code.move(finger=4,position=1)
#             elif command == 'PEACE':
#                 arm_code.move(finger=0,position=0)
#                 arm_code.move(finger=1,position=1)
#                 arm_code.move(finger=2,position=1)
#                 arm_code.move(finger=3,position=0)
#                 arm_code.move(finger=4,position=0)
#             elif command == 'HIGHFIVE':
#                 arm_code.move(finger=0,position=1)
#                 arm_code.move(finger=1,position=1)
#                 arm_code.move(finger=2,position=1)
#                 arm_code.move(finger=3,position=1)
#                 arm_code.move(finger=4,position=1)
#             return True
#         except Exception as e:
#             print(f"[ARDUINO] Error sending command: {e}")
#             return False
    
#     def close(self):
#         """Close serial connection"""
#         if self.connected and self.serial_port:
#             self.serial_port.close()
#             self.connected = False

# ================== VOICE RECOGNITION ==================
class VoiceControl:
    # This component provides two key functionalities:

    #   Speech recognition - Listens for voice commands in a background thread

    #   Text-to-speech - Provides audible feedback when gestures are recognized

    # The voice recognition continuously monitors for specific commands like "fist", "rock", "peace", and "high five", executing the same actions as when the gestures are physically detected.
    
    def __init__(self, callback):
        # Setup text-to-speech engine
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        self.tts.setProperty('volume', 0.8)
        
        # Setup speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.microphone = sr.Microphone()
        self.running = True
        
        # Start voice recognition in background thread
        threading.Thread(target=self.recognition_loop, daemon=True).start()
    
    def speak(self, text):
        """Speak text without blocking the main thread"""
        if hasattr(self, 'is_speaking') and self.is_speaking:
            print(f"[VOICE] Already speaking, skipping: {text}")
            return
        
        def speak_thread():
            try:
                self.is_speaking = True
                self.tts.say(text)
                self.tts.runAndWait()
            except Exception as e:
                print(f"[VOICE] Error: {e}")
            finally:
                self.is_speaking = False
        
        threading.Thread(target=speak_thread, daemon=True).start()
        self.is_speaking = True  # Mark as speaking right away


    
    def recognition_loop(self):
        """Background loop for voice recognition"""
        with self.microphone as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source)
            print("[VOICE] Voice recognition started")
            
            while self.running:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"[VOICE] Recognized: {text}")
                    
                    # Check for gesture commands
                    if 'fist' in text or 'close' in text:
                        self.callback('fist')
                    elif 'rock' in text:
                        self.callback('rock')
                    elif 'peace' in text:
                        self.callback('peace')
                    elif 'five' in text or 'high' in text:
                        self.callback('highfive')
                    elif 'stop' in text or 'exit' in text or 'quit' in text:
                        self.speak("Stopping system")
                        time.sleep(1)
                        self.running = False
                        
                except sr.WaitTimeoutError:
                    pass  # Continue listening
                except sr.UnknownValueError:
                    pass  # Speech not understood
                except sr.RequestError:
                    print("[VOICE] Could not request results from Google")
                except Exception as e:
                    print(f"[VOICE] Error: {e}")
                
                time.sleep(0.1)
    
    def stop(self):
        """Stop voice recognition"""
        self.running = False

# ================== MAIN CONTROL SYSTEM ==================
class GestureControlSystem:
    # This class integrates all components and manages the system flow. The main_loop() method implements the core processing pipeline:

    #     Frame capture: Get an image from the camera

    #     Frame processing:

    #       Skip frames as needed for performance

    #       Detect hand and recognize gestures

    #     Command handling:

    #       When a gesture is detected with sufficient confidence, execute corresponding action

    #       Speak feedback through text-to-speech

    #       Send command to Arduino

    #     Display:

    #       Show the processed frame with annotations

    #       Display recognized gesture and confidence

    #       Show current FPS
    def __init__(self):
        # System state
        self.running = True
        self.frame_count = 0
        self.last_gesture_time = 0
        self.last_gesture = None
        
        # Initialize components
        self.recognizer = HandGestureRecognizer()
        # self.arduino = ArduinoInterface(simulation_mode=SIMULATION_MODE)
        # self.voice = VoiceControl(self.handle_gesture_command)
        
        # Performance monitoring
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        # Camera initialization
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    def handle_gesture_command(self, gesture):
        """Handle recognized gesture with response"""
        responses = {
            'fist': ("Fist bump!", "FIST"),
            'rock': ("Rock on!", "ROCK"),
            'peace': ("Peace!", "PEACE"),
            'highfive': ("High five!", "HIGHFIVE")
        }
        
        current_time = time.time()
        if gesture in responses and (gesture != self.last_gesture or current_time - self.last_gesture_time > 2.0):
            text, cmd = responses[gesture]
            # self.voice.speak(text)
            self.send_command(cmd)
            
            self.last_gesture = gesture
            self.last_gesture_time = current_time
            
            print(f"[ACTION] {gesture} detected - Command: {cmd}")
            
    def send_command(self, command):
        print('sending command to arm ....', command)
        
        try:
            if command == 'FIST':
                arm_code.move(finger=0,position=0)
                arm_code.move(finger=1,position=0)
                arm_code.move(finger=2,position=0)
                arm_code.move(finger=3,position=0)
                arm_code.move(finger=4,position=0)
            elif command == 'ROCK':
                arm_code.move(finger=0,position=2)
                arm_code.move(finger=1,position=2)
                arm_code.move(finger=2,position=0)
                arm_code.move(finger=3,position=0)
                arm_code.move(finger=4,position=2)
            elif command == 'PEACE':
                arm_code.move(finger=0,position=0)
                arm_code.move(finger=1,position=2)
                arm_code.move(finger=2,position=2)
                arm_code.move(finger=3,position=0)
                arm_code.move(finger=4,position=0)
            elif command == 'HIGHFIVE':
                arm_code.move(finger=0,position=2)
                arm_code.move(finger=1,position=2)
                arm_code.move(finger=2,position=2)
                arm_code.move(finger=3,position=2)
                arm_code.move(finger=4,position=2)
            return True
        except Exception as e:
            print(f"[ARDUINO] Error sending command: {e}")
            return False
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_queue.append(fps)
        return np.mean(self.fps_queue)
    
    def main_loop(self):
        """Main processing loop"""
        print("[SYSTEM] Starting gesture recognition system")
        
        # Create window
        window_name = 'Hand Gesture Control'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while self.running and self.cap.isOpened():
                # self.voice.running
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Invert camera if needed
                if INVERT_CAMERA:
                    frame = cv2.flip(frame, 1)
                
                # Process only every nth frame for performance
                self.frame_count += 1
                if self.frame_count % FRAME_SKIP != 0:
                    # Still display the frame but skip processing
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        self.running = False
                        break
                    continue
                
                # Process frame for gesture recognition
                gesture, confidence, processed_frame = self.recognizer.predict_gesture(frame)
                
                # Handle detected gesture
                if gesture and confidence > CONFIDENCE_THRESHOLD:
                    self.handle_gesture_command(gesture)
                
                # Display gesture and confidence
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
                
                # Calculate and display FPS
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
                
                # Show frame
                cv2.imshow(window_name, processed_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    self.running = False
                    break
                
                # Check if window is closed (THIS FIXES THE WINDOW CLOSING ISSUE)
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
                    break
        
        except Exception as e:
            print(f"[ERROR] Exception in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources properly"""
        print("[SYSTEM] Cleaning up resources...")
        
        # Stop all threads and components
        self.running = False
        # self.voice.stop()
        
        # Release camera
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        # Close Arduino connection
        # self.arduino.close()
        
        # Close all windows - multiple calls to ensure proper closure
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        
        print("[SYSTEM] System shut down")

# ================== ARDUINO CODE ==================
"""
// Arduino code for gesture control system

#define LED_PIN 13  // Built-in LED for status

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  
  // Configure pins for different gestures
  pinMode(2, OUTPUT);  // Pin for FIST
  pinMode(3, OUTPUT);  // Pin for ROCK
  pinMode(4, OUTPUT);  // Pin for PEACE
  pinMode(5, OUTPUT);  // Pin for HIGHFIVE
  
  // Start-up indication
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
  
  Serial.println("Gesture Control System Ready");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Reset all outputs
    digitalWrite(2, LOW);
    digitalWrite(3, LOW);
    digitalWrite(4, LOW);
    digitalWrite(5, LOW);
    
    // Process commands
    if (command == "FIST") {
      digitalWrite(2, HIGH);
      blink();
    }
    else if (command == "ROCK") {
      digitalWrite(3, HIGH);
      blink();
    }
    else if (command == "PEACE") {
      digitalWrite(4, HIGH);
      blink();
    }
    else if (command == "HIGHFIVE") {
      digitalWrite(5, HIGH);
      blink();
    }
    
    // Acknowledge receipt
    Serial.println("Received: " + command);
  }
}

void blink() {
  // Visual feedback for received command
  digitalWrite(LED_PIN, HIGH);
  delay(50);
  digitalWrite(LED_PIN, LOW);
}
"""

# ================== MAIN ==================
if __name__ == "__main__":
    try:
        system = GestureControlSystem()
        system.main_loop()
    except KeyboardInterrupt:
        print("\n[USER] Program stopped by user")
    except Exception as e:
        print(f"[ERROR] {e}")