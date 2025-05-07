import cv2
import numpy as np
import time

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

# Color ranges in HSV space
color_ranges = {
    'red': [(0, 120, 70), (10, 255, 255)],  # Lower and Upper bounds for Red
    'orange': [(10, 100, 100), (25, 255, 255)],  # Lower and Upper bounds for Orange
    'yellow': [(25, 50, 50), (35, 255, 255)],  # Lower and Upper bounds for Yellow
    'blue': [(90, 50, 50), (130, 255, 255)],  # Lower and Upper bounds for Blue
    'pink': [(140, 50, 50), (170, 255, 255)],  # Lower and Upper bounds for Pink
}

# Command mapping for colors to ASL signs
color_to_asl = {
    'red': 'R',     # R for Red
    'orange': 'O',  # O for Orange
    'yellow': 'Y',  # Y for Yellow
    'blue': 'B',    # B for Blue
    'pink': 'P'     # P for Pink
}

# Command mapping to arm positions 
def send_asl_command(letter):
    print(f"[ASL] Recognized: {letter}")
    try:
        if letter == 'R':  # Red
            for f in range(5): arm_code.move(finger=f, position=0)  # Example gesture for Red
        elif letter == 'O':  # Orange
            for f in range(1, 5): arm_code.move(finger=f, position=2)
            arm_code.move(finger=0, position=0)
        elif letter == 'Y':  # Yellow
            for f in range(5): arm_code.move(finger=f, position=2)  # Example gesture for Yellow
        elif letter == 'B':  # Blue
            arm_code.move(finger=1, position=2)  # Blue gesture example
            arm_code.move(finger=2, position=2)
            arm_code.move(finger=0, position=0)
        elif letter == 'P':  # Pink
            arm_code.move(finger=0, position=2)  # Pink gesture example
            arm_code.move(finger=4, position=2)
            for f in [1, 2, 3]: arm_code.move(finger=f, position=0)
    except Exception as e:
        print(f"[ERROR] Failed to send ASL command: {e}")

# Color detection and ASL command sending
class ColorASLController:
    def __init__(self):
        self.last_color = None
        self.last_time = 0

    def detect_color(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Check each color range
        for color, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)

            # Create a mask for the current color
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # If any color is detected, return it
            if np.any(mask):
                return color
        
        return None

    def send_color_asl_command(self, color):
        print(f"[Color ASL] Detected Color: {color}")
        asl_letter = color_to_asl.get(color)

        if asl_letter:
            send_asl_command(asl_letter)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if INVERT_CAMERA:
                frame = cv2.flip(frame, 1)

            color = self.detect_color(frame)
            
            if color and (color != self.last_color or time.time() - self.last_time > 2):
                self.send_color_asl_command(color)
                self.last_color = color
                self.last_time = time.time()

            # Display the detected color on the frame
            if color:
                cv2.putText(frame, f"Detected Color: {color}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Color ASL Control", frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    color_controller = ColorASLController()
    color_controller.run()
