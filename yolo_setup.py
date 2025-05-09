import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLO11n model
model = YOLO("yolo11n.pt")

try:
	while True:
		frame = picam2.capture_array()
		# frame_resized = cv2.resize(frame)
		results = model(frame)
		
		result = results[0]
		annotated_frame = result.plot()
		cv2.imshow("Camera", annotated_frame)
		if cv2.waitKey(1) == ord("q"):
			break
			
except KeyboardInterrupt:
    print("Interrupted by user.")
    
finally:
    picam2.stop()  # Properly stop the camera
    cv2.destroyAllWindows()
    
