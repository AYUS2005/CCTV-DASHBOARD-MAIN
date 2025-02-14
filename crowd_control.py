import cv2
import time
import pygame
import threading
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO('yolov8n.pt').to('cpu')  # Using CPU for inference

# Initialize Pygame for Sound Alerts
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alarm.mp3")  # Load sound

# Alert control variables
crowd_threshold = 1  # Minimum people required to trigger alert
alert_active = False  # Flag to prevent continuous sound

def play_alert():
    """Plays alert sound."""
    global alert_active
    if not alert_active:
        alert_active = True
        alert_sound.play(-1)  # 🔥 Play continuously until stopped

def stop_alert():
    """Stops alert sound."""
    global alert_active
    if alert_active:
        alert_sound.stop()  # ⛔ Stop sound instantly
        alert_active = False

# Open IP Camera Stream (Change IP as per your mobile)
camera_url = "http://192.168.0.100:8080/video"  # Use your phone's IP webcam URL
camera = cv2.VideoCapture(camera_url)

# Set FPS for smoother performance
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Get screen resolution for full-screen display
screen_width = int(camera.get(3))  # Get width
screen_height = int(camera.get(4))  # Get height

# Set OpenCV window to fullscreen
cv2.namedWindow("Crowd Control - YOLOv8", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Crowd Control - YOLOv8", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame = None  # Initialize frame

def capture_frame():
    """Capture frame from IP camera in a separate thread for better performance."""
    global frame
    while True:
        ret, new_frame = camera.read()
        if ret:
            frame = new_frame

# Start background thread for capturing frames
threading.Thread(target=capture_frame, daemon=True).start()

while True:
    if frame is None:
        continue  # 🔴 FIX: Wait until frame is available

    # Convert frame to NumPy array (Faster processing)
    frame_np = np.array(frame)

    # Run YOLOv8 Inference (Faster with torch.no_grad)
    with torch.no_grad():
        results = model.track(frame_np, conf=0.5, persist=True)

    # Count persons detected
    person_count = sum(1 for obj in results[0].boxes.cls if int(obj) == 0)  # Class 0 = 'person'

    # Draw bounding boxes only for 'person' class
    for i, box in enumerate(results[0].boxes.xyxy):
        if int(results[0].boxes.cls[i]) == 0:  # Class 0 = Person
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display person count
    cv2.putText(frame_np, f"People Count: {person_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Trigger Alert only if crowd exceeds limit
    if person_count >= crowd_threshold:
        threading.Thread(target=play_alert, daemon=True).start()
    else:
        stop_alert()  # ⛔ Stop sound immediately when count is below threshold

    # Display Video in Full Screen
    frame_resized = cv2.resize(frame_np, (screen_width, screen_height))  # Full resolution
    cv2.imshow("Crowd Control - YOLOv8", frame_resized)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
