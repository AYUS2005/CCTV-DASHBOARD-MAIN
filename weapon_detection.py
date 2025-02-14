import cv2
import time
import pygame
import threading
from ultralytics import YOLO

# Load YOLOv8 model (Ensure yolov8n.pt is in the same directory or provide path)
model = YOLO("yolov8n.pt")

# Initialize Pygame for Sound
pygame.mixer.init()
alert_sound = "sound.mp3"

prev_time = 0
alert_cooldown = 2  # Minimum time gap between sounds (in seconds)
frame_skip = 2  # Skip every 2nd frame to speed up processing
frame_counter = 0

# Function to play alert sound
def play_alert():
    """Function to play alert sound without overlapping"""
    if not pygame.mixer.get_busy():  # Play only if no other sound is playing
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play()

# Start continuous monitoring
while True:
    # Load Video File
    camera = cv2.VideoCapture('test_video.mp4')

    # Ensure the video loaded properly
    if not camera.isOpened():
        print("Error: Couldn't open video.")
        break

    while True:
        start_time = time.time()  # Start time for FPS control

        ret, frame = camera.read()
        if not ret:
            print("Video ended, restarting...")
            break  # Restart video if ended

        if frame_counter % frame_skip != 0:  # Skip frames to reduce workload
            frame_counter += 1
            continue

        # Run YOLOv8 on the frame
        results = model(frame)

        weapon_detected = False
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])  # Confidence score
                cls = int(box.cls[0])  # Class index

                # Only detect weapons (Check class IDs based on YOLO model)
                if cls in [0, 1]:  # Update IDs based on your weapon dataset
                    weapon_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"Weapon ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        curr_time = time.time()

        # Play sound only if weapon is detected and cooldown time is passed
        if weapon_detected and (curr_time - prev_time) > alert_cooldown:
            print("🔴 Weapon Detected! Playing Alert Sound.")
            threading.Thread(target=play_alert, daemon=True).start()
            prev_time = curr_time

        # Display the frame
        cv2.imshow("Weapon Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Program Exiting...")
            camera.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            exit()

        frame_counter += 1

    camera.release()
    cv2.destroyAllWindows()  