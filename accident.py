import cv2
import torch
import time
import threading
import numpy as np
from ultralytics import YOLO
from playsound import playsound

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Fast YOLO model

# Load video
cap = cv2.VideoCapture("accident.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Alert sound setup
alert_sound = "sound.mp3"
last_alert_time = 0
alert_cooldown = 5  # Cooldown time (seconds)
alert_playing = False

# Motion tracking variables
vehicle_positions = {}  # Track vehicle movement
vehicle_speeds = {}  # Track vehicle speed
collision_detected = False

def play_alert():
    """ Function to play alert sound """
    global alert_playing
    alert_playing = True
    try:
        playsound(alert_sound, block=True)
    except Exception as e:
        print("Error playing sound:", e)
    alert_playing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video Ended. Restarting...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Run YOLOv8 on the frame
    results = model(frame)

    # Store previous positions for speed tracking
    prev_positions = vehicle_positions.copy()
    vehicle_positions.clear()
    collision_detected = False  # Reset collision detection

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

            # Vehicle classes in COCO dataset (Car, Bus, Motorcycle, Truck)
            if cls in [2, 3, 5, 7] and conf > 0.4:
                label = f"Vehicle: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Compute aspect ratio (to detect flipped vehicles)
                aspect_ratio = (y2 - y1) / (x2 - x1)

                # Track movement
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                vehicle_positions[cls] = center

                # Calculate speed (using previous frame position)
                if cls in prev_positions:
                    prev_x, prev_y = prev_positions[cls]
                    speed = np.linalg.norm(np.array(center) - np.array((prev_x, prev_y)))

                    # Store speed
                    if cls in vehicle_speeds:
                        prev_speed = vehicle_speeds[cls]
                    else:
                        prev_speed = speed

                    vehicle_speeds[cls] = speed

                    # 🚨 **Collision Detection Conditions** 🚨
                    # Condition 1: High speed suddenly drops to very low
                    if prev_speed > 20 and speed < 2:  # Collision impact
                        collision_detected = True
                        cv2.putText(frame, "⚠️ COLLISION DETECTED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Condition 2: Flipped vehicle detected
                    if aspect_ratio > 1.8:
                        collision_detected = True
                        cv2.putText(frame, "⚠️ VEHICLE FLIPPED!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Play alert **ONLY if collision detected**
    current_time = time.time()
    if collision_detected and (current_time - last_alert_time > alert_cooldown) and not alert_playing:
        last_alert_time = current_time
        threading.Thread(target=play_alert, daemon=True).start()
        print("⚠️ Collision Alert Triggered!")

    # Show video
    cv2.imshow("Accident Detection - YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
