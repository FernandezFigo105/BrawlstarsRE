import cv2
import numpy as np
from ultralytics import YOLO

# ==============================
# Configuration
# ==============================
VIDEO_PATH = "C:/Users/Figo/Desktop/BrawlStars/BlueStacks App Player 2 2025-02-21 21-04-44.mp4"          # Your screen recording
MODEL_PATH = "C:/Users/Figo/Desktop/BrawlStars/runs/detect/train13/weights/best.pt"                          # Your trained model
OUTPUT_PATH = "output_detected.mp4"            # Save result video
CONFIDENCE_THRESHOLD = 0.3                      # Minimum confidence to show
SHOW_VIDEO = True                               # Set to False for headless
SAVE_VIDEO = True                               # Save output to file

# Load the YOLO model
model = YOLO(MODEL_PATH)

# Open the video
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define class names (optional: if not in model)
# If your model has custom classes, they are already in model.names
# But you can override or verify:
CLASS_NAMES = model.names  # e.g., {0: 'player', 1: 'enemy', ...}

# Generate consistent colors for each class
np.random.seed(42)
COLORS = {}
for class_id in CLASS_NAMES:
    COLORS[class_id] = tuple(np.random.randint(0, 255, 3).tolist())

# Video writer to save output
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Loop over results and draw boxes
    for result in results:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Convert to integers

            # Get confidence and class
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = CLASS_NAMES[cls_id]

            # Get color for this class
            color = COLORS[cls_id]

            # Draw bounding box (thicker for visibility)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Label text
            label = f"{class_name} {conf:.2f}"

            # Text background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the frame (optional)
    if SHOW_VIDEO:
        cv2.imshow("YOLO Detection - Brawl Stars", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Save to output video
    if SAVE_VIDEO:
        out.write(frame)

# Release resources
cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()

print(f"[INFO] Detection complete. Output saved to {OUTPUT_PATH}")