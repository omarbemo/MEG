import cv2
from ultralytics import YOLO
import threading
import os
import time



# --- CONFIGURATION ---
CAM_IP = "10.1.2.200"
CAM_USER = "admin"
CAM_PASS = "megglass_1"
CAM_RTSP = f"rtsp://{CAM_USER}:{CAM_PASS}@{CAM_IP}:554/Streaming/Channels/101"

# Since you want to test on the laptop camera right now, we set the source to 1 (or 0).
# Change this to 0 if your external camera keeps showing instead!
CAM_SOURCE = 0 

# Force TCP to prevent h264 packet loss errors
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

print("Loading AI Model")
model = YOLO('yolov8s.pt') 

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread to prevent RTSP buffer lag (latency).
    """
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.src = src

    def start(self):    
        threading.Thread(target=self.get, args=(), daemon=True).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                # Attempt to reconnect if stream drops
                time.sleep(1)
                self.stream = cv2.VideoCapture(self.src)
                (self.grabbed, self.frame) = self.stream.read()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Start video stream thread
print("Connecting to camera stream...")
cap = VideoGet(CAM_SOURCE).start()
time.sleep(1) # Give camera time to connect

# Target classes to detect
TARGET_CLASSES = ['bottle', 'vase', 'cup', 'kettle', 'wine glass']

while not cap.stopped:
    success, frame = cap.read()
    if not success or frame is None:
        continue

    # Perform Tracking
    results = model.track(frame, persist=True, conf=0.20, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box)
            
            # Determine color based on the object detected
            # OpenCV uses BGR (Blue, Green, Red)
            if label == 'bottle':
                color = (255, 0, 0)       # Blue for bottles
            elif label in ['wine glass', 'vase', 'cup', 'bowl']:
                color = (0, 255, 255)     # Yellow for glass/ceramic things
            elif label == 'person':
                color = (0, 0, 255)       # Red for people
            else:
                color = (0, 255, 0)       # Green for everything else
            
            # Draw simple bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("SmartVision - YOLO Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.stop()
cv2.destroyAllWindows()