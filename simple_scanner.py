import cv2
import os
import re
import time
from collections import deque, Counter

import pytesseract

# Configure Tesseract path for Windows (Default installation location)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Setup Camera Connection
CAM_IP = "10.1.2.200"
CAM_USER = "admin"
CAM_PASS = "megglass_1"
CAM_RTSP = f"rtsp://{CAM_USER}:{CAM_PASS}@{CAM_IP}:554/Streaming/Channels/101"

# Force TCP to prevent h264 packet loss errors
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# --- Majority-vote confirmation settings ---
# A single OCR pass can misread a digit (0/9, 7/9 are common mix-ups) due
# to a blurry moment or bad angle. Instead of trusting the very first
# reading, we look at the last VOTE_WINDOW OCR passes and only confirm a
# code once it shows up in at least VOTE_THRESHOLD of them. This is pure
# text/logic — it does NOT change anything about how the image looks.
VOTE_WINDOW = 5
VOTE_THRESHOLD = 3

# --- Overlay masking ---
# Your camera burns a live timestamp (top-left) and a "Camera 01" label
# (bottom-right) directly into the video. The timestamp changes every
# second, so OCR was reading "static pallet number + constantly-changing
# clock" mashed together, which is why every read looked different even
# though the paper never moved. We blank out those two fixed corners
# (as fractions of frame size, so it works at any resolution) AFTER
# thresholding — this does not change the Otsu/grayscale pipeline at all,
# it just hides two known trouble spots from OCR.
# Format: (x1_frac, x2_frac, y1_frac, y2_frac) — adjust if the overlay
# isn't fully covered (or too much is covered) for your camera's layout.
TIMESTAMP_REGION = (0.0, 0.35, 0.0, 0.12)
CAMERA_LABEL_REGION = (0.65, 1.0, 0.85, 1.0)


def mask_region(img, region):
    h, w = img.shape[:2]
    x1, x2, y1, y2 = region
    img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 255


print("Connecting to camera... Please wait.")
cap = cv2.VideoCapture(CAM_RTSP, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not connect to camera.")
    exit()

print("Camera connected! A window will pop up.")
print("Hold your paper up to the camera. Press 'q' on your keyboard to quit.")
print("-" * 50)

frame_count = 0
seen_codes = set()                        # codes already confirmed & printed
recent_codes = deque(maxlen=VOTE_WINDOW)   # codes found in each of the last N OCR passes

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lost connection to camera. Retrying...")
        cap.release()      # avoid leaking the old connection
        time.sleep(1)       # avoid hammering the camera in a tight loop
        cap = cv2.VideoCapture(CAM_RTSP, cv2.CAP_FFMPEG)
        continue

    # The camera might be 4K! Resize it so it fits on your screen (width 1280)
    # This stops it from looking "too zoomed in"
    height, width = frame.shape[:2]
    if width > 1280:
        scale = 1280 / width
        frame = cv2.resize(frame, (1280, int(height * scale)))

    # Convert to grayscale for much better and faster barcode reading
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Lightweight Enhancement Pipeline: Otsu's Binarization
    # This automatically finds the best threshold to turn blurry gray smudges into solid black and white
    _, enhanced_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Blank out the camera's timestamp and label overlays so OCR never
    # sees them (see comment above for why this matters).
    mask_region(enhanced_frame, TIMESTAMP_REGION)
    mask_region(enhanced_frame, CAMERA_LABEL_REGION)

    # Process every 10th frame to keep CPU usage low on older laptops
    frame_count += 1
    if frame_count % 10 == 0:
        try:
            # Extract English text/numbers from the enhanced black-and-white frame.
            # Restricting to digits helps since we only care about 7-digit codes.
            text = pytesseract.image_to_string(
                enhanced_frame,
                config='--psm 6 -c tessedit_char_whitelist=0123456789'
            )
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            text = ""

        # Use Regex to hunt for EXACTLY 7-digit numbers anywhere on the page
        pallet_numbers = re.findall(r'\b\d{7}\b', text)
        recent_codes.append(pallet_numbers)

        # Tally how many times each code appeared across the last
        # VOTE_WINDOW OCR passes, and only confirm ones that clear
        # VOTE_THRESHOLD — this filters out one-off misreads.
        vote_counts = Counter(code for codes in recent_codes for code in codes)
        confirmed_now = {code for code, count in vote_counts.items() if count >= VOTE_THRESHOLD}
        new_codes = confirmed_now - seen_codes

        if new_codes:
            for code in new_codes:
                print(f"✅ SUCCESS! FOUND PALLET NUMBER (OCR): {code}")
                print("=" * 50)
            seen_codes.update(new_codes)
        elif pallet_numbers:
            # Only print debug output when at least one 7-digit candidate
            # was actually found this pass — suppresses noise fragments
            # like "18" or "402" that aren't even the right length.
            print(f"👀 Candidate(s) this pass: {pallet_numbers}")

    # Show the live video feed (Original view)
    cv2.imshow("Camera View (Original)", frame)

    # Show the Enhanced view so the user can see how the computer fixes the blur
    cv2.imshow("AI Vision (Enhanced)", enhanced_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()