import cv2
import re
import time
from pyzbar import pyzbar
import pytesseract
from collections import deque

# --- CONFIGURATION ---
CAM_IP = "10.1.2.200"
CAM_USER = "admin"
CAM_PASS = "megglass_1"
CAM_RTSP = f"rtsp://{CAM_USER}:{CAM_PASS}@{CAM_IP}:554/Streaming/Channels/101"

import os
# Force TCP to prevent h264 packet loss errors
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Tesseract path (Windows requirement)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def start_combined_scanner():
    print("\n" + "="*40)
    print("      SELECT CAMERA SOURCE")
    print("="*40)
    print("1: Laptop Webcam (Local testing)")
    print("2: Hikvision Camera (RTSP Stream)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '2':
        CAM_SOURCE = CAM_RTSP
        print(f"\nConnecting to Hikvision camera at {CAM_IP}...")
    else:
        CAM_SOURCE = 0
        print("\nConnecting to Laptop Webcam...")

    print("Starting combined scanner (PyZbar + OCR)...")
    cap = cv2.VideoCapture(CAM_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAM_SOURCE}.")
        return

    frame_count = 0
    
    # Memory buffers to store recent scans for cross-validation
    # We keep the last few scans to allow a slight delay between barcode and OCR detection
    recent_barcodes = deque(maxlen=20) 
    recent_ocr_texts = deque(maxlen=3) # OCR runs 10x slower, so smaller buffer
    
    # Keep track of fully validated numbers so we only print them once
    fully_validated = set()

    while True:
        success, frame = cap.read()
        if not success:
            continue
            
        frame_count += 1
        
        # ---------------------------------------------------------
        # 1. BARCODE SCANNER (Runs every frame for instant response)
        # ---------------------------------------------------------
        barcodes = pyzbar.decode(frame)
        
        for barcode in barcodes:
            code_data = barcode.data.decode('utf-8').strip()
            
            # Check if it's a 7-digit number
            if code_data.isdigit() and len(code_data) == 7:
                recent_barcodes.append(code_data)
                
                # Draw green box for barcode
                (x, y, w, h) = barcode.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Barcode: {code_data}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # ---------------------------------------------------------
        # 2. OCR TEXT SCANNER (Runs every 10 frames to save CPU)
        # ---------------------------------------------------------
        if frame_count % 10 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Otsu's binarization for better OCR reading
            _, enhanced_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            try:
                text = pytesseract.image_to_string(
                    enhanced_frame,
                    config='--psm 6 -c tessedit_char_whitelist=0123456789'
                )
                
                # Find all 7-digit numbers in the OCR text
                pallet_numbers = re.findall(r'\b\d{7}\b', text)
                for num in pallet_numbers:
                    recent_ocr_texts.append(num)
                    
            except Exception as e:
                pass # Ignore OCR errors temporarily
                
        # ---------------------------------------------------------
        # 3. CROSS-VALIDATION ENGINE
        # ---------------------------------------------------------
        # We check if any recent barcode matches any recent OCR read
        for b_code in list(recent_barcodes):
            if b_code in recent_ocr_texts:
                if b_code not in fully_validated:
                    fully_validated.add(b_code)
                    print("=" * 60)
                    print(f"✅✅ FULLY VALIDATED MATCH: {b_code} ✅✅")
                    print("=" * 60)
                    
                # We can draw an overarching "VALIDATED" status on the screen
                cv2.putText(frame, f"VALIDATED: {b_code}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # Show the camera feed
        cv2.imshow("Combined Cross-Validating Scanner", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_combined_scanner()
