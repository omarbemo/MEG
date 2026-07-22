import cv2
from pyzbar import pyzbar

# Connect to the laptop camera (Index 0 or 1 depending on your setup)
# If this opens your external camera, change this to 1.
CAM_SOURCE = 0 

print("Starting laptop camera for barcode scanning...")
cap = cv2.VideoCapture(CAM_SOURCE)

# Memory to prevent spamming the console with the same barcode
last_scanned = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Decode barcodes in the current frame
    barcodes = pyzbar.decode(frame)
    
    for barcode in barcodes:
        # Extract the bounding box location of the barcode and draw a rectangle
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Decode the barcode data to a readable string
        code_data = barcode.data.decode('utf-8').strip()
        code_type = barcode.type
        
        # FILTER: Only process barcodes that are exactly 7 digits long (numbers only)
        if not (code_data.isdigit() and len(code_data) == 7):
            continue
        
        # Display the barcode data and type on the screen
        text = f"{code_data} ({code_type})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Print to console only if it's a new scan (de-duplication)
        if code_data != last_scanned:
            print(f">>> SCANNED: {code_data} | TYPE: {code_type}")
            last_scanned = code_data

    # Show the camera feed directly on the screen
    cv2.imshow("SmartPallet - Local Barcode Scanner", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up when done
cap.release()
cv2.destroyAllWindows()