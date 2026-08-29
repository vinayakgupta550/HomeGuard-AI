import cv2
import os
import time

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Initialize the camera
cap = cv2.VideoCapture(0)

print("Warming up the Mac camera sensor...")
time.sleep(2) # Give the hardware 2 seconds to adjust exposure

# Flush the first 10 frames from the buffer
for _ in range(10):
    cap.read()

# Now capture the actual, well-lit frame
ret, frame = cap.read()

if ret:
    file_path = "data/test_frame.jpg"
    cv2.imwrite(file_path, frame)
    print(f"Success! Frame saved to {file_path}")
else:
    print("Error: Could not read a valid frame.")

# Release the camera hardware
cap.release()