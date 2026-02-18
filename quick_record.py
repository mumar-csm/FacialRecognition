import cv2
import time

print("Recording 10-second test video...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend for better performance on Windows

# get camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, 20.0, (width, height))

start_time = time.time()
duration = 10  # seconds

print("Recording... move your face around!")
print("Recording will stop automatically after 10 seconds.")

while time.time() - start_time < duration:
    ret, frame = cap.read()
    if ret:
        # show live view
        remaining = int(duration - (time.time() - start_time))
        cv2.putText(frame, f"Recording... {remaining}s left",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        
        cv2.imshow('Recording (auto-stop in 10s)', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording finished. Saved as 'test_video.mp4'.")
print("Run: python recognize.py --mode video --source test_video.mp4 --display")