# test_cameras.py
import cv2

for i in range(4):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Index {i}: not available")
        continue
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"Camera {i} - press any key", frame)
        print(f"Index {i}: WORKING")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()