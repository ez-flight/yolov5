import cv2

cap = cv2.VideoCapture(1)  # Try index 1 or 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read a frame from the camera.")
        break

    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
