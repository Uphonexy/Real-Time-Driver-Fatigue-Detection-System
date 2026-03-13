import cv2
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
print("Opened:", cap.isOpened())
ret, frame = cap.read()
print("Frame read:", ret)
cap.release()
