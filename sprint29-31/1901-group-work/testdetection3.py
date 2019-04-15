from detection import detection
import cv2

det = detection()

while True:
    dst = det.object_detection()
    det._test_camera()
    
    k = cv2.waitKey(1)
    if k == 13:
        break