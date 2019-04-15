from detection import detection
import cv2
import numpy as np
import time

count = 0
det = detection()
#cap = cv2.VideoCapture(0)
#time.sleep(1)
while True:
    dst, zeropad, bound, frame = det.object_detection_returnframe()
    #cv2.imshow("aa", frame)
    if np.any(dst):
        #cv2.imshow("frame", frame)
        cv2.imwrite('data/zeropad/{}.jpg'.format(count), zeropad)
        cv2.imwrite('data/frame/{}.jpg'.format(count), frame)
        print("書き込み成功" + str(count))
        det.set_mode()
        #cv2.imshow("ab", dst)
        #time.sleep(1)
        count += 1

    k = cv2.waitKey(1)
    if k == 13:
        det.cap_release()
        break