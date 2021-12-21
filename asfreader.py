import cv2

inputFile = '/Users/tharakarehan/Downloads/testhighway.asf'
vs = cv2.VideoCapture(inputFile)

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    cv2.imshow('pedestron',frame)
    # cv2.waitKey(10)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break
vs.release()