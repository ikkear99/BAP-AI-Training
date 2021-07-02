import cv2
import  numpy as np

def findColor():
    img = cv2.imread(r'E:\DIEN_BAPSoftware\Bap_venv\image_opencv\virtual_painter\image.png') #default: BGR image
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    lower_green = np.array([30, 52, 72])
    upper_green = np.array([102, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_img, lower_green, upper_green)

    # Bitwise-AND mask and original image
    red = cv2.bitwise_and(img, img , mask= mask)

    x, y = getContours(mask)

    cv2.imshow('hi', red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y



def showcam():

    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        if ret:
            cv2.imshow("frame", frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    findColor()