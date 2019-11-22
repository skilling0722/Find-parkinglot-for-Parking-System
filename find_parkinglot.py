import numpy as np
import cv2 as cv
from glob import glob

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def hsv_mask(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    sensitivity = 45
    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    mask = cv.inRange(hsv, lower_white, upper_white)

    res = cv.bitwise_and(img, img, mask=mask)

    return res

def find_squares(img):
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            contours, _ = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)

                if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def createblankimage(img):
    width = np.size(img, 0)
    height = np.size(img, 1)

    blankimage = np.zeros((width, height, 3), np.uint8)
    return blankimage

if __name__ == '__main__':
    for fn in glob('../sample.jpg'):
        #이미지 로드
        img = cv.imread(fn)
        cv.imshow('init', img)
        #흰색 성분 검출
        img = hsv_mask(img)
        #사각형 검출
        squares = find_squares(img)
        #빈 이미지 생성
        outputimage = createblankimage(img)
        #빈 이미지에 검출된 사각형 그리기
        cv.drawContours(outputimage, squares, -1, (254, 154, 46), 3)
        #이미지 띄우기
        cv.imshow('squares', outputimage)
        #이미지 저장
        cv.imwrite('../result.png', outputimage)
        cv.waitKey(0)
        cv.destroyAllWindows()
