'''
Created on May 25, 2013

@author: Luca
'''

import cv2
import sys
import numpy as np
import zbar
from VideoCapture import Device

class Resettable:
    def __init__(self, x, step = 1):
        self.start = x
        self.v = x
        self.step = step
        
    @property
    def v(self):
        return self.v
    
    @v.setter
    def v(self, x):
        self.v = x
        
    def set(self, x):
        if (x - self.start) % self.step == 0:
            self.v = x    

if __name__ == '__main__':
    out1_w = Resettable(9)
    out1_it = Resettable(2)
    out1_t = Resettable(100)
    out2_w = Resettable(5)
    
    cap = cv2.VideoCapture(0)
      
    if not cap.isOpened():
        print "dev not opened"
        sys.exit(-1)
        
    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')
    
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 768)
    print str(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)) + " " + str(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))    
            
#     wr = cv2.VideoWriter()
#     wr.open("d:\\tmp\\pippo.avi", cv2.cv.CV_FOURCC('i', 'Y', 'U', 'V'), cap.get(cv2.cv.CV_CAP_PROP_FPS), ((cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
        
    cv2.namedWindow("video")
    cv2.namedWindow("out1")
    cv2.namedWindow("out2")
    cv2.createTrackbar("out1_t", "video", out1_t.v, 255, lambda x: out1_t.set(x))
    cv2.createTrackbar("out2_w", "video", out2_w.v, 20, lambda x: out2_w.set(x))
    while True:
        
        cap.grab()
        success, frame = cap.retrieve()        
        
#         cv2.imwrite("d:\\tmp\\pippo.jpg", frame)
        
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_img = cv2.GaussianBlur(gray1, (5, 5), 3)
        gray = cv2.addWeighted(gray1, 1.5, out_img, -0.5, 0)
        out_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (out1_w.v, 1)), iterations=out1_it.v)
        t, out_img = cv2.threshold(out_img, out1_t.v, 255, cv2.THRESH_BINARY)
        out_img2 = cv2.morphologyEx(out_img,  cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (out2_w.v, 1)), iterations=1)
        out_img2 = cv2.morphologyEx(out_img2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (out2_w.v, out2_w.v)), iterations=2)
        out_img2 = cv2.dilate(out_img2, cv2.getStructuringElement(cv2.MORPH_RECT, (out2_w.v, out2_w.v)), iterations=1)
        
        cv2.imshow("out1", out_img2)
        
        w = 1024
        h = 768
        bCode_img = np.copy(gray)
        bCode_img = cv2.resize(bCode_img, (w, h))
        bCode_img.fill(0)
        x = y = 0
        
        contours, hierarchy = cv2.findContours(out_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours: 
            if cv2.contourArea(contour) > 2500:
                rect = cv2.minAreaRect(contour)
                rect = ((rect[0][0] + 5, rect[0][1] + 5), (rect[1][0], rect[1][1]), rect[2])
                box = np.int0(cv2.cv.BoxPoints(rect))
                cv2.drawContours(frame, [box], 0, (0, 255, 0), thickness=2) 
                
                bRect = cv2.boundingRect(contour)
                bRect = (bRect[0] + 5, bRect[1], bRect[2], bRect[3])
                
                #segment out the barcode
                angle = rect[2]
                size = (int(rect[1][0]), int(rect[1][1]))
                if angle < -45:
                    angle += 90
                    size = (int(rect[1][1]), int(rect[1][0]))
                    
                M = cv2.getRotationMatrix2D((bRect[2] / 2, bRect[3] / 2), angle, 1)
                bCode_roi = gray1[bRect[1]:bRect[1] + bRect[3], bRect[0]:bRect[0] + bRect[2]]
                bCode_roi = cv2.warpAffine(bCode_roi, M, (bRect[2], bRect[3]))
                
#                 laplace = cv2.convertScaleAbs(cv2.Laplacian(bCode_roi, -1))
#                 bCode_roi = cv2.subtract(bCode_roi, laplace)
                bCode_roi = cv2.resize(bCode_roi, (bCode_roi.shape[1] * 1.5, bCode_roi.shape[0] * 1.5))
#                 bCode_roi = cv2.equalizeHist(bCode_roi)
                out_img = cv2.GaussianBlur(bCode_roi, (3, 3), 1)
                bCode_roi  = cv2.addWeighted(bCode_roi, 1.5, out_img, 0, 0)
                #t, bCode_roi = cv2.threshold(bCode_roi, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 bCode_roi = cv2.adaptiveThreshold(bCode_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 1)
                bCode_roi = cv2.morphologyEx(bCode_roi, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=1)
#                 bCode_roi = cv2.morphologyEx(bCode_roi, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
                
                if x + bCode_roi.shape[1] > w:
                    x = 0
                    y += 100
                 
                bCode_img[y:y + bCode_roi.shape[0], x:x + bCode_roi.shape[1]] = bCode_roi
                x += bCode_roi.shape[1] + 10
                
                zImage = zbar.Image(bCode_roi.shape[1], bCode_roi.shape[0], 'Y800', bCode_roi.tostring())
                scanner.scan(zImage) 
                for symbol in zImage:
                    cv2.putText(frame, str(symbol.data), (bRect[0] + 10, bRect[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
#                     print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data   
         
        cv2.imshow("video", frame)
        cv2.imshow("out2", bCode_img)
        
        #wr.write(frame)
        
        if cv2.waitKey(30) >= 0: break
        
    cv2.destroyAllWindows()
#     wr.release()