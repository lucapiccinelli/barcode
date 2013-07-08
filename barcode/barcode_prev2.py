'''
Created on May 18, 2013

@author: Luca
'''

import os 
import cv2
import numpy as np
import time

import zbar

# if __name__ == '__main__':
#     for dirname, dirnames, filenames in os.walk("d:\\barcode"):
#         if dirname != 'd:\\barcode\\processed':            
#             for filename in filenames:
#             #dirname = 'd:\\barcode\jpg'
#             #filename = '00052322.jpg'
#                                         
#                 start = time.clock()
#                 
#                 img = cv2.imread(os.path.join(dirname, filename), cv2.CV_LOAD_IMAGE_COLOR)
#                 img = cv2.resize(img, (860, int( (860.0 / img.shape[1]) * img.shape[0] )), interpolation=cv2.INTER_LINEAR)
#                  
#                 gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 out_img = cv2.GaussianBlur(gray_img, (3, 3), 3)
#                 out_img = cv2.addWeighted(gray_img, 1.5, out_img, -0.5, 0)
#                 out_img = cv2.morphologyEx(out_img, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1)), iterations=2)
#                 out_img = cv2.morphologyEx(out_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=1)
#                 t, out_img = cv2.threshold(out_img, 90, 255, cv2.THRESH_BINARY)
#                 out_img = cv2.morphologyEx(out_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=2)
#                 out_img = cv2.morphologyEx(out_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=2)
#                 out_img = cv2.erode(out_img, cv2.getStructuringElement(cv2.MORPH_RECT, (5,1)), iterations=2)
#                 out_img = cv2.morphologyEx(out_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=3)
#                 out_img = cv2.morphologyEx(out_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=2)
#                 out_img = cv2.morphologyEx(out_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)), iterations=2)
# #                 out_img = cv2.morphologyEx(out_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)), iterations=2)
#                 out_img = cv2.dilate(out_img, cv2.getStructuringElement(cv2.MORPH_RECT, (5,1)), iterations=2)
# #                    
#                 contours, hierarchy = cv2.findContours(out_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#                 cv2.drawContours(img, contours, -1, (0, 255, 0), thickness=2, offset=(35, 6))
#                 
#                 end = time.clock()
#                 
#                 outname = os.path.join('d:\\barcode', 'processed', filename)
#                 cv2.imwrite(outname, img)
#                 print "{0}: time {1}".format(outname, end - start)


if __name__ == '__main__':
    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')
    
    for dirname, dirnames, filenames in os.walk("d:\\barcode"):
        if dirname != 'd:\\barcode\\processed':            
            for filename in filenames:
            #dirname = 'd:\\barcode\jpg'
            #filename = '00052322.jpg'
                start = time.clock()
                                         
                img = cv2.imread(os.path.join(dirname, filename), cv2.CV_LOAD_IMAGE_COLOR)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #out_img = cv2.GaussianBlur(img, (7, 7), 3)
                #out_img = cv2.addWeighted(img, 1.5, out_img, -0.5, 0)
                out_img = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)), iterations=2)
                out_img = cv2.morphologyEx(out_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1)), iterations=1)
                out_img = cv2.morphologyEx(out_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20)), iterations=1)
                t, out_img = cv2.threshold(out_img, 110, 255, cv2.THRESH_BINARY)
                out_img = cv2.morphologyEx(out_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)), iterations=2)
                out_img = cv2.dilate(out_img, cv2.getStructuringElement(cv2.MORPH_RECT, (10,3)), iterations=2)
                  
                contours, hierarchy = cv2.findContours(out_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                out_contours = []
                for i, contour in enumerate(contours):
                    if cv2.contourArea(contour) > 5000:
                        out_contours.append(contour)
                        bRect = cv2.boundingRect(contour)
                        bRect = (bRect[0] + 20, bRect[1], bRect[2], bRect[3])
                        
                        rect = cv2.minAreaRect(contour)
                        rect = ((rect[0][0] + 20, rect[0][1]), (rect[1][0], rect[1][1]), rect[2])
                        rect2 = ((bRect[0] + bRect[2] / 2, bRect[1] + bRect[3] / 2), (bRect[2], bRect[3]), 0)
                        box = np.int0(cv2.cv.BoxPoints(rect))
                        
                        #crop the barcode adjusting the rotation
                        
                        angle = rect[2]
                        size = (int(rect[1][0]), int(rect[1][1]))
                        if angle < -45:
                            angle += 90
                            size = (int(rect[1][1]), int(rect[1][0]))
                        
                        M = cv2.getRotationMatrix2D((bRect[2] / 2, bRect[3] / 2), angle, 1)
                        out_img = gray_img[bRect[1]:bRect[1] + bRect[3], bRect[0]:bRect[0] + bRect[2]]
                        out_img = cv2.warpAffine(out_img, M, (bRect[2], bRect[3]))
                        #out_img = cv2.equalizeHist(out_img)
                        #out_img = cv2.Sobel(out_img, -1, 1, 0)
                        #t, out_img = cv2.threshold(out_img, 100, 255, cv2.THRESH_BINARY)
                        
                        zImage = zbar.Image(bRect[2], bRect[3], 'Y800', out_img.tostring())
                        scanner.scan(zImage)
                        cv2.drawContours(img, [box], 0, (0, 255, 0), thickness=3) 
                        for symbol in zImage:
#                             print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data
                            cv2.putText(img, str(symbol.data), (bRect[0] + 10, bRect[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
                         
#                         outname = os.path.join('d:\\barcode', 'processed', str(i) + '_' + filename)
#                         cv2.imwrite(outname, out_img)
#                         
                        
                        
                        
                        
#                 cv2.drawContours(img, out_contours, -1, (0, 255, 0), thickness=3, offset=(20, 0))
#                 for contour in contours:
#                     rect = cv2.boundingRect(contour)
#                     cv2.rectangle(img, (rect[0] + 20, rect[1]), (rect[0] + rect[2] + 20, rect[1] + rect[3]), (0, 255, 0), 3)
                 
                end = time.clock()
                 
                outname = os.path.join('d:\\barcode', 'processed', filename)
                cv2.imwrite(outname, img)
                print "{0}: time {1}".format(outname, end - start)                