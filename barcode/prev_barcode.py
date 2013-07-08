'''
Created on May 18, 2013

@author: Luca
'''

import os 
import cv2
import numpy as np
import time


if __name__ == '__main__':
    for dirname, dirnames, filenames in os.walk("d:\\barcode"):
        if dirname != 'd:\\barcode\\processed':            
            for filename in filenames:
            #dirname = 'd:\\barcode\jpg'
            #filename = '00052322.jpg'
                start = time.clock()
                                        
                img = cv2.cvtColor(cv2.imread(os.path.join(dirname, filename), cv2.CV_LOAD_IMAGE_COLOR), cv2.COLOR_BGR2GRAY)
                imgx = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (15,2))) #cv2.Sobel(img, -1, 1, 0)
                imgy = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (15,2)))  #cv2.Sobel(img, -1, 0, 1)
                out_img = cv2.subtract(imgx, imgy)
                #t, out_img = cv2.threshold(out_img, 120, 255, cv2.THRESH_BINARY)
                
                #out_img = cv2.erode(out_img, cv2.getStructuringElement(cv2.MORPH_RECT, (1,7)), iterations=1)
                #out_img = cv2.dilate(out_img, cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)), iterations=6)
                #out_img = cv2.GaussianBlur(out_img, (11, 11), 1)
                #t, out_img = cv2.threshold(out_img, 80, 255, cv2.THRESH_BINARY)
               # out_img = cv2.erode(out_img, cv2.getStructuringElement(cv2.MORPH_RECT, (30,1)), iterations=2)
                #out_img = cv2.bitwise_and(img, out_img)
                
                end = time.clock()
                
                outname = os.path.join('d:\\barcode', 'processed', filename)
                cv2.imwrite(outname, out_img)
                print "{0}: time {1}".format(outname, end - start)