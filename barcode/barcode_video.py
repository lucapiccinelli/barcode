'''
Created on May 25, 2013

@author: Luca
'''

import cv2
import sys
import numpy as np    
from BarcodeTools import BarcodeTools, BarcodeInfo
import winsound

CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
FRAME_COUNT = 1

if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print "dev not opened"
        sys.exit(-1)
        
    barcodeTools = BarcodeTools(True) 
    
    w = 1024
    h = 768
    i = 0
    
    
    bcodes_info = {}
    frames_with_same_len = 0
    frames_without_barcodes = 0
    image_decoded = False
    decode = True
    while True:
        x = y = 0
        extracted_bcodes = np.zeros((h, w), np.uint8)
        s_extracted_bcodes = np.zeros((h, w), np.uint8)    
        
        if i == FRAME_COUNT:
            cap.set(CV_CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080)
            
        i += 1
        
        cap.grab()
        success, frame = cap.retrieve()
        
        if i > FRAME_COUNT and not image_decoded:
            prev_info_len = len(bcodes_info)
            
            contours, bcode_imgs, shocked_bcode_imgs, center = barcodeTools.detect(frame)
            bcodes_num = len(contours)
            
            if bcodes_num == 0:
                frames_without_barcodes += 1
            else:
                frames_without_barcodes = 0
            
            if frames_without_barcodes > 10: decode = True
            
            if frames_without_barcodes < 20 and decode:
                for bcode_img, s_bcode_img, contour in zip(bcode_imgs, shocked_bcode_imgs, contours):
                    box = np.int0(cv2.cv.BoxPoints(contour))
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), thickness = 2)
                    
                    if x + bcode_img.shape[1] > w:
                        x = 0
                        y += 100
                    extracted_bcodes[y:y + bcode_img.shape[0], x:x + bcode_img.shape[1]] = bcode_img
                    s_extracted_bcodes[y:y + bcode_img.shape[0], x:x + bcode_img.shape[1]] = s_bcode_img
                    x += bcode_img.shape[1] + 10

                    
                    decoded1 = barcodeTools.decode(bcode_img)
                    decoded2 = barcodeTools.decode(s_bcode_img)
                    
                    decoded = decoded2 if decoded2 != '' else decoded1
                    if decoded != '':
                        cv2.putText(frame, decoded1, (contour[0][0], contour[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
                        cv2.putText(frame, decoded2, (contour[0][0], contour[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    
                        
                        if decoded not in bcodes_info:
                            bcodes_info[decoded] = BarcodeInfo(bcode_img, decoded)
                           
                info_len = len(bcodes_info) 
                if bcodes_num == 0: 
                    frames_with_same_len = 0
                    
                if info_len == prev_info_len and info_len != 0 and info_len == bcodes_num:
                    frames_with_same_len += 1
                else:
                    frames_with_same_len = 0
                    if info_len > bcodes_num and bcodes_num != 0: 
                        bcodes_info = {}
                
                cv2.putText(frame, 'bcodes num:' + str(bcodes_num), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
                cv2.putText(frame, 'len:' + str(info_len), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
                cv2.putText(frame, 'same len:' + str(frames_with_same_len), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
                cv2.putText(frame, '0 len:' + str(frames_without_barcodes), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
                if frames_with_same_len == 5:
                    for k,v in bcodes_info.iteritems():
                        print k
                    print '\n'
                    winsound.Beep(2500, 300)
                    #image_decoded = True
                    frames_with_same_len = 0
                    bcodes_info = {}
                    decode = False
        
        cv2.imshow("original", frame)
        cv2.imshow("barcodes", extracted_bcodes)
        cv2.imshow("shocked barcodes", s_extracted_bcodes)
        if cv2.waitKey(30) == 27: break
        
    cv2.destroyAllWindows()