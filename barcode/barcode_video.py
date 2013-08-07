'''
Created on May 25, 2013

@author: Luca
'''

import cv2
import sys
import numpy as np    
from BarcodeTools import BarcodeTools, BarcodeInfo, BarcodeFrame
import winsound

CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
FRAME_COUNT = 10

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
    
    barcodes = []
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
            curr_frame = []            
            barcodes_len = len(barcodes)
            
            for bcode_img, s_bcode_img, contour in zip(bcode_imgs, shocked_bcode_imgs, contours):                
                decoded1 = barcodeTools.decode(bcode_img)
                decoded2 = barcodeTools.decode(s_bcode_img)
                decoded = decoded2 if decoded2 != '' else decoded1
                
                #calcolo le feature del barcode corrente
                barcode_frame = BarcodeFrame(contour[0][0], contour[0][1], contour[2], bcode_img, decoded, contour)
                for c in contours:
                    if c != contour:
                        barcode_frame.add_context_point(c[0][0], c[0][1])
                curr_frame.append(barcode_frame)
                
                if barcodes_len == 0:
                    barcodes.append(BarcodeInfo(bcode_img, decoded, barcode_frame, contour)) 
                
                                
            #eseguo il il match col frame precedente e aggiorno la lista dei barcode
            if barcodes_len > 0:
                for b in curr_frame:
                    prev_match = min(((b.match_score(prev_b.barcode_frame), prev_b)) for prev_b in barcodes)
                    prev_match[1].add_update_candidate((prev_match[0], b))
                    
                for barcode in barcodes: 
                    new_barcodes = barcode.update()
                    for new_barcode in new_barcodes: 
                        barcodes.append(BarcodeInfo(new_barcode.bcode_img, new_barcode.decode_str, new_barcode, new_barcode.contour)) 
                        
                barcodes = [barcode for barcode in barcodes if barcode.to_remove < 10]
                
                # disegno il tutto
                for barcode in barcodes: 
                    box = np.int0(cv2.cv.BoxPoints(barcode.contour))
                    cv2.drawContours(frame, [box], 0, barcode.color, thickness = 2)
                     
                    bcode_img = barcode.bcode_img
                    if x + bcode_img.shape[1] > w:
                        x = 0
                        y += 100
                        extracted_bcodes[y:y + bcode_img.shape[0], x:x + bcode_img.shape[1]] = bcode_img
                        x += bcode_img.shape[1] + 10
                         
                    if barcode.decode_str != '':
                        cv2.putText(frame, barcode.decode_str, (barcode.contour[0][0], barcode.contour[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
                    
        cv2.imshow("original", frame)
        cv2.imshow("barcodes", extracted_bcodes)
        cv2.imshow("shocked barcodes", s_extracted_bcodes)
        if cv2.waitKey(30) == 27: break
        
    cv2.destroyAllWindows()