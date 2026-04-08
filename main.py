import os
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

def non_max_suppression(boxes, overlapThresh=0.5):
    """
    boxes: lista de tuplas (x, y, w, h)
    overlapThresh: umbral de solapamiento (0–1)
    """
    if len(boxes) == 0:
        return []

    # Convertir a array numpy
    boxes = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]

    # Área de cada caja
    area = boxes[:,2] * boxes[:,3]
    # Ordenar por la coordenada inferior derecha
    idxs = np.argsort(y2)

    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        idxs = idxs[:-1]

        xx1 = np.maximum(x1[last], x1[idxs])
        yy1 = np.maximum(y1[last], y1[idxs])
        xx2 = np.minimum(x2[last], x2[idxs])
        yy2 = np.minimum(y2[last], y2[idxs])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Intersección sobre unión (IoU)
        overlap = (w * h) / area[idxs]

        # Mantener solo cajas con poco solapamiento
        idxs = idxs[overlap <= overlapThresh]

    return boxes[pick].astype(int).tolist()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    # Load training data
    img = cv2.imread("train_detection/00000.png")
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Create the detector

    mser = cv2.MSER_create(
        min_area=300,
        max_area=8000
    )

    regions, _ = mser.detectRegions(gray)

    boxes = []

    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        boxes.append((x, y, w, h))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    filtered_boxes = []

    for (x, y, w, h) in boxes:
        ratio = w / h
        
        if 0.5 < ratio < 2.5: 
            filtered_boxes.append((x, y, w, h))
    
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_ideal = np.ones((40, 80), dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    # máscara de azul adaptativa
    mask_blue = cv2.inRange(h, 100, 140)
    mask_s = cv2.inRange(s, 50, 255)
    mask_v = cv2.inRange(v, 50, 255)

    mask = cv2.bitwise_and(mask_blue, mask_s)
    mask = cv2.bitwise_and(mask, mask_v)
    mask_binary = (mask > 0).astype(np.uint8)

    masks_resized = []

    resized_rois = []

    for (x, y, w, h) in filtered_boxes:
        roi = img[y:y+h, x:x+w]
        
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(roi_hsv, lower_blue, upper_blue)
        
        # binarizar 0/1
        mask_bin = (mask > 0).astype(np.uint8)
        
        # redimensionar a 40x80
        mask_resized = cv2.resize(mask_bin, (80, 40))
        
        resized_rois.append(mask_resized)

    
    correlation_values = []

    for mask_roi in resized_rois:
        corr = np.sum(mask_roi * mask_ideal) / np.sum(mask_ideal)
        correlation_values.append(corr)
        
    threshold = 0.5

    final_panel_boxes = []

    for i, corr in enumerate(correlation_values):
        if corr > threshold:
            final_panel_boxes.append(filtered_boxes[i])

    for (x, y, w, h) in final_panel_boxes:
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    final_boxes_nms = non_max_suppression(final_panel_boxes, overlapThresh=0.4)
    

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for (x, y, w, h) in final_boxes_nms:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    detections = []

    img_name = "00000.png"

    for (x, y, w, h) in final_boxes_nms:
        left = x
        top = y
        right = x + w
        bottom = y + h
        
        class_id = 1
        score = 1.0
        
        detections.append((img_name, left, top, right, bottom, class_id, score))
    
    for i, (x, y, w, h) in enumerate(filtered_boxes):
        if correlation_values[i] > threshold:
            score = correlation_values[i]
            
            detections.append((img_name, x, y, x+w, y+h, 1, score))
    
    ## Save in file .txt (format required)
    ## Evaluator uses CSV with ;
    """ with open("resultado.txt", "w") as f:
        for det in detections:
            f.write(f"{det[0]};{det[1]};{det[2]};{det[3]};{det[4]};{det[5]};{det[6]}\n") """

    plt.imshow(img_rgb)
    plt.title("Detecciones finales por correlación")
    plt.axis('off')
    plt.show()

    # Load testing data

    # Evaluate detections





