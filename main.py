import os
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

from detector import RoadPanelDetector



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
    
    # Create the detector

    detector = RoadPanelDetector()


    # Load testing data

    os.makedirs("resultado_imgs", exist_ok=True)

    result_file = open("resultado.txt", "w")

    test_images = sorted([
        f for f in os.listdir(args.test_path)
        if f.endswith(".png")
    ])

    # Evaluate detections

    for img_name in test_images:

        img_path = os.path.join(args.test_path, img_name)
        image = cv2.imread(img_path)

        detections = detector.detect(image)
        for det in detections:

            x1, y1, x2, y2, score = det

            # Guardar txt
            line = f"{img_name};{x1};{y1};{x2};{y2};1;{score:.4f}\n"

            result_file.write(line)

            # Dibujar
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                image,
                f"{score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        cv2.imwrite(
            os.path.join("resultado_imgs", img_name),
            image
        )

    result_file.close()

    print("Detección finalizada")





