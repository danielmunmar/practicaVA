import os
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from detector import RoadPanelDetector

class PanelDetectionPipeline:
    def __init__(self, train_path=None, test_path=None, detector_name=""):
            """
            Initialize the pipeline.
            
            Args:
                train_path: Path to training directory
                test_path: Path to test directory
                detector_name: Name of the detector to use
            """
            self.train_path = train_path
            self.test_path = test_path
            self.detector_name = detector_name
            self.detector = None
            self.total_detections = 0
            self.detections_per_image = {}

    def validate_inputs(self):
            """Validates that input paths are correct."""
            if not self.test_path or not os.path.isdir(self.test_path):
                print("ERROR: --test_path is not valid or not specified")
                return False
            
            if self.train_path and not os.path.isdir(self.train_path):
                print(f"WARNING: train_path is invalid: {self.train_path}")
                self.train_path = None
            
            return True

    def load_ground_truth(self, gt_path):
            """
            Loads annotations from a gt.txt file.
            
            Args:
                gt_path: Path to the gt.txt file
                
            Returns:
                dict: {img_name: [(x1, y1, x2, y2, class, score), ...]}
            """
            bboxes = {}
            
            if not os.path.exists(gt_path):
                print(f"WARNING: File not found: {gt_path}")
                return bboxes
            
            try:
                with open(gt_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split(';')
                        if len(parts) < 5:
                            continue
                        
                        img_name = parts[0]
                        try:
                            x1, y1, x2, y2 = map(int, parts[1:5])
                            class_id = int(parts[5]) if len(parts) > 5 else 1
                            score = float(parts[6]) if len(parts) > 6 else 1.0
                            
                            if img_name not in bboxes:
                                bboxes[img_name] = []
                            
                            bboxes[img_name].append((x1, y1, x2, y2, class_id, score))
                        except (ValueError, IndexError):
                            print(f"WARNING: Malformed line: {line}")
                            continue
            
            except Exception as e:
                print(f"ERROR reading GT: {e}")
            
            return bboxes

    def analyze_ground_truth(self, bboxes):
            """
            Analyzes statistics of annotated panels.
            Useful for adjusting detector parameters.
            
            Args:
                bboxes: dict of annotations
            """
            if not bboxes:
                return
            
            aspect_ratios = []
            areas = []
            widths = []
            heights = []
            total_panels = 0
            
            for img_name, boxes in bboxes.items():
                for x1, y1, x2, y2, _, _ in boxes:
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w > 0 and h > 0:
                        aspect_ratios.append(w / h)
                        areas.append(w * h)
                        widths.append(w)
                        heights.append(h)
                        total_panels += 1
            
            if not aspect_ratios:
                return
            
            # Display analysis
            print("\n" + "=" * 70)
            print("STATISTICAL ANALYSIS OF PANELS IN TRAINING")
            print("=" * 70)
            
            print(f"\nQuantity:")
            print(f"   Total images: {len(bboxes)}")
            print(f"   Total panels: {total_panels}")
            print(f"   Average per image: {total_panels / len(bboxes):.1f}")
            
            print(f"\nAspect Ratio (Width/Height):")
            print(f"   Mean: {np.mean(aspect_ratios):.3f}")
            print(f"   Std Dev: {np.std(aspect_ratios):.3f}")
            print(f"   Range: [{np.min(aspect_ratios):.3f}, {np.max(aspect_ratios):.3f}]")
            print(f"   Percentiles [25%, 50%, 75%]: {np.percentile(aspect_ratios, [25, 50, 75])}")
            
            print(f"\nArea (pixels²):")
            print(f"   Mean: {np.mean(areas):.0f}")
            print(f"   Range: [{np.min(areas):.0f}, {np.max(areas):.0f}]")
            print(f"   Percentiles [25%, 50%, 75%]: {np.percentile(areas, [25, 50, 75])}")
            
            print(f"\nWidth (pixels):")
            print(f"   Mean: {np.mean(widths):.0f}")
            print(f"   Range: [{np.min(widths):.0f}, {np.max(widths):.0f}]")
            
            print(f"\nHeight (pixels):")
            print(f"   Mean: {np.mean(heights):.0f}")
            print(f"   Range: [{np.min(heights):.0f}, {np.max(heights):.0f}]")
            
            print("=" * 70 + "\n")

    def create_detector(self):
            """Creates and initializes the detector."""
            print("Initializing detector...")
            self.detector = RoadPanelDetector()
            print("Detector ready\n")
        
    def setup_output_directories(self):
            """Creates necessary output directories."""
            os.makedirs("resultado_imgs", exist_ok=True)
            print("Output directory created: resultado_imgs/")
        
    def get_test_images(self):
            """Gets list of PNG images in the test directory."""
            images = sorted([
                f for f in os.listdir(self.test_path)
                if f.lower().endswith('.png')
            ])
            
            if not images:
                print(f"WARNING: No .png images found in {self.test_path}")
            
            return images

    def process_image(self, img_name, result_file):
            """
            Processes an individual image:
            - Reads image
            - Detects panels
            - Writes results
            - Draws visualization
            
            Args:
                img_name: Name of image file
                result_file: Opened file to write results
                
            Returns:
                int: Number of detections in this image
            """
            try:
                img_path = os.path.join(self.test_path, img_name)
                image = cv2.imread(img_path)
                
                if image is None:
                    raise IOError(f"Could not read image: {img_path}")
                
                # Detect panels
                detections = self.detector.detect(image)
                
                if detections is None:
                    detections = []
                
                # Process each detection
                num_detections = 0
                
                for det in detections:
                    try:
                        x1, y1, x2, y2, score = det
                        
                        # Explicit conversion to native Python types
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        score = float(score)
                        
                        # Validate coordinates are valid
                        if x1 >= x2 or y1 >= y2:
                            print(f"   WARNING: Invalid coordinates: ({x1},{y1}) -> ({x2},{y2})")
                            continue
                        
                        # Limit to image dimensions
                        x1 = max(0, min(x1, image.shape[1] - 1))
                        y1 = max(0, min(y1, image.shape[0] - 1))
                        x2 = max(x1 + 1, min(x2, image.shape[1]))
                        y2 = max(y1 + 1, min(y2, image.shape[0]))
                        
                        # Write to results file (standard format)
                        line = f"{img_name};{x1};{y1};{x2};{y2};1;{score:.4f}\n"
                        result_file.write(line)
                        
                        # Draw rectangle (red)
                        cv2.rectangle(
                            image,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),  # BGR: Red
                            2
                        )
                        
                        # Draw score (yellow)
                        text_size = cv2.getTextSize(
                            f"{score:.2f}",
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            2
                        )[0]
                        
                        text_x = x1
                        text_y = max(15, y1 - 5)
                        
                        # Semi-transparent background for text
                        overlay = image.copy()
                        cv2.rectangle(
                            overlay,
                            (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + 2),
                            (0, 0, 0),
                            -1
                        )
                        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
                        
                        # Yellow text
                        cv2.putText(
                            image,
                            f"{score:.2f}",
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),  # BGR: Yellow
                            2
                        )
                        
                        num_detections += 1
                    
                    except Exception as e:
                        print(f"   WARNING: Error processing detection: {e}")
                        continue
                
                # Save image with visualization
                output_path = os.path.join("resultado_imgs", img_name)
                cv2.imwrite(output_path, image)
                
                return num_detections
            
            except Exception as e:
                print(f"   ERROR: {e}")
                return 0

    def print_summary(self, test_images):
            """Prints a summary of the results."""
            images_with_detections = sum(1 for count in self.detections_per_image.values() if count > 0)
            images_without_detections = len(test_images) - images_with_detections
            
            print("=" * 70)
            print("DETECTION COMPLETED SUCCESSFULLY")
            print("=" * 70)
            
            print(f"\nSTATISTICS:")
            print(f"   Total images processed: {len(test_images)}")
            print(f"   Total panels detected: {self.total_detections}")
            print(f"   Average per image: {self.total_detections / len(test_images):.2f}")
            print(f"   Images with detections: {images_with_detections}")
            print(f"   Images without detections: {images_without_detections}")
            
            print(f"\nOUTPUT FILES:")
            print(f"   resultado.txt (standard detection format)")
            print(f"   resultado_imgs/ (images with drawn boxes)")
            
            print(f"\nNEXT STEPS:")
            print(f"   1. Run evaluation: python evaluar_resultados.py --test_path {self.test_path}")
            print(f"   2. Review precision-recall curve")
            print(f"   3. Adjust parameters if necessary")
            print(f"\n" + "=" * 70)

    def run(self):
            """Runs the complete detection pipeline."""
            
            # Validate inputs
            if not self.validate_inputs():
                return False
            
            print(f"Test directory: {self.test_path}")

            # Load training data and analyze if available
            if self.train_path:
                print(f"Training directory: {self.train_path}")
                train_gt = os.path.join(self.train_path, "gt.txt")
                train_bboxes = self.load_ground_truth(train_gt)
                self.analyze_ground_truth(train_bboxes)

            # Create the detector
            self.create_detector()
            
            # Setup output directories
            self.setup_output_directories()
            
            # Load testing data
            test_images = self.get_test_images()
            
            if not test_images:
                print("ERROR: No images to process")
                return False
            
            # Evaluate detections
            print(f"Processing {len(test_images)} test images...")
            print("=" * 70)
            
            with open("resultado.txt", "w") as result_file:
                for idx, img_name in enumerate(test_images):
                    progress = f"[{idx + 1:3d}/{len(test_images)}]"
                    num_det = self.process_image(img_name, result_file)
                    self.detections_per_image[img_name] = num_det
                    self.total_detections += num_det
                    
                    # Progress information
                    status = "OK" if num_det > 0 else "NO"
                    print(f"{status} {progress} {img_name:20s} -> {num_det} panels detected")
            
            # Print summary
            self.print_summary(test_images)
            
            return True

def main():
    """Main function."""
        
    # Command line arguments
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')
    args = parser.parse_args()
        
    # Create and run pipeline (init)
    pipeline = PanelDetectionPipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        detector_name=args.detector
    )
        
    success = pipeline.run()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
