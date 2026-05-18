import cv2
import numpy as np


class RoadPanelDetector:
    """
    Detects road information panels using:
    1. MSER (Maximally Stable Extremal Regions) for region detection
    2. HSV + LAB + Houghes for similarity with panel score
    3. NMS for duplicate elimination
    """
    
    def __init__(self):
        """Initialize the detector with optimized parameters."""
        
        # MSER parameters - tuned for highway panel detection
        # min_area: minimum region size in pixels
        # max_area: maximum region size in pixels
        # delta: threshold increase between MSER calculations
        self.mser = cv2.MSER_create(
            min_area=300,
            max_area=80000,
            delta=10
        )

        # Standard size for blue score computation
        self.std_w = 40
        self.std_h = 80

    def detect(self, image):
        """
        Main detection pipeline.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of detections: [[x1, y1, x2, y2, score], ...]
        """
        detections = []

        # Step 1: Preprocess image to improve quality
        image = self.preprocess_image(image)

        # Step 2: Convert to grayscale for MSER
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(8, 8)
        )
        gray = clahe.apply(gray)

        # Step 4: Detect high-contrast regions with MSER
        regions, _ = self.mser.detectRegions(gray)

        # Step 5: Convert regions to bounding boxes and filter
        candidate_boxes = []

        for region in regions:
            x, y, w, h = cv2.boundingRect(region)

            # Filter by minimum size
            if w < 30 or h < 20:
                continue

            # Filter by aspect ratio (panels are wider than tall)
            aspect_ratio = w / float(h)

            # Adjusted range based on real panel proportions
            if aspect_ratio < 0.8 or aspect_ratio > 6.5:
                continue

            # Expand bounding box to include white border
            # MSER detects the blue interior, we need the whole panel
            pad_x = int(w * 0.025)
            pad_y = int(h * 0.05)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + w + pad_x)
            y2 = min(image.shape[0], y + h + pad_y)

            candidate_boxes.append((x1, y1, x2, y2))

        # Step 6: Score each candidate box
        for box in candidate_boxes:
            x1, y1, x2, y2 = box

            # Extract region from image
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Compute score
            score = self.compute_score(crop)

            if score > 0.3:
                detections.append([x1, y1, x2, y2, score])

        # Step 7: Remove duplicate detections (NMS)
        detections = self.non_max_suppression(detections)

        return detections

    def compute_score(self, crop):
        """
        Computes a score representing how much the crop looks like a road panel.
        
        Strategy:
        1. Use HSV to detect blue
        2. Use LAB to verify white color
        3. Use Hough lines to detect panel line borders
        4. Compute final score
        
        Args:
            crop: Image region to analyze
            
        Returns:
            float: Score between 0 (not similar to a panel) and 1 (panel)
        """
        
        # Resize to standard size for consistent scoring
        resized = cv2.resize(crop, (self.std_w, self.std_h))

        # 1. BLUE DETECTION
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Blue range in HSV (tight to avoid false positives)
        lower_blue = np.array([100, 130, 60])
        upper_blue = np.array([125, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Clean small holes and smooth regions
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.GaussianBlur(blue_mask, (7, 7), 0)

        # Normalize to [0,1]
        blue = blue_mask.astype(np.float32) / 255.0

        # Ideal blue mask
        ideal = np.zeros((self.std_h, self.std_w), dtype=np.float32)
        ideal[3:-3, 3:-3] = 1.0
        ideal[12:-12, 12:-12] = 0.7
        ideal[20:-20, 20:-20] = 0.4

        blue_score = np.sum(blue * ideal) / (np.sum(ideal) + 1e-6)

        # 2. WHITE DETECTION (LAB)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        white = (
            (L > 170) &
            (np.abs(A - 128) < 18) &
            (np.abs(B - 128) < 18)
        ).astype(np.uint8)

        white_ratio = np.mean(white)

        # Reject if too much white (likely text/signs)
        if white_ratio > 0.3:
            return 0.0

        # Expect white border region
        border = np.zeros_like(white)

        t = 3
        border[:t, :] = 1
        border[-t:, :] = 1
        border[:, :t] = 1
        border[:, -t:] = 1

        white_border_score = np.sum(white * border) / (np.sum(border) + 1e-6)

        # 3. HOUGH LINES SCORE
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)

        # Detect line segments
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=25,
            minLineLength=10,
            maxLineGap=5
        )

        h_lines = 0
        v_lines = 0
        total_len = 0

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]

                # Line length contribution
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_len += length

                # Separate horizontal vs vertical structure
                if abs(x2 - x1) > abs(y2 - y1):
                    h_lines += length
                else:
                    v_lines += length

        # Normalize total amount of structure
        line_score = min(1.0, total_len / 80.0)

        # Penalize unstructured / noisy line distribution
        if (h_lines + v_lines) > 0:
            balance = abs(h_lines - v_lines) / (h_lines + v_lines)
        else:
            balance = 1.0

        # Prefer balanced rectangular structure
        structure_score = line_score * (1.0 - balance)

        # 4. FINAL SCORE
        score = (
            0.6 * blue_score +
            0.2 * white_border_score +
            0.2 * structure_score
        )

        return float(score)

    def non_max_suppression(self, detections):
        """
        Perform Non-Maximum Suppression (NMS) to remove redundant detections.

        Removes duplicate detections by keeping only the best one (biggest with best score) 
        from each group of overlapping boxes.
        
        Args:
            detections: List of [x1, y1, x2, y2, score]
            
        Returns:
            List of non-overlapping detections
        """
        if len(detections) == 0:
            return []

        # Order by score and size (score*size^0.2)
        detections = sorted(
            detections, 
            key=lambda d: d[4] * ( (d[2]-d[0]) * (d[3]-d[1]) )**0.2, 
            reverse=True)

        final = []

        # Greedy selection of the best one
        for det in detections:

            x1, y1, x2, y2, s1 = det
            area1 = (x2 - x1) * (y2 - y1)

            keep = True

            for f in final:
                fx1, fy1, fx2, fy2, s2 = f

                # 1. IoU
                iou = self.compute_iou(det, f)

                # 2. Containment
                inter_x1 = max(x1, fx1)
                inter_y1 = max(y1, fy1)
                inter_x2 = min(x2, fx2)
                inter_y2 = min(y2, fy2)

                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                area_small = min(area1, (fx2 - fx1) * (fy2 - fy1))

                containment = inter_area / (area_small + 1e-6)

                # Decision
                if iou > 0.15 or containment > 0.7:
                    keep = False
                    break

            if keep:
                final.append(det)

        return final

    def compute_iou(self, boxA, boxB):
        """
        Compute Intersection over Union between two boxes.
        
        IoU = Intersection Area / Union Area
        
        Args:
            boxA: [x1, y1, x2, y2, score]
            boxB: [x1, y1, x2, y2, score]
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Compute intersection coordinates
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute intersection area
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        # Compute area of each box
        areaA = ((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = ((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # Compute union area
        union = areaA + areaB - interArea

        # Return IoU
        return interArea / (union + 1e-6)

    def preprocess_image(self, image):
        """
        Preprocess image to improve detection quality.
        
        Steps:
        1. Gamma correction to handle backlight
        2. White balance to normalize colors
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Preprocessed image
        """
        
        # Step 1: Gamma correction (handles lighting variations)
        gamma = self.estimate_gamma(image)
        image = np.uint8(255 * np.power(image / 255.0, gamma))
        
        # Step 2: White balance (normalizes colors)
        image = self.white_balance(image)
        
        return image

    def white_balance(self, image):
        """
        Apply automatic white balance using channel stretching.
        
        For each color channel:
        1. Find 1st and 99th percentiles
        2. Stretch to [0, 255]
        
        This removes color casts and improves color consistency.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            White balanced image
        """
        result = image.astype(np.float32)
        for i in range(3):  # Process each color channel
            channel = result[:, :, i]
            
            # Find extreme values (ignore outliers)
            min_val, max_val = np.percentile(channel, 1), np.percentile(channel, 99)
            
            # Stretch channel to [0, 255]
            if max_val > min_val:
                result[:, :, i] = np.clip((channel - min_val) / (max_val - min_val) * 255, 0, 255)
        
        return result.astype(np.uint8)

    def estimate_gamma(self, image):
        """
        Estimate gamma correction value based on image brightness.
        
        Gamma < 1.0: brightens image (good for dark/backlit images)
        Gamma = 1.0: no change
        Gamma > 1.0: darkens image (good for overexposed images)
        
        Args:
            image: Input image (BGR)
            
        Returns:
            float: Gamma value
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # Adaptive gamma based on brightness
        if mean_brightness < 80:
            # Very dark image - brighten significantly
            gamma = 0.6
        elif mean_brightness < 120:
            # Dark image - brighten
            gamma = 0.75
        elif mean_brightness > 200:
            # Very bright image - darken significantly
            gamma = 0.4
        elif mean_brightness > 180:
            # Bright image - darken
            gamma = 0.55
        else:
            # Normal brightness - no change
            gamma = 1.0
        
        return gamma