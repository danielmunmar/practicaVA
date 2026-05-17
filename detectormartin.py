import cv2
import numpy as np


class RoadPanelDetector:
    """
    Detects road information panels using:
    1. MSER (Maximally Stable Extremal Regions) for region detection
    2. HSV + LAB color analysis for blue color verification
    3. Morphological operations for robustness
    4. NMS for duplicate elimination
    """
    
    def __init__(self):
        """Initialize the detector with optimized parameters."""
        
        # MSER parameters - tuned for highway panel detection
        # min_area: minimum region size in pixels
        # max_area: maximum region size in pixels
        # delta: threshold increase between MSER calculations
        self.mser = cv2.MSER_create(
            min_area=300,
            max_area=20000,
            delta=10
        )

        # Standard size for blue score computation
        self.std_w = 40
        self.std_h = 80

        # Ideal blue mask (all ones - full blue color expected)
        self.ideal_mask = np.ones(
            (self.std_h, self.std_w),
            dtype=np.uint8
        )

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
        image = self._preprocess_image(image)

        # Step 2: Convert to grayscale for MSER
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
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
            if w < 60 or h < 30:
                continue

            # Filter by aspect ratio (panels are wider than tall)
            aspect_ratio = w / float(h)

            # Adjusted range based on real panel proportions
            if aspect_ratio < 1.2 or aspect_ratio > 4.0:
                continue

            # Expand bounding box to include white border
            # MSER detects the blue interior, we need the whole panel
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.1)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + w + pad_x)
            y2 = min(image.shape[0], y + h + pad_y)

            candidate_boxes.append((x1, y1, x2, y2))

        # Step 6: Score each candidate box based on blue color
        for box in candidate_boxes:
            x1, y1, x2, y2 = box

            # Extract region from image
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Compute blue color score (0 to 1, where 1 is perfect blue)
            score = self.compute_blue_score(crop)

            if score > 0.5:
                detections.append([x1, y1, x2, y2, score])

        # Step 7: Remove duplicate detections (NMS)
        detections = self.non_max_suppression(detections)

        return detections

    def compute_blue_score(self, crop):
        """
        Computes a score representing how much the crop looks like a blue panel.
        
        Strategy:
        1. Use HSV to detect blue hue and saturation
        2. Use LAB to verify blue color (low b channel)
        3. Combine both masks
        4. Compute intensity and clarity
        5. Return weighted score
        
        Args:
            crop: Image region to analyze
            
        Returns:
            float: Score between 0 (not blue) and 1 (perfectly blue)
        """
        
        # Resize to standard size for consistent scoring
        resized = cv2.resize(crop, (self.std_w, self.std_h))
        
        # Convert to HSV for hue-based color detection
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Convert to LAB for blue channel analysis
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)
        
        # Get brightness (Value in HSV)
        avg_v = np.mean(hsv[:, :, 2])
        
        # ===== HSV BLUE DETECTION =====
        # Adapt saturation threshold based on brightness
        sat_min = 50 if avg_v > 50 else 30
        
        # Blue limits in HSV:
        lower_blue = np.array([95, sat_min, 25])
        upper_blue = np.array([145, 255, 255])
        
        mask_hsv = cv2.inRange(hsv, lower_blue, upper_blue)
        lower_white = np.array([0, 0, 170])
        upper_white = np.array([180, 70, 255])

        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        border = np.zeros_like(white_mask)

        border[:4, :] = 1
        border[-4:, :] = 1
        border[:, :4] = 1
        border[:, -4:] = 1

        white_border_score = np.sum(
            (white_mask > 0) & (border > 0)
        ) / np.sum(border)

        inner = np.zeros_like(mask_hsv)

        inner[5:-5, 5:-5] = 1

        blue_inner_score = np.sum(
            (mask_hsv > 0) & (inner > 0)
        ) / np.sum(inner)
        
        # Morphological closing to fill small holes
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        
        # ===== LAB BLUE DETECTION =====
        b_channel = lab[:, :, 2]
        mask_lab = cv2.inRange(b_channel, 0, 118)
        
        # ===== COMBINE MASKS =====
        # Both HSV and LAB must agree it's blue
        mask = cv2.bitwise_and(mask_hsv, mask_lab)
        
        # ===== COMPUTE SCORE =====
        # Intensity: what percentage of pixels are blue
        intensity = np.sum(mask) / (self.std_w * self.std_h * 255.0)
        
        # Clarity: is the image blurry? (Laplacian variance)
        clarity = cv2.Laplacian(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() / 1000.0
        
        # Weight: intensity is more important than clarity
        # In bad weather (low clarity), we don't penalize much
        clarity_weight = 0.1 if clarity < 0.1 else 0.2
        
        # Final score: weighted combination
        score = (1.0 - clarity_weight) * intensity + clarity_weight * min(clarity, 1.0)
        score = (
            0.7 * blue_inner_score +
            0.3 * white_border_score
        )
        return float(score)

    def non_max_suppression(self, detections):
        """
        Remove duplicate detections by keeping only the best one
        from each group of overlapping boxes.
        
        Args:
            detections: List of [x1, y1, x2, y2, score]
            
        Returns:
            List of non-overlapping detections
        """
        if len(detections) == 0:
            return []

        # Sort by area (largest first)
        # This way we prioritize larger detections
        detections = sorted(
            detections,
            key=lambda d: (d[4]),
            reverse=True
        )

        final = []

        for det in detections:
            overlaps = False

            for kept in final:
                # Compute IoU (Intersection over Union)
                iou = self.compute_iou(det, kept)

                if iou > 0.4:
                    overlaps = True
                    break

            # Keep this detection if it doesn't overlap much with others
            if not overlaps:
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
        areaA = (
            (boxA[2] - boxA[0]) *
            (boxA[3] - boxA[1])
        )

        areaB = (
            (boxB[2] - boxB[0]) *
            (boxB[3] - boxB[1])
        )

        # Compute union area
        union = areaA + areaB - interArea

        # Return IoU
        return interArea / (union + 1e-6)

    def _preprocess_image(self, image):
        """
        Preprocess image to improve detection quality.
        
        Steps:
        1. Detail enhancement to reduce fog/haze effects
        2. Bilateral filter to smooth while preserving edges
        3. Gamma correction to handle backlight
        4. White balance to normalize colors
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Preprocessed image
        """
        
        # Step 1: Detail enhancement (reduces fog/haze)
        image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
        
        # Step 2: Bilateral filter (smooths while preserving edges)
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Step 3: Gamma correction (handles lighting variations)
        gamma = self._estimate_gamma(image)
        image = np.uint8(255 * np.power(image / 255.0, gamma))
        
        # Step 4: White balance (normalizes colors)
        image = self._white_balance(image)
        
        return image

    def _white_balance(self, image):
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

    def _estimate_gamma(self, image):
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
        """if mean_brightness < 80:
            gamma = 0.75
        elif mean_brightness > 180:
            gamma = 0.55
        else:
            gamma = 1.0
        return gamma"""

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