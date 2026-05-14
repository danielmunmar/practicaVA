import cv2
import numpy as np


class RoadPanelDetector:
    def __init__(self):
        # Parámetros MSER
        self.mser = cv2.MSER_create(
            min_area=300,
            max_area=20000,
            delta=10
        )

        # Tamaño estándar
        self.std_w = 64
        self.std_h = 64

        # Máscara ideal azul
        self.ideal_mask = np.ones(
            (self.std_h, self.std_w),
            dtype=np.uint8
        )

    def detect(self, image):
        detections = []

        image = self._preprocess_image(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Mejorar contraste
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        gray = clahe.apply(gray)

        # Detectar regiones
        regions, _ = self.mser.detectRegions(gray)

        candidate_boxes = []

        for region in regions:
            x, y, w, h = cv2.boundingRect(region)

            # Filtrar tamaños pequeños
            if w < 60 or h < 30:
                continue

            aspect_ratio = w / float(h)

            if aspect_ratio < 1.2 or aspect_ratio > 4.0:
                continue

            # Expandir caja
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.1)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + w + pad_x)
            y2 = min(image.shape[0], y + h + pad_y)

            candidate_boxes.append((x1, y1, x2, y2))

        for box in candidate_boxes:
            x1, y1, x2, y2 = box

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            score = self.compute_blue_score(crop)

            if score > 0.5:
                detections.append([x1, y1, x2, y2, score])

        detections = self.non_max_suppression(detections)

        return detections

    def compute_blue_score(self, crop):
        resized = cv2.resize(crop, (self.std_w, self.std_h))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)
 
        avg_v = np.mean(hsv[:, :, 2])
 
        # Rango HSV ampliado para azules desteñidos por lluvia/niebla.
        #   - Hue: 95–145 (era 100–140) — cubre azules más verdosos o violáceos
        #   - Sat: mínimo 50 con buena luz, 30 con poca (era 100/60) — niebla
        #     desatura mucho el color
        #   - Val mínimo: 25 (era 40) — captura paneles en sombra profunda
        sat_min = 50 if avg_v > 50 else 30
        lower_blue = np.array([95,  sat_min, 25])
        upper_blue = np.array([145, 255,     255])
 
        mask_hsv = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
 
        # Canal b de LAB corregido.
        #   En OpenCV, Lab se remapea a [0, 255]: el valor neutro es 128.
        #   Azul → b < 128 (valores BAJOS), no altos como estaba antes (140-255).
        #   Rango 0–118 captura azules puros y los ligeramente grises por niebla.
        b_channel = lab[:, :, 2]
        mask_lab = cv2.inRange(b_channel, 0, 118)
 
        mask = cv2.bitwise_and(mask_hsv, mask_lab)
 
        intensity = np.sum(mask) / (self.std_w * self.std_h * 255.0)
        clarity = cv2.Laplacian(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() / 1000.0
 
        # Peso de clarity reducido en condiciones adversas (imagen borrosa
        #   por lluvia penalizaba innecesariamente). Si la imagen es muy borrosa
        #   (clarity < 0.1) no penalizar — es culpa del clima, no del panel.
        clarity_weight = 0.1 if clarity < 0.1 else 0.2
        score = (1.0 - clarity_weight) * intensity + clarity_weight * min(clarity, 1.0)
        return float(score)

    def non_max_suppression(self, detections):
        if len(detections) == 0:
            return []

        # Ordenar por área DESCENDENTE
        # (más grande primero)
        detections = sorted(
            detections,
            key=lambda d: (
                (d[2] - d[0]) *
                (d[3] - d[1])
            ),
            reverse=True
        )

        final = []

        for det in detections:
            overlaps = False

            for kept in final:
                iou = self.compute_iou(det, kept)

                if iou > 0.0:
                    overlaps = True
                    break

            if not overlaps:
                final.append(det)

        return final

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)

        interArea = interW * interH

        areaA = (
            (boxA[2] - boxA[0]) *
            (boxA[3] - boxA[1])
        )

        areaB = (
            (boxB[2] - boxB[0]) *
            (boxB[3] - boxB[1])
        )

        union = areaA + areaB - interArea

        return interArea / (union + 1e-6)

    def _preprocess_image(self, image):
        # Reducción de niebla
        image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
 
        # Filtro bilateral
        image = cv2.bilateralFilter(image, 9, 75, 75)
 
        # Gamma corregida — contraluz tiene media ALTA pero zonas
        #    de interés oscuras; bajar gamma (< 1.0) aclara las sombras.
        #    Antes: imagen oscura→0.6 (aclaraba bien), imagen clara→1.4 (oscurecía
        #    más, empeorando el contraluz). Ahora imagen muy clara→0.7 para
        #    recuperar detalle en las sombras del contraluz.
        gamma = self._estimate_gamma(image)
        image = np.uint8(255 * np.power(image / 255.0, gamma))
 
        # Balance de blancos automático (sin opencv-contrib)
        image = self._white_balance(image)
 
        return image
    def _white_balance(self, image):
        """Balance de blancos por escalado de canal (equivalente a SimpleWB)."""
        result = image.astype(np.float32)
        for i in range(3):
            channel = result[:, :, i]
            mn, mx = np.percentile(channel, 1), np.percentile(channel, 99)
            if mx > mn:
                result[:, :, i] = np.clip((channel - mn) / (mx - mn) * 255, 0, 255)
        return result.astype(np.uint8)
 
    def _estimate_gamma(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)

        if mean < 80:
            gamma = 0.75
        elif mean > 180:
            gamma = 0.55
        else:
            gamma = 1.0
        return gamma