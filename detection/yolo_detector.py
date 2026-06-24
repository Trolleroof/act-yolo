import numpy as np

CLASS_NAMES = {0: 'cube', 1: 'target_zone'}


class YOLODetector:
    def __init__(self, weights='weights/yolov8n_pickplace.pt', conf=0.4):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, rgb_img: np.ndarray) -> dict:
        """Return dict[class_name -> (5,) array or None].

        Each value is [cx, cy, w, h, conf] normalized. Keeps highest-conf
        box per class when multiple instances appear.
        """
        results = self.model(rgb_img, conf=self.conf, verbose=False)[0]
        detections = {name: None for name in CLASS_NAMES.values()}

        for box in results.boxes:
            cls = int(box.cls)
            name = CLASS_NAMES.get(cls)
            if name is None:
                continue
            xywhn = box.xywhn[0].cpu().numpy()
            conf = float(box.conf[0])
            candidate = np.append(xywhn, conf)
            if detections[name] is None or conf > detections[name][4]:
                detections[name] = candidate

        return detections

    def get_crop(self, rgb_img: np.ndarray, class_name: str,
                 dets: dict = None, pad: float = 0.1, size: int = 224) -> np.ndarray:
        """Crop around detected object; falls back to resized full frame if undetected."""
        import cv2
        if dets is None:
            dets = self.detect(rgb_img)
        if dets[class_name] is None:
            return cv2.resize(rgb_img, (size, size))

        H, W = rgb_img.shape[:2]
        cx, cy, w, h = dets[class_name][:4]
        x1 = int((cx - w / 2 - pad) * W)
        y1 = int((cy - h / 2 - pad) * H)
        x2 = int((cx + w / 2 + pad) * W)
        y2 = int((cy + h / 2 + pad) * H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        crop = rgb_img[y1:y2, x1:x2]
        return cv2.resize(crop, (size, size))
