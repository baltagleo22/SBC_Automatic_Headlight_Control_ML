import cv2
import numpy as np
from ultralytics import YOLO
import config

class MatrixEngine:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)

    def process_frame(self, frame, conf_threshold, intensity):
        h, w = frame.shape[:2]
        # bytetrack tracking results
        results = self.model.track(frame, persist=True, conf=conf_threshold, verbose=False)[0]
        
        # 2. mask creation (white image with black boxes where to cut)
        mask = np.full((h, w, 3), 255, dtype=np.uint8)
        
        detections = []
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2))
                
                # define black rectangle area on mask
                cv2.rectangle(mask, 
                              (max(0, x1 - config.MASK_PADDING), 0), 
                              (min(w, x2 + config.MASK_PADDING), h), 
                              (0, 0, 0), -1)

        # 3. Blending
        processed = cv2.addWeighted(frame, 1.0, mask, intensity, 0)
        
        return processed, detections