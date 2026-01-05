import cv2
import numpy as np

class AnalyticsManager:
    def __init__(self, height, width):
        self.heatmap_accumulator = np.zeros((height, width), dtype=np.float32)

    def update_heatmap(self, detections):
        for (x1, y1, x2, y2) in detections:
            # for the heatmap, increment the area of detection
            self.heatmap_accumulator[y1:y2, x1:x2] += 1

    def generate_heatmap(self):
        #normalization and colormap application
        heatmap_norm = cv2.normalize(self.heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_norm = np.uint8(heatmap_norm)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        return heatmap_color