import cv2
import numpy as np
from ultralytics import YOLO
import config

class MatrixEngine:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        # Dictionary to track lost cars: {id: [last_coords, frame_counter]}
        self.ghost_tracks = {}

    def process_frame(self, frame, conf_threshold, intensity):
        h, w = frame.shape[:2]
        results = self.model.track(frame, persist=True, conf=conf_threshold, verbose=False)[0]
        
        mask = np.full((h, w, 3), 255, dtype=np.uint8)
        
        detections = []
        intervals = []
        active_ids = set()

        # 1. Process active detections from YOLO
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                active_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2))
                
                # Update ghost storage with current position
                self.ghost_tracks[track_id] = [[x1, x2], 0] 
                
                start_x = max(0, x1 - config.MASK_PADDING)
                end_x = min(w, x2 + config.MASK_PADDING)
                intervals.append([start_x, end_x])

        # 2. Add "Ghost" intervals for missing cars (Anti-Flicker)
        for track_id in list(self.ghost_tracks.keys()):
            if track_id not in active_ids:
                coords, count = self.ghost_tracks[track_id]
                x1_last, x2_last = coords
                
                # Check if car is NOT near the vertical edges of the video
                not_at_edge = x1_last > config.EDGE_MARGIN and x2_last < (w - config.EDGE_MARGIN)
                
                # Keep masking if it's not too old and not at the edge
                if count < config.MAX_GHOST_FRAMES and not_at_edge:
                    self.ghost_tracks[track_id][1] += 1 # Increment skip counter
                    
                    start_x = max(0, x1_last - config.MASK_PADDING)
                    end_x = min(w, x2_last + config.MASK_PADDING)
                    intervals.append([start_x, end_x])
                else:
                    del self.ghost_tracks[track_id] # Permanent removal

        # 3. Sort and Merge intervals
        intervals.sort(key=lambda x: x[0])
        merged_intervals = []

        if intervals:
            curr_start, curr_end = intervals[0]
            for i in range(1, len(intervals)):
                next_start, next_end = intervals[i]
                if next_start <= curr_end + config.MERGE_THRESHOLD:
                    curr_end = max(curr_end, next_end)
                else:
                    merged_intervals.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged_intervals.append((curr_start, curr_end))

        # 4. Draw final columns
        for m_start, m_end in merged_intervals:
            cv2.rectangle(mask, (m_start, 0), (m_end, h), (0, 0, 0), -1)

        processed = cv2.addWeighted(frame, 1.0, mask, intensity, 0)
        return processed, detections