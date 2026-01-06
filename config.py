import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "data", "1.mp4")
DEFAULT_VIDEO = "/Users/leo/Downloads/1.mp4"

#YOLO model
CLASS_NAMES = {
    0: "vehicle",
    1: "light_source"
}
# Confidence treshold
DEFAULT_CONF = 0.45

HIGH_BEAM_INTENSITY= 0.35

#added padding to mask boxes( for merging close boxes)
MASK_PADDING = 15      # Extra horizontal safety margin
MERGE_THRESHOLD = 20   # Maximum gap (in pixels) to merge two columns
# Number of frames to keep a car masked after it disappears from detection
MAX_GHOST_FRAMES = 10 

# Margin from the edges (in pixels) to consider a car is "exiting" the scene
# If a car is NOT in this margin (meaning it's more towards the center), 
# we apply persistence logic
EDGE_MARGIN = 50

MASK_PADDING = 20 

# tracking with ByteTrack
USE_TRACKING = True

#UI Settings
APP_TITLE = "AI Matrix LED - Intelligent Lighting Assistant"

#colors for UI
UI_COLORS = {
    "primary": "#00FFAA",    
    "background": "#0E1117", 
    "text": "#FFFFFF"
}

# messages
STATUS_MESSAGES = {
    "active": "SYSTEM ACTIVE: MASKING TRAFFIC",
    "idle": "FULL HIGH BEAM: ROAD CLEAR",
    "error": "ERROR: SENSOR/CAMERA OBSTRUCTED"
}