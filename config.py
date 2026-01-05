import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "data", "video_test.mp4")
DEFAULT_VIDEO = "/Users/leo/Downloads/1.mp4"

#YOLO model
CLASS_NAMES = {
    0: "vehicle",
    1: "light_source"
}
# Confidence treshold
DEFAULT_CONF = 0.45

HIGH_BEAM_INTENSITY= 0.35


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