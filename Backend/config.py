import os
from dotenv import load_dotenv

load_dotenv()

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-2.0-flash-exp")

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_NAME = os.getenv("MONGODB_NAME", "VideoProcessor")

# Google Drive Configuration
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
DRIVE_FILE_PERMISSION = os.getenv("DRIVE_FILE_PERMISSION", "anyone")

# Processing Configuration
TEMP_DIR = "temp"
MAX_AUDIO_CHUNK_DURATION = 300  # 5 minutes in seconds
AUDIO_OVERLAP_DURATION = 30  # 30 seconds overlap
KEYFRAME_INTERVAL = 60  # Extract keyframe every 60 seconds
MAX_FRAMES_PER_VIDEO = 120  # Maximum frames to extract
MAX_ANALYSIS_FRAMES = 150 # Max frames to send to Gemini for deep analysis (Phase 2)
# Concurrency Configuration (Tune based on Server RAM and API Limits)
# For Render Free (512MB): Use 2
# For Render Starter (2GB): Use 5
# For Render Pro (4GB+): Use 10+
MAX_CONCURRENT_TRANSCRIBES = 2
MAX_CONCURRENT_VISION_TASKS = 2
MAX_CONCURRENT_UPLOADS = 3

AUDIO_SAMPLE_RATE = 16000  # 16kHz for transcription

# YouTube Configuration (optional)
YOUTUBE_COOKIES_PATH = os.getenv("YOUTUBE_COOKIES_PATH", None)  # Path to cookies.txt file
YOUTUBE_COOKIES_FROM_BROWSER = os.getenv("YOUTUBE_COOKIES_FROM_BROWSER", None)  # e.g., "chrome", "firefox", "edge"

# API Configuration
# Allow localhost for development, add your production domains here
CORS_ORIGINS = [
    # Local development
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    # Production - Add your Vercel URL here
    "https://video-summarizer-dusky.vercel.app",
    # For testing - remove in production for security
    "https://long-from-video-summariser-beyond-ai.onrender.com",
]

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)
