# Video Intelligence Pipeline - Backend

AI-powered long-form video processing pipeline using Gemini 2.5 Pro, Google Drive, and MongoDB.

## 🏗️ Architecture

```
Backend/
├── main.py                 # FastAPI application
├── config.py              # Configuration and environment variables
├── requirements.txt       # Python dependencies
├── models/               # Data models and database
│   ├── database.py       # MongoDB connection
│   └── video_job.py      # Job schemas
├── services/             # Core services
│   ├── drive_service.py  # Google Drive API
│   ├── gemini_service.py # Gemini API wrapper
│   └── pipeline.py       # Processing orchestration
├── utils/                # Utility functions
│   └── ffmpeg_utils.py   # Video/audio processing
└── routes/               # API endpoints
    └── video_routes.py   # Video processing routes
```

## 📋 Prerequisites

1. **Python 3.10+**
2. **FFmpeg** - Required for video/audio processing
3. **Google Cloud Account** - For Gemini API and Drive
4. **MongoDB Atlas** - For data storage

## 🚀 Setup Instructions

### Step 1: Install FFmpeg

**Windows (PowerShell):**
```powershell
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
# Add to PATH: C:\ffmpeg\bin
```

**Verify installation:**
```powershell
ffmpeg -version
```

### Step 2: Install Python Dependencies

```powershell
cd Backend
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Your `.env` file is already configured with:
- ✅ Gemini API key
- ✅ MongoDB connection
- ✅ Google Drive OAuth credentials
- ✅ Drive folder ID

### Step 4: Run the Server

```powershell
python main.py
```

Or with uvicorn:
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

## 📡 API Endpoints

### 1. Process Video
```http
POST /api/videos/process
Content-Type: application/json

{
  "drive_video_url": "https://drive.google.com/file/d/FILE_ID/view",
  "video_name": "My Seminar Video"
}
```

**Response:**
```json
{
  "job_id": "507f1f77bcf86cd799439011",
  "status": "pending",
  "progress": 0.0,
  "video_name": "My Seminar Video",
  "created_at": "2026-01-15T10:30:00Z"
}
```

### 2. Check Status
```http
GET /api/videos/status/{job_id}
```

**Response:**
```json
{
  "job_id": "507f1f77bcf86cd799439011",
  "status": "processing",
  "progress": 0.65,
  "video_name": "My Seminar Video"
}
```

**Status values:** `pending`, `downloading`, `extracting`, `transcribing`, `analyzing`, `synthesizing`, `completed`, `failed`

### 3. Get Results
```http
GET /api/videos/results/{job_id}
```

**Response:**
```json
{
  "job_id": "507f1f77bcf86cd799439011",
  "status": "completed",
  "video_name": "My Seminar Video",
  "duration": 7530.5,
  "executive_summary": "This seminar covers...",
  "topics": [
    {
      "title": "Introduction to Data Strategy",
      "timestamp_range": ["00:00:00", "00:15:30"],
      "summary": "Overview of key concepts...",
      "key_points": ["Point 1", "Point 2"],
      "frames": [
        {
          "timestamp": "00:05:23",
          "frame_number": 5,
          "drive_url": "https://drive.google.com/...",
          "description": "Slide showing framework diagram",
          "ocr_text": "Data Strategy Framework",
          "type": "slide"
        }
      ],
      "quotes": ["The most important aspect is..."]
    }
  ],
  "key_takeaways": ["Takeaway 1", "Takeaway 2"],
  "entities": {
    "companies": ["Google", "Microsoft"],
    "concepts": ["Machine Learning", "Data Pipeline"],
    "tools": ["Python", "TensorFlow"]
  },
  "total_frames": 125,
  "completed_at": "2026-01-15T11:00:00Z"
}
```

### 4. List All Jobs
```http
GET /api/videos/list?limit=50&skip=0
```

### 5. Delete Job
```http
DELETE /api/videos/{job_id}
```

## 🔄 Processing Pipeline

1. **Download** - Video downloaded from Google Drive
2. **Extract Audio** - Audio extracted as 16kHz mono WAV
3. **Transcribe** - Audio split into 5-min chunks, transcribed with Gemini
4. **Analyze Transcript** - Topics, entities, key moments extracted
5. **Extract Frames** - Keyframes extracted every 60 seconds
6. **Analyze Frames** - Gemini Vision analyzes frames (OCR + description)
7. **Upload Frames** - Frames uploaded to Drive with public links
8. **Synthesize** - Results combined into structured output
9. **Store** - Final results saved to MongoDB

## 🧪 Testing

### Quick Test with cURL

```bash
# Start processing
curl -X POST http://localhost:8000/api/videos/process \
  -H "Content-Type: application/json" \
  -d '{"drive_video_url": "YOUR_DRIVE_URL", "video_name": "Test Video"}'

# Check status (replace JOB_ID)
curl http://localhost:8000/api/videos/status/JOB_ID

# Get results
curl http://localhost:8000/api/videos/results/JOB_ID
```

### Test with Python

```python
import requests

# Process video
response = requests.post(
    "http://localhost:8000/api/videos/process",
    json={
        "drive_video_url": "https://drive.google.com/file/d/YOUR_FILE_ID/view",
        "video_name": "My Test Video"
    }
)
job = response.json()
job_id = job["job_id"]

# Poll status
import time
while True:
    status = requests.get(f"http://localhost:8000/api/videos/status/{job_id}").json()
    print(f"Progress: {status['progress']*100:.1f}% - {status['status']}")
    
    if status["status"] in ["completed", "failed"]:
        break
    
    time.sleep(10)

# Get results
results = requests.get(f"http://localhost:8000/api/videos/results/{job_id}").json()
print(f"Topics: {len(results['topics'])}")
print(f"Frames: {results['total_frames']}")
```

## ⚙️ Configuration

Edit `config.py` to adjust:

```python
MAX_AUDIO_CHUNK_DURATION = 300  # 5 minutes
AUDIO_OVERLAP_DURATION = 30     # 30 seconds overlap
KEYFRAME_INTERVAL = 60          # Extract frame every 60s
MAX_FRAMES_PER_VIDEO = 120      # Max frames to extract
AUDIO_SAMPLE_RATE = 16000       # 16kHz for transcription
```

## 📊 Cost Estimation

**Per 2-hour video:**
- Transcription (24 chunks × 5min): ~$0.50-1
- Transcript analysis: ~$0.50-1
- Vision analysis (~120 frames): ~$2-3
- Synthesis: ~$0.50-1

**Total: ~$4-6 per video**

## 🐛 Troubleshooting

### FFmpeg not found
```powershell
# Check if in PATH
ffmpeg -version

# Add to PATH if needed
$env:Path += ";C:\ffmpeg\bin"
```

### MongoDB connection issues
- Verify `MONGODB_URI` in `.env`
- Check MongoDB Atlas IP whitelist (allow `0.0.0.0/0` for development)
- Ensure network access is configured

### Google Drive authentication
- Refresh token expires after ~6 months
- Re-generate OAuth token if needed
- Check `CLIENT_ID` and `CLIENT_SECRET` are correct

### Out of memory
- Reduce `MAX_FRAMES_PER_VIDEO` in config
- Process shorter videos first
- Increase system RAM allocation

## 🚀 Next Steps (Phase 2)

- [ ] Smart frame selection with scene detection (PySceneDetect)
- [ ] Audio-cued frame extraction
- [ ] Better speaker diarization
- [ ] Batch frame analysis by topic
- [ ] Cost tracking per video
- [ ] Caching and resumable processing

## 📝 Notes

- Videos remain in Google Drive (not downloaded permanently)
- Temp files auto-cleaned after processing
- Frames stored in Drive with public links
- MongoDB stores metadata and results
- Background processing with FastAPI BackgroundTasks
