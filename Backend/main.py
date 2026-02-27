from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from models.database import db
from routes.video_routes import router as video_router
from routes.topic_routes import router as topic_router
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    await db.connect_db()
    print("🚀 Video Intelligence Pipeline API started")
    print(f"📁 Temp directory: {config.TEMP_DIR}")
    print(f"🔑 Using model: {config.MODEL}")
    
    # Check YouTube cookies configuration
    if config.YOUTUBE_COOKIES_PATH:
        from services.youtube_service import YouTubeService
        cookies_path = YouTubeService._resolve_cookies_path()
        if cookies_path:
            file_size = os.path.getsize(cookies_path)
            print(f"✅ YouTube cookies configured: {cookies_path} ({file_size} bytes)")
        else:
            print(f"⚠️ YouTube cookies path set but file not found: {config.YOUTUBE_COOKIES_PATH}")
            print(f"   Current working directory: {os.getcwd()}")
    elif config.YOUTUBE_COOKIES_FROM_BROWSER:
        print(f"⚠️ YOUTUBE_COOKIES_FROM_BROWSER set to: {config.YOUTUBE_COOKIES_FROM_BROWSER}")
        print(f"   Note: This won't work on servers like Render. Use YOUTUBE_COOKIES_PATH instead.")
    else:
        print("ℹ️ No YouTube cookies configured - using tv_embedded client")
    
    yield
    
    # Shutdown
    await db.close_db()
    print("👋 API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Video Intelligence Pipeline API",
    description="Process long-form videos with AI for intelligent insights",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video_router)
app.include_router(topic_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Intelligence Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        database = db.get_db()
        await database.command("ping")
        
        return {
            "status": "healthy",
            "database": "connected",
            "model": config.MODEL
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    # Use PORT from environment (Render sets this) or default to 8000
    app_port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=app_port,
        reload=False
    )
