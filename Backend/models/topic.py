from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from models.video_job import PyObjectId


class TopicVideo(BaseModel):
    """Represents a single video within a topic/playlist"""
    video_url: str
    video_title: str
    duration: Optional[float] = 0
    job_id: Optional[str] = None  # Links to VideoJob._id
    status: str = "pending"  # pending | processing | completed | failed
    order: int = 0  # Position in playlist


class Topic(BaseModel):
    """Represents a playlist-based learning topic (like a book)"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    playlist_url: str
    title: str
    description: Optional[str] = ""
    channel: Optional[str] = ""
    user_id: Optional[str] = None  # Clerk user ID
    video_count: int = 0
    videos: List[TopicVideo] = []
    status: str = "pending"  # pending | processing | completed | failed
    progress: float = 0.0  # completed / total
    current_video_index: int = 0  # Which video is being processed now
    error_message: Optional[str] = None

    # Generated after all videos are processed (Phase 4)
    topic_summary: Optional[str] = None
    learning_objectives: List[str] = []
    prerequisites: List[str] = []
    difficulty_level: Optional[str] = None
    estimated_total_time: Optional[str] = None
    chapter_outline: List[Dict[str, Any]] = []

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class TopicCreate(BaseModel):
    """Request model for creating a new topic from a playlist"""
    playlist_url: str
    user_id: Optional[str] = None


class TopicResponse(BaseModel):
    """Response after creating a topic"""
    topic_id: str
    title: str
    video_count: int
    status: str
    created_at: datetime


class TopicProgress(BaseModel):
    """Lightweight progress response for polling"""
    topic_id: str
    status: str
    progress: float
    video_count: int
    completed_count: int
    current_video_index: int
    current_video_title: Optional[str] = None


class TopicDetail(BaseModel):
    """Full topic detail with all video statuses"""
    topic_id: str
    playlist_url: str
    title: str
    description: Optional[str] = ""
    channel: Optional[str] = ""
    user_id: Optional[str] = None
    video_count: int
    videos: List[TopicVideo] = []
    status: str
    progress: float
    current_video_index: int = 0
    topic_summary: Optional[str] = None
    learning_objectives: List[str] = []
    prerequisites: List[str] = []
    difficulty_level: Optional[str] = None
    estimated_total_time: Optional[str] = None
    chapter_outline: List[Dict[str, Any]] = []
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        json_encoders = {ObjectId: str}
