from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class Frame(BaseModel):
    timestamp: str
    frame_number: int
    drive_url: Optional[str] = None
    description: Optional[str] = None
    ocr_text: Optional[str] = None
    type: Optional[str] = None  # "slide", "demo", "diagram", etc.


class Topic(BaseModel):
    title: str
    timestamp_range: List[str]  # [start, end]
    summary: Optional[str] = None
    key_points: List[str] = []
    frames: List[Frame] = []
    quotes: List[str] = []


class TranscriptSegment(BaseModel):
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    confidence: Optional[float] = None


class VideoJobCreate(BaseModel):
    drive_video_url: str
    video_name: Optional[str] = None


class VideoJob(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    drive_video_url: str
    drive_file_id: Optional[str] = None
    video_name: Optional[str] = None
    status: str = "pending"  # pending, downloading, processing, completed, failed
    progress: float = 0.0
    error_message: Optional[str] = None
    
    # Storage paths
    drive_folder_id: Optional[str] = None
    audio_drive_id: Optional[str] = None
    
    # Processing results
    transcript: List[TranscriptSegment] = []
    topics: List[Topic] = []
    executive_summary: Optional[str] = None
    key_takeaways: List[str] = []
    entities: Dict[str, List[str]] = {}
    
    # Metadata
    duration: Optional[float] = None
    total_frames: int = 0
    processing_cost: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class VideoJobResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    video_name: Optional[str] = None
    created_at: datetime
    
    class Config:
        json_encoders = {ObjectId: str}


class VideoJobResult(BaseModel):
    job_id: str
    status: str
    video_name: Optional[str] = None
    duration: Optional[float] = None
    executive_summary: Optional[str] = None
    topics: List[Topic] = []
    key_takeaways: List[str] = []
    entities: Dict[str, List[str]] = {}
    total_frames: int = 0
    processing_cost: Optional[float] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {ObjectId: str}
