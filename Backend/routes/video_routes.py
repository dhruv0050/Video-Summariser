from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from bson import ObjectId
from datetime import datetime
from typing import List, Dict
import os
import shutil
import json

from models.database import db
from models.video_job import (
    VideoJobCreate,
    YouTubeJobCreate,
    UploadJobCreate,
    VideoJobResponse, 
    VideoJobResult,
    VideoJob,
    ReportSummary
)
from services.pipeline import pipeline
from services.gemini_service import GeminiService
import config

router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.post("/process", response_model=VideoJobResponse)
async def process_video(
    job_request: VideoJobCreate,
    background_tasks: BackgroundTasks
):
    """
    Start processing a video from Google Drive
    
    Args:
        job_request: Video job request with Drive URL
        background_tasks: FastAPI background tasks
    
    Returns:
        Job response with job_id and initial status
    """
    try:
        # Create job in database
        database = db.get_db()
        
        job = VideoJob(
            drive_video_url=job_request.drive_video_url,
            video_name=job_request.video_name,
            user_id=job_request.user_id,
            status="pending",
            progress=0.0
        )
        
        # Insert into MongoDB
        result = await database.video_jobs.insert_one(job.dict(by_alias=True))
        job_id = str(result.inserted_id)
        
        # Start processing in background
        background_tasks.add_task(pipeline.process_video, job_id)
        
        return VideoJobResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            video_name=job_request.video_name,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@router.post("/process-youtube", response_model=VideoJobResponse)
async def process_youtube_video(
    job_request: YouTubeJobCreate,
    background_tasks: BackgroundTasks
):
    """
    Start processing a video from YouTube
    
    Args:
        job_request: YouTube job request with YouTube URL
        background_tasks: FastAPI background tasks
    
    Returns:
        Job response with job_id and initial status
    """
    try:
        # Create job in database
        database = db.get_db()
        
        job = VideoJob(
            youtube_url=job_request.youtube_url,
            video_name=job_request.video_name,
            user_id=job_request.user_id,
            video_source="youtube",
            status="pending",
            progress=0.0
        )
        
        # Insert into MongoDB
        result = await database.video_jobs.insert_one(job.dict(by_alias=True))
        job_id = str(result.inserted_id)
        
        # Start processing in background
        background_tasks.add_task(pipeline.process_video, job_id)
        
        return VideoJobResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            video_name=job_request.video_name,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create YouTube job: {str(e)}")


@router.post("/process-upload", response_model=VideoJobResponse)
async def process_uploaded_video(
    file: UploadFile = File(...),
    video_name: str = Form(None),
    user_id: str = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Start processing a video from direct file upload
    
    Args:
        file: Uploaded video file
        video_name: Optional video name
        background_tasks: FastAPI background tasks
    
    Returns:
        Job response with job_id and initial status
    """
    try:
        # Validate file type
        content_type = file.content_type or ""
        if not any(ext in content_type for ext in ["video", "mp4", "mov", "avi", "mkv", "webm"]):
            # Also check file extension
            filename = file.filename or ""
            if not any(filename.lower().endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]):
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
        
        # Create job in database first
        database = db.get_db()
        
        job = VideoJob(
            video_name=video_name or file.filename or "Uploaded Video",
            video_source="upload",
            user_id=user_id,
            status="pending",
            progress=0.0
        )
        
        # Insert into MongoDB
        result = await database.video_jobs.insert_one(job.dict(by_alias=True))
        job_id = str(result.inserted_id)
        
        # Save uploaded file to temp directory
        video_path = os.path.join(config.TEMP_DIR, f"{job_id}_video{os.path.splitext(file.filename or '')[-1]}")
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update job with video path
        await database.video_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"uploaded_video_path": video_path}}
        )
        
        # Start processing in background
        background_tasks.add_task(pipeline.process_video, job_id)
        
        return VideoJobResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            video_name=video_name or file.filename or "Uploaded Video",
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create upload job: {str(e)}")


@router.get("/status/{job_id}", response_model=VideoJobResponse)
async def get_job_status(job_id: str):
    """
    Get the current status of a processing job
    
    Args:
        job_id: Job ID to check
    
    Returns:
        Job status and progress
    """
    try:
        database = db.get_db()
        
        job = await database.video_jobs.find_one({"_id": ObjectId(job_id)})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return VideoJobResponse(
            job_id=job_id,
            status=job.get("status", "unknown"),
            progress=job.get("progress", 0.0),
            video_name=job.get("video_name"),
            current_action=job.get("current_action", ""),
            processing_logs=job.get("processing_logs", []),
            created_at=job.get("created_at", datetime.utcnow())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.get("/results/{job_id}", response_model=VideoJobResult)
async def get_job_results(job_id: str):
    """
    Get the results of a completed job
    
    Args:
        job_id: Job ID to retrieve results for
    
    Returns:
        Complete job results including topics, summary, frames
    """
    try:
        database = db.get_db()
        
        job = await database.video_jobs.find_one({"_id": ObjectId(job_id)})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.get("status") not in ["completed", "failed"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Job is still processing (status: {job.get('status')})"
            )
        
        if job.get("status") == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Job failed: {job.get('error_message', 'Unknown error')}"
            )
        
        return VideoJobResult(
            job_id=job_id,
            status=job.get("status"),
            video_name=job.get("video_name"),
            duration=job.get("duration"),
            executive_summary=job.get("executive_summary"),
            topics=job.get("topics", []),
            key_takeaways=job.get("key_takeaways", []),
            entities=job.get("entities", {}),
            slide_summary=job.get("slide_summary", []),
            total_frames=job.get("total_frames", 0),
            processing_cost=job.get("processing_cost"),
            completed_at=job.get("completed_at"),
            video_genre=job.get("video_genre"),
            genre_confidence=job.get("genre_confidence"),
            genre_reason=job.get("genre_reason")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job results: {str(e)}")


@router.get("/list")
async def list_jobs(limit: int = 50, skip: int = 0):
    """
    List all video processing jobs
    
    Args:
        limit: Maximum number of jobs to return
        skip: Number of jobs to skip (for pagination)
    
    Returns:
        List of jobs with basic info
    """
    try:
        database = db.get_db()
        
        cursor = database.video_jobs.find(
            {},
            {"_id": 1, "video_name": 1, "status": 1, "progress": 1, "created_at": 1}
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        jobs = await cursor.to_list(length=limit)
        
        return [
            {
                "job_id": str(job["_id"]),
                "video_name": job.get("video_name", "Unknown"),
                "status": job.get("status", "unknown"),
                "progress": job.get("progress", 0.0),
                "created_at": job.get("created_at")
            }
            for job in jobs
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated data
    
    Args:
        job_id: Job ID to delete
    
    Returns:
        Success message
    """
    try:
        database = db.get_db()
        
        job = await database.video_jobs.find_one({"_id": ObjectId(job_id)})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # TODO: Delete associated Drive files
        
        # Delete from database
        await database.video_jobs.delete_one({"_id": ObjectId(job_id)})
        
        return {"message": f"Job {job_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


@router.get("/reports", response_model=List[ReportSummary])
async def get_reports(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    status: str = Query(None),
    user_id: str = Query(None)
):
    """
    Get all past reports with pagination
    
    Args:
        page: Page number (1-indexed)
        limit: Number of results per page
        status: Filter by status (optional)
        user_id: Filter by Clerk user ID (optional)
    
    Returns:
        List of report summaries
    """
    try:
        database = db.get_db()
        
        # Build query - force completed reports only
        query = {"status": "completed"}
        # If a status filter is provided, keep it only if it's completed; otherwise still enforce completed
        if status and status == "completed":
            query["status"] = "completed"
        
        # Filter by user_id if provided
        if user_id:
            query["user_id"] = user_id
        
        # Calculate skip
        skip = (page - 1) * limit
        
        # Get completed jobs sorted by creation date (newest first)
        jobs = await database.video_jobs.find(query)\
            .sort("created_at", -1)\
            .skip(skip)\
            .limit(limit)\
            .to_list(length=limit)
        
        reports = []
        for job in jobs:
            thumbnail_url = None
            # Prefer top-level frames (stored during processing)
            frames = job.get("frames", []) or []
            if frames:
                thumb_frame = frames[0]
                thumbnail_url = thumb_frame.get("drive_url") or thumb_frame.get("url")

            # Fallback: look inside topics for a frame with a drive_url
            if not thumbnail_url:
                for topic in job.get("topics", []):
                    topic_frames = topic.get("frames", []) or []
                    if topic_frames:
                        thumb_frame = topic_frames[0]
                        thumbnail_url = thumb_frame.get("drive_url") or thumb_frame.get("url")
                        if thumbnail_url:
                            break

            reports.append(ReportSummary(
                job_id=str(job.get("_id")),
                video_name=job.get("video_name"),
                status=job.get("status"),
                topic_id=job.get("topic_id"),
                duration=job.get("duration"),
                topics_count=len(job.get("topics", [])),
                created_at=job.get("created_at"),
                completed_at=job.get("completed_at"),
                executive_summary=job.get("executive_summary"),
                thumbnail_url=thumbnail_url,
                video_source=job.get("video_source"),
                youtube_url=job.get("youtube_url"),
                youtube_video_id=job.get("youtube_video_id"),
                drive_video_url=job.get("drive_video_url"),
                drive_file_id=job.get("drive_file_id"),
                video_genre=job.get("video_genre"),
                genre_confidence=job.get("genre_confidence"),
            ))
        
        return reports
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get reports: {str(e)}")


@router.get("/{job_id}/download/transcript")
async def download_transcript(job_id: str, format: str = Query("json", regex="^(json|txt)$")):
    """
    Download transcript for a completed job
    
    Args:
        job_id: Job ID
        format: Output format - "json" or "txt"
    
    Returns:
        Transcript file download
    """
    try:
        database = db.get_db()
        job = await database.video_jobs.find_one({"_id": ObjectId(job_id)})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not completed (status: {job.get('status')})"
            )
        
        transcript = job.get("transcript", [])
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found for this job")
        
        if format == "json":
            # Return as JSON
            return StreamingResponse(
                iter([json.dumps(transcript, indent=2)]),
                media_type="application/json",
                headers={
                    "Content-Disposition": f'attachment; filename="transcript_{job_id}.json"'
                }
            )
        else:
            # Return as plain text with timestamps
            lines = []
            for seg in transcript:
                start_ts = _format_timestamp(seg.get("start_time", 0))
                end_ts = _format_timestamp(seg.get("end_time", 0))
                speaker = seg.get("speaker", "")
                text = seg.get("text", "")
                
                if speaker:
                    lines.append(f"[{start_ts} - {end_ts}] {speaker}: {text}")
                else:
                    lines.append(f"[{start_ts} - {end_ts}] {text}")
            
            transcript_text = "\n".join(lines)
            return StreamingResponse(
                iter([transcript_text]),
                media_type="text/plain",
                headers={
                    "Content-Disposition": f'attachment; filename="transcript_{job_id}.txt"'
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download transcript: {str(e)}")


@router.get("/{job_id}/download/audio")
async def download_audio(job_id: str):
    """
    Download audio file for a completed job
    
    Args:
        job_id: Job ID
    
    Returns:
        Audio file download
    """
    try:
        database = db.get_db()
        job = await database.video_jobs.find_one({"_id": ObjectId(job_id)})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not completed (status: {job.get('status')})"
            )
        
        # Check if audio path exists in job
        audio_path = job.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            # Try to construct path from job_id
            audio_path = os.path.join(config.TEMP_DIR, f"{job_id}_audio.wav")
            if not os.path.exists(audio_path):
                raise HTTPException(status_code=404, detail="Audio file not found for this job")
        
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename=f"audio_{job_id}.wav",
            headers={
                "Content-Disposition": f'attachment; filename="audio_{job_id}.wav"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download audio: {str(e)}")


def _format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ChatRequest(BaseModel):
    question: str
    conversation_history: List[Dict[str, str]] = []


@router.post("/chat/{job_id}")
async def chat_with_video(job_id: str, chat_request: ChatRequest):
    """
    Chat with a video using Gemini AI based on its transcript and report
    
    Args:
        job_id: Job ID of the video to chat about
        chat_request: Chat request with question and conversation history
    
    Returns:
        AI-generated response based on video context
    """
    try:
        database = db.get_db()
        
        # Fetch job from database
        job = await database.video_jobs.find_one({"_id": ObjectId(job_id)})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Video is still processing. Current status: {job.get('status')}"
            )
        
        # Prepare context from transcript and report
        transcript_text = ""
        if job.get("transcript"):
            transcript_segments = []
            for segment in job["transcript"]:
                timestamp = _format_timestamp(segment.get("start_time", 0))
                text = segment.get("text", "")
                speaker = segment.get("speaker", "")
                speaker_prefix = f"{speaker}: " if speaker else ""
                transcript_segments.append(f"[{timestamp}] {speaker_prefix}{text}")
            transcript_text = "\n".join(transcript_segments)
        
        # Extract key information from the job
        executive_summary = job.get("executive_summary", "")
        key_takeaways = job.get("key_takeaways", [])
        topics = job.get("topics", [])
        video_name = job.get("video_name", "this video")
        video_genre = job.get("video_genre", "")
        
        # Build topics summary
        topics_summary = ""
        if topics:
            topics_list = []
            for topic in topics:
                title = topic.get("title", "")
                timestamp_range = topic.get("timestamp_range", [])
                summary = topic.get("summary", "")
                key_points = topic.get("key_points", [])
                
                topic_str = f"**{title}**"
                if timestamp_range and len(timestamp_range) >= 2:
                    topic_str += f" ({timestamp_range[0]} - {timestamp_range[1]})"
                if summary:
                    topic_str += f"\n  Summary: {summary}"
                if key_points:
                    topic_str += "\n  Key Points:\n" + "\n".join([f"    - {kp}" for kp in key_points])
                topics_list.append(topic_str)
            topics_summary = "\n\n".join(topics_list)
        
        # Build the context prompt
        context = f"""You are a Video Insights Assistant helping users understand and analyze the video titled "{video_name}".

VIDEO OVERVIEW:
Genre: {video_genre}
Executive Summary: {executive_summary}

KEY TAKEAWAYS:
{chr(10).join([f"- {kt}" for kt in key_takeaways]) if key_takeaways else "Not available"}

TOPICS COVERED:
{topics_summary if topics_summary else "Not available"}

FULL TRANSCRIPT:
{transcript_text if transcript_text else "Transcript not available"}

Your role is to:
1. Answer questions about the video content, topics, and key points discussed
2. Provide specific quotes and timestamps when relevant
3. Deep dive into topics when asked
4. Help users discover insights from the video content
5. Be conversational and helpful
6. Reference the transcript to provide accurate information

When answering:
- Reference specific parts of the transcript with timestamps when applicable
- Cite the executive summary and key takeaways when relevant
- Be concise but thorough
- If you don't find the answer in the context, say so honestly"""

        # Build conversation history
        conversation = []
        if chat_request.conversation_history:
            for msg in chat_request.conversation_history[-5:]:  # Keep last 5 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    conversation.append(f"User: {content}")
                else:
                    conversation.append(f"Assistant: {content}")
        
        conversation_context = "\n".join(conversation) if conversation else ""
        
        # Prepare the prompt for Gemini
        prompt = f"""{context}

{"PREVIOUS CONVERSATION:" if conversation_context else ""}
{conversation_context}

USER QUESTION:
{chat_request.question}

Please provide a helpful, accurate response based on the video content above. Include relevant timestamps and quotes when applicable."""

        # Use Gemini to generate response
        gemini_service = GeminiService()
        response = gemini_service.text_model.generate_content(prompt)
        
        return {
            "response": response.text,
            "job_id": job_id,
            "video_name": video_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {str(e)}")

