from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Optional
from bson import ObjectId
from datetime import datetime

from models.database import db
from models.topic import TopicCreate, TopicResponse, TopicProgress, TopicDetail, TopicVideo
from services.playlist_service import playlist_service


router = APIRouter(prefix="/api/topics", tags=["topics"])


@router.post("/process", response_model=TopicResponse)
async def process_playlist(
    request: TopicCreate,
    background_tasks: BackgroundTasks
):
    """
    Accept a playlist URL, extract videos, create a Topic, and start processing.
    """
    try:
        # Create the topic (extracts playlist info)
        topic_id = await playlist_service.create_topic_from_playlist(
            playlist_url=request.playlist_url,
            user_id=request.user_id
        )

        # Fetch the created topic for the response
        database = db.get_db()
        topic = await database.topics.find_one({"_id": ObjectId(topic_id)})

        # Start processing in background
        background_tasks.add_task(playlist_service.process_topic, topic_id)

        return TopicResponse(
            topic_id=topic_id,
            title=topic.get("title", ""),
            video_count=topic.get("video_count", 0),
            status="processing",
            created_at=topic.get("created_at", datetime.utcnow())
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create topic: {str(e)}")


@router.get("", response_model=list)
async def list_topics(user_id: Optional[str] = Query(None)):
    """List all topics for a user."""
    try:
        database = db.get_db()
        query = {}
        if user_id:
            query["user_id"] = user_id

        cursor = database.topics.find(query).sort("created_at", -1)
        topics = await cursor.to_list(length=100)

        results = []
        for t in topics:
            completed_count = sum(
                1 for v in t.get("videos", []) if v.get("status") == "completed"
            )
            results.append({
                "topic_id": str(t["_id"]),
                "title": t.get("title", ""),
                "channel": t.get("channel", ""),
                "video_count": t.get("video_count", 0),
                "completed_count": completed_count,
                "status": t.get("status", "pending"),
                "progress": t.get("progress", 0.0),
                "topic_summary": t.get("topic_summary"),
                "difficulty_level": t.get("difficulty_level"),
                "created_at": t.get("created_at"),
                "completed_at": t.get("completed_at")
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list topics: {str(e)}")


@router.get("/{topic_id}", response_model=TopicDetail)
async def get_topic_detail(topic_id: str):
    """Get full topic detail with all video statuses."""
    try:
        database = db.get_db()
        topic = await database.topics.find_one({"_id": ObjectId(topic_id)})

        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")

        return TopicDetail(
            topic_id=str(topic["_id"]),
            playlist_url=topic.get("playlist_url", ""),
            title=topic.get("title", ""),
            description=topic.get("description", ""),
            channel=topic.get("channel", ""),
            user_id=topic.get("user_id"),
            video_count=topic.get("video_count", 0),
            videos=[TopicVideo(**v) for v in topic.get("videos", [])],
            status=topic.get("status", "pending"),
            progress=topic.get("progress", 0.0),
            current_video_index=topic.get("current_video_index", 0),
            topic_summary=topic.get("topic_summary"),
            learning_objectives=topic.get("learning_objectives", []),
            prerequisites=topic.get("prerequisites", []),
            difficulty_level=topic.get("difficulty_level"),
            estimated_total_time=topic.get("estimated_total_time"),
            chapter_outline=topic.get("chapter_outline", []),
            created_at=topic.get("created_at", datetime.utcnow()),
            completed_at=topic.get("completed_at")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get topic: {str(e)}")


@router.get("/{topic_id}/progress", response_model=TopicProgress)
async def get_topic_progress(topic_id: str):
    """Lightweight progress endpoint for polling."""
    try:
        database = db.get_db()
        topic = await database.topics.find_one(
            {"_id": ObjectId(topic_id)},
            {"videos": 1, "status": 1, "progress": 1, "video_count": 1,
             "current_video_index": 1}
        )

        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")

        videos = topic.get("videos", [])
        completed_count = sum(1 for v in videos if v.get("status") == "completed")
        current_idx = topic.get("current_video_index", 0)
        current_title = videos[current_idx].get("video_title") if current_idx < len(videos) else None

        return TopicProgress(
            topic_id=topic_id,
            status=topic.get("status", "pending"),
            progress=topic.get("progress", 0.0),
            video_count=topic.get("video_count", 0),
            completed_count=completed_count,
            current_video_index=current_idx,
            current_video_title=current_title
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")
