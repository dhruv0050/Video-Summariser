import asyncio
from datetime import datetime
from typing import Dict, Optional
from bson import ObjectId

from models.database import db
from models.video_job import VideoJob
from models.topic import Topic, TopicVideo
from services.youtube_service import youtube_service
from services.pipeline import pipeline


class PlaylistService:
    """Orchestrates playlist processing — creates a Topic and processes videos sequentially."""

    async def create_topic_from_playlist(self, playlist_url: str, user_id: str = None) -> str:
        """
        Extract playlist info, create a Topic document, and return the topic_id.
        Does NOT start processing — that's done separately via process_topic().
        """
        # Step 1: Extract playlist metadata via yt-dlp
        print(f"Extracting playlist info from: {playlist_url}")
        playlist_info = youtube_service.extract_playlist_info(playlist_url)

        # Step 2: Build TopicVideo list
        videos = []
        for v in playlist_info.get("videos", []):
            videos.append(TopicVideo(
                video_url=v["video_url"],
                video_title=v["video_title"],
                duration=v.get("duration", 0),
                order=v["order"],
                status="pending"
            ))

        # Step 3: Create Topic document
        topic = Topic(
            playlist_url=playlist_url,
            title=playlist_info.get("title", "Untitled Playlist"),
            description=playlist_info.get("description", ""),
            channel=playlist_info.get("channel", ""),
            user_id=user_id,
            video_count=len(videos),
            videos=videos,
            status="pending",
            progress=0.0,
            current_video_index=0
        )

        database = db.get_db()
        result = await database.topics.insert_one(topic.dict(by_alias=True))
        topic_id = str(result.inserted_id)
        print(f"Created topic {topic_id}: '{topic.title}' with {len(videos)} videos")
        return topic_id

    async def process_topic(self, topic_id: str):
        """
        Process all videos in a topic sequentially.
        Each video goes through the existing pipeline.process_video().
        """
        try:
            database = db.get_db()
            topic = await database.topics.find_one({"_id": ObjectId(topic_id)})
            if not topic:
                print(f"Topic {topic_id} not found")
                return

            videos = topic.get("videos", [])
            total = len(videos)
            user_id = topic.get("user_id")

            await self._update_topic(topic_id, {
                "status": "processing",
                "progress": 0.0
            })

            completed_count = 0

            for i, video_info in enumerate(videos):
                video_url = video_info.get("video_url")
                video_title = video_info.get("video_title", f"Video {i+1}")

                print(f"\n{'='*60}")
                print(f"Processing video {i+1}/{total}: {video_title}")
                print(f"{'='*60}")

                # Update topic progress
                await self._update_topic(topic_id, {
                    "current_video_index": i,
                    f"videos.{i}.status": "processing",
                    "progress": completed_count / total if total > 0 else 0
                })

                try:
                    # Create a VideoJob for this video (same as existing YouTube flow)
                    job = VideoJob(
                        youtube_url=video_url,
                        video_name=video_title,
                        video_source="youtube",
                        user_id=user_id,
                        topic_id=topic_id,  # Link back to topic
                        status="pending",
                        progress=0.0
                    )

                    result = await database.video_jobs.insert_one(job.dict(by_alias=True))
                    job_id = str(result.inserted_id)

                    # Update topic with job_id reference
                    await database.topics.update_one(
                        {"_id": ObjectId(topic_id)},
                        {"$set": {f"videos.{i}.job_id": job_id}}
                    )

                    # Build playlist context from previously completed videos (Phase 2)
                    playlist_context = await self._build_playlist_context(topic_id, i)

                    # Process through existing pipeline (await completion)
                    await pipeline.process_video(job_id, playlist_context=playlist_context)

                    # Check actual job status — pipeline catches errors internally
                    updated_job = await database.video_jobs.find_one(
                        {"_id": ObjectId(job_id)}, {"status": 1, "error_message": 1}
                    )
                    actual_status = updated_job.get("status") if updated_job else "failed"

                    if actual_status == "completed":
                        # Mark this video as completed
                        completed_count += 1
                        await database.topics.update_one(
                            {"_id": ObjectId(topic_id)},
                            {"$set": {
                                f"videos.{i}.status": "completed",
                                "progress": completed_count / total
                            }}
                        )
                        print(f"✅ Video {i+1}/{total} completed: {video_title}")
                    else:
                        # Pipeline marked job as failed internally
                        error_msg = updated_job.get("error_message", "Unknown error") if updated_job else "Unknown error"
                        print(f"❌ Video {i+1}/{total} failed (pipeline error): {video_title} — {error_msg}")
                        await database.topics.update_one(
                            {"_id": ObjectId(topic_id)},
                            {"$set": {f"videos.{i}.status": "failed"}}
                        )

                except Exception as e:
                    import traceback
                    print(f"❌ Video {i+1}/{total} failed: {video_title} — {e}")
                    traceback.print_exc()
                    await database.topics.update_one(
                        {"_id": ObjectId(topic_id)},
                        {"$set": {f"videos.{i}.status": "failed"}}
                    )
                    # Continue with next video even if one fails

            # All videos processed
            final_progress = completed_count / total if total > 0 else 1.0
            final_status = "completed" if completed_count == total else "partial"

            await self._update_topic(topic_id, {
                "status": final_status,
                "progress": final_progress,
                "completed_at": datetime.utcnow()
            })

            print(f"\n🎉 Topic {topic_id} processing complete: {completed_count}/{total} videos succeeded")

            # Phase 4: Generate topic-level curriculum summary
            if completed_count > 0:
                await self._generate_topic_summary(topic_id)

        except Exception as e:
            print(f"Error processing topic {topic_id}: {e}")
            await self._update_topic(topic_id, {
                "status": "failed",
                "error_message": str(e)
            })

    async def _build_playlist_context(self, topic_id: str, current_index: int) -> Optional[str]:
        """Build a context string from previously completed videos in this topic."""
        if current_index == 0:
            return None  # First video has no prior context

        database = db.get_db()
        topic = await database.topics.find_one({"_id": ObjectId(topic_id)})
        if not topic:
            return None

        videos = topic.get("videos", [])
        playlist_title = topic.get("title", "")
        total_videos = len(videos)

        context_parts = []
        context_parts.append(
            f"PLAYLIST CONTEXT: This video is Chapter {current_index + 1} of {total_videos} "
            f'in "{playlist_title}".'
        )
        context_parts.append("Previously covered chapters:")

        for i in range(current_index):
            v = videos[i]
            job_id = v.get("job_id")
            if not job_id or v.get("status") != "completed":
                continue

            # Fetch the completed job's summary
            job = await database.video_jobs.find_one(
                {"_id": ObjectId(job_id)},
                {"executive_summary": 1, "key_takeaways": 1, "video_name": 1}
            )
            if job:
                summary = job.get("executive_summary") or ""
                # Truncate to keep context manageable
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                takeaways = job.get("key_takeaways") or []
                takeaways = takeaways[:3]
                takeaways_str = "; ".join(takeaways) if takeaways else ""

                context_parts.append(
                    f'- Chapter {i+1}: "{v.get("video_title", "")}" — {summary}'
                )
                if takeaways_str:
                    context_parts.append(f"  Key concepts: {takeaways_str}")

        context_parts.append("")
        context_parts.append(
            "Use this context to: avoid re-explaining concepts already introduced, "
            "note when this video builds on prior material, and highlight what is NEW "
            "in this chapter."
        )

        return "\n".join(context_parts)

    async def _generate_topic_summary(self, topic_id: str):
        """Generate a curriculum-level summary after all videos are processed."""
        from services.gemini_service import gemini_service

        try:
            database = db.get_db()
            topic = await database.topics.find_one({"_id": ObjectId(topic_id)})
            if not topic:
                return

            videos = topic.get("videos", [])
            chapter_summaries = []

            for i, v in enumerate(videos):
                job_id = v.get("job_id")
                if not job_id or v.get("status") != "completed":
                    continue

                job = await database.video_jobs.find_one(
                    {"_id": ObjectId(job_id)},
                    {"executive_summary": 1, "key_takeaways": 1, "video_name": 1, "duration": 1}
                )
                if job:
                    chapter_summaries.append({
                        "chapter_number": i + 1,
                        "title": v.get("video_title") or job.get("video_name") or f"Chapter {i+1}",
                        "executive_summary": job.get("executive_summary") or "",
                        "key_takeaways": job.get("key_takeaways") or [],
                        "duration_minutes": (job.get("duration") or 0) / 60
                    })

            if not chapter_summaries:
                print(f"No completed chapter summaries to generate topic summary for {topic_id}")
                return

            print(f"\n📖 Generating curriculum summary for topic {topic_id} ({len(chapter_summaries)} chapters)...")
            
            result = await gemini_service.generate_topic_summary(
                topic_title=topic.get("title") or "",
                channel=topic.get("channel") or "",
                chapter_summaries=chapter_summaries
            )

            if not result:
                print(f"⚠️ Gemini returned empty result for topic summary {topic_id}")
                return

            # Store in topic document
            update_fields = {
                "topic_summary": result.get("topic_summary") or "",
                "learning_objectives": result.get("learning_objectives") or [],
                "prerequisites": result.get("prerequisites") or [],
                "difficulty_level": result.get("difficulty_level") or "",
                "estimated_total_time": result.get("estimated_total_time") or "",
                "chapter_outline": result.get("chapter_outline") or []
            }
            await self._update_topic(topic_id, update_fields)
            print(f"✅ Topic summary generated and stored for {topic_id}")

        except Exception as e:
            print(f"⚠️ Failed to generate topic summary for {topic_id}: {e}")
            # Don't fail the whole topic — summary is a bonus

    async def _update_topic(self, topic_id: str, updates: Dict):
        """Update topic in database."""
        database = db.get_db()
        updates["updated_at"] = datetime.utcnow()
        await database.topics.update_one(
            {"_id": ObjectId(topic_id)},
            {"$set": updates}
        )


# Singleton
playlist_service = PlaylistService()
