import os
import json
import time
import re
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from PIL import Image
import config
from models.video_job import TranscriptSegment, Topic, Frame

# Configure Gemini
genai.configure(api_key=config.GEMINI_API_KEY)

def timestamp_to_seconds(timestamp_str: str) -> float:
    """Convert HH:MM:SS format to seconds"""
    try:
        parts = timestamp_str.strip().split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        else:
            return float(parts[0])
    except:
        return 0.0

def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def retry_with_backoff(func, max_retries=3, initial_delay=2):
    """Retry a function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2 ** attempt)
            print(f"Retry {attempt + 1}/{max_retries} after {delay}s due to: {str(e)[:100]}")
            time.sleep(delay)


class GeminiService:
    def __init__(self):
        self.model_name = config.MODEL
        self.text_model = genai.GenerativeModel(self.model_name)
        # Use gemini-2.5-flash for Vision (higher quota limits)
        self.vision_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Genre mapping for fuzzy matching
        self.genre_mapping = {
            # Educational variations
            "educational": "educational_lecture",
            "educational_lecture": "educational_lecture",
            "educational_content": "educational_lecture",
            "educational_tutorial": "educational_lecture",
            "lecture": "educational_lecture",
            "tutorial": "educational_lecture",
            "course": "educational_lecture",
            "lesson": "educational_lecture",
            "training": "educational_lecture",
            
            # Podcast variations
            "podcast": "podcast_panel",
            "podcast_panel": "podcast_panel",
            "podcast_interview": "podcast_panel",
            "podcast_discussion": "podcast_panel",
            "panel_discussion": "podcast_panel",
            "roundtable": "podcast_panel",
            
            # Interview variations
            "interview": "interview_qna",
            "interview_qna": "interview_qna",
            "qna": "interview_qna",
            "question_answer": "interview_qna",
            "conversation": "interview_qna",
            
            # Vlog variations
            "vlog": "vlog",
            "vlog_personal": "vlog",
            "day_in_life": "vlog",
            "travel_vlog": "vlog",
            "lifestyle": "vlog",
            
            # Meeting variations
            "meeting": "meeting_presentation",
            "meeting_presentation": "meeting_presentation",
            "presentation": "meeting_presentation",
            "business_meeting": "meeting_presentation",
            "conference": "meeting_presentation",
            
            # Single speaker variations
            "single_speaker": "single_speaker_general",
            "single_speaker_general": "single_speaker_general",
            "monologue": "single_speaker_general",
            "talk": "single_speaker_general",
            "speech": "single_speaker_general",
        }
        
        # Prompt style snippets keyed by genre. These are appended to existing prompts
        # while keeping the output JSON schema unchanged.
        self.genre_prompt_snippets: Dict[str, Dict[str, str]] = {
            "podcast_panel": {
                "analysis": (
                    "Genre guidance: This is a podcast/panel with multiple speakers. "
                    "Prefer topics organized by discussion segments, speaker turns, questions, and debates. "
                    "Capture noteworthy quotes and disagreements. Avoid assuming slides unless mentioned."
                ),
                "synthesis": (
                    "Genre guidance: Podcast/panel. Emphasize key arguments by different speakers, "
                    "consensus vs dissent, and notable quotes. Keep it conversational and accurate."
                ),
            },
            "educational_lecture": {
                "analysis": (
                    "Genre guidance: Educational lecture/tutorial. Prefer chaptering by concepts, "
                    "definitions, examples, steps, and recap. If slides/demos are likely, mark visual cues."
                ),
                "synthesis": (
                    "Genre guidance: Educational. Emphasize learning objectives, step-by-step breakdowns, "
                    "definitions, examples, and actionable study takeaways."
                ),
            },
            "vlog": {
                "analysis": (
                    "Genre guidance: Vlog. Prefer segments by locations/activities/time-of-day changes. "
                    "Summaries should reflect narrative flow and key moments rather than formal chapters."
                ),
                "synthesis": (
                    "Genre guidance: Vlog. Emphasize storyline, highlights, places/activities, and memorable moments."
                ),
            },
            "single_speaker_general": {
                "analysis": (
                    "Genre guidance: Single-speaker general talk (non-educational). "
                    "Prefer segments by topics, anecdotes, opinions, and conclusions."
                ),
                "synthesis": (
                    "Genre guidance: Single-speaker general. Emphasize main points, opinions, and memorable quotes."
                ),
            },
            "interview_qna": {
                "analysis": (
                    "Genre guidance: Interview/Q&A. Prefer segments by questions and answers. "
                    "Clearly identify the question context and the answer summary."
                ),
                "synthesis": (
                    "Genre guidance: Interview/Q&A. Emphasize key questions, concise answers, and notable quotes."
                ),
            },
            "meeting_presentation": {
                "analysis": (
                    "Genre guidance: Meeting/presentation. Prefer segments by agenda items, decisions, action items, "
                    "and key updates. Capture commitments and owners if present."
                ),
                "synthesis": (
                    "Genre guidance: Meeting/presentation. Emphasize decisions, action items, and summary of updates."
                ),
            },
            "unknown": {
                "analysis": "Genre guidance: Unknown. Use a neutral, general chaptering approach.",
                "synthesis": "Genre guidance: Unknown. Use a neutral summary approach.",
            },
        }
    
    def _normalize_genre(self, genre_raw: str) -> str:
        """Normalize genre string with fuzzy matching"""
        if not isinstance(genre_raw, str):
            return "unknown"
        
        genre_lower = genre_raw.lower().strip()
        
        # Direct match
        if genre_lower in self.genre_mapping:
            return self.genre_mapping[genre_lower]
        
        # Fuzzy matching - check if any key is contained in the genre string
        for key, value in self.genre_mapping.items():
            if key in genre_lower or genre_lower in key:
                return value
        
        # Check for keywords
        if any(word in genre_lower for word in ["educational", "lecture", "tutorial", "course", "lesson"]):
            return "educational_lecture"
        elif any(word in genre_lower for word in ["podcast", "panel", "discussion", "roundtable"]):
            return "podcast_panel"
        elif any(word in genre_lower for word in ["interview", "qna", "question", "conversation"]):
            return "interview_qna"
        elif any(word in genre_lower for word in ["vlog", "day", "life", "travel", "lifestyle"]):
            return "vlog"
        elif any(word in genre_lower for word in ["meeting", "presentation", "business", "conference"]):
            return "meeting_presentation"
        elif any(word in genre_lower for word in ["single", "monologue", "talk", "speech"]):
            return "single_speaker_general"
        
        return "unknown"

    def _genre_snippet(self, genre: Optional[str], key: str) -> str:
        g = (genre or "unknown").strip() if genre else "unknown"
        if g not in self.genre_prompt_snippets:
            g = "unknown"
        return self.genre_prompt_snippets[g].get(key, "")

    async def classify_video_genre(
        self,
        transcript_text: str,
        duration: float,
    ) -> Dict[str, Any]:
        """
        Classify video genre based on transcript (fast, small prompt).

        Returns:
            { "genre": str, "confidence": float, "reason": str }
        """
        def _classify():
            # Keep this small and fast: only use a slice of transcript
            sample = transcript_text[:8000]
            prompt = f"""
You are classifying the genre of a video from a transcript sample.
Video duration: {seconds_to_timestamp(duration)}.

Pick ONE best genre from this list (return exactly one key as 'genre'):
- podcast_panel (multiple speakers, conversational)
- educational_lecture (single speaker teaching/tutorial)
- interview_qna (interviewer + guest Q&A)
- vlog (personal day-in-life / travel / activities)
- meeting_presentation (work/meeting/agenda/action-items)
- single_speaker_general (single speaker talk, non-educational)
- unknown

Transcript sample:
{sample}

Return ONLY valid JSON:
{{
  "genre": "educational_lecture",
  "confidence": 0.0,
  "reason": "Short reason based on transcript cues"
}}
"""
            resp = self.text_model.generate_content(prompt)
            parsed = self._parse_json_response(resp.text) or {}
            genre_raw = parsed.get("genre", "unknown")
            confidence = parsed.get("confidence", 0.0)
            reason = parsed.get("reason", "")
            
            # Normalize genre with fuzzy matching
            genre = self._normalize_genre(genre_raw)
            
            # Basic normalization
            if not isinstance(confidence, (int, float)):
                confidence = 0.0
            if not isinstance(reason, str):
                reason = ""
            
            print(f"Detected genre: {genre} (raw: {genre_raw}, confidence={confidence:.2f})")
            return {"genre": genre, "confidence": float(confidence), "reason": reason}

        try:
            return retry_with_backoff(_classify, max_retries=2, initial_delay=1) or {
                "genre": "unknown",
                "confidence": 0.0,
                "reason": "",
            }
        except Exception as e:
            print(f"Genre classification failed: {e}")
            return {"genre": "unknown", "confidence": 0.0, "reason": ""}

    async def detect_transcript_visual_cues(
        self,
        transcript_segments: List[TranscriptSegment]
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: The "Audio Cue" Scout
        Identifies timestamps where speaker references visuals.
        """
        if not transcript_segments:
            return []

        # Prepare transcript with timestamps
        formatted_transcript = ""
        for seg in transcript_segments:
            start_ts = seconds_to_timestamp(seg.start_time)
            formatted_transcript += f"[{start_ts}] {seg.text}\n"

        # If transcript is huge, we might need to chunk it, but for now let's try strict truncation or send it all if model supports
        # Gemini 1.5 Flash has large context, so ~1 hour transcript should fit.
        # We'll take the first 30k chars for safety/speed in this scout phase if needed, 
        # but ideal is full context. Let's send it all but watch for limits.
        
        def _scout():
            prompt = f"""
You are a Video Editor Assistant. Your task is to identify specific timestamps in the transcript where the speaker explicitly references visual information being shown on screen.

Look for cues such as:
- "As you can see on this chart..."
- "Looking at this graph..."
- "If we turn to the next slide..."
- "Here in the code..."
- "This diagram illustrates..."

Transcript Segment:
{formatted_transcript}

Return a JSON object with a list of "visual_cues":
{{
  "visual_cues": [
    {{
      "timestamp": "00:04:23",
      "cue_phrase": "As shown in this bar chart",
      "confidence": "high",
      "expected_visual_type": "chart"
    }}
  ]
}}
// expected_visual_type options: slide, demo, code, diagram, unknown

If no cues are found, return an empty list.
"""
            print("Running Audio Cue Scout...")
            response = self.text_model.generate_content(prompt)
            result = self._parse_json_response(response.text)
            return result.get("visual_cues", []) if result else []

        try:
            return retry_with_backoff(_scout, max_retries=2)
        except Exception as e:
            print(f"Audio Cue Scout failed: {e}")
            return []

    async def evaluate_frame_content(
        self,
        frame_path: str
    ) -> Dict[str, Any]:
        """
        Phase 1: The "Gatekeeper"
        Classify frame as Slide/Demo/Diagram (Useful) or Person/Other (Junk).
        """
        def _gatekeep():
            img = Image.open(frame_path)
            
            prompt = """
Analyze this video frame. Your goal is to determine if this frame contains valuable static information (like a presentation slide, coding terminal, or data dashboard) or if it is generic footage (like a person talking or a transition).

Classify the image into one of these categories:
1. "slide_presentation" (PowerPoint, Keynote)
2. "software_demo" (IDE, Dashboard, Browser)
3. "technical_diagram" (Whiteboard, Architecture)
4. "talking_head" (Person on camera)
5. "other"

Return JSON:
{
  "category": "slide_presentation",
  "information_density": "high", // options: high, medium, low, none
  "contains_text": true,
  "is_useful": true // Set to false if it's blurry, a transition, or just a person
}
"""
            # Using Vision model (Flash)
            response = self.vision_model.generate_content([prompt, img])
            result = self._parse_json_response(response.text)
            
            if not result:
                return {
                    "category": "other",
                    "information_density": "none",
                    "contains_text": False,
                    "is_useful": False
                }
            return result

        try:
            return retry_with_backoff(_gatekeep, max_retries=2)
        except Exception as e:
            print(f"Gatekeeper analysis failed for {frame_path}: {e}")
            return {
                "category": "error",
                "information_density": "none",
                "contains_text": False,
                "is_useful": False
            }
    
    async def transcribe_audio(
        self, 
        audio_path: str,
        start_time: float = 0
    ) -> List[TranscriptSegment]:
        """
        Transcribe audio using Gemini with retry logic
        
        Args:
            audio_path: Path to audio file
            start_time: Start time offset for this chunk
        
        Returns:
            List of transcript segments
        """
        def _transcribe():
            # Upload audio file
            print(f"Uploading audio chunk starting at {start_time}s...")
            
            # Use the correct API for google-generativeai >= 0.4.0
            with open(audio_path, 'rb') as audio_file_obj:
                audio_file = genai.upload_file(
                    path=audio_path,
                    mime_type="audio/wav"
                )
            
            prompt = """
            Transcribe this audio with speaker diarization. 
            Label speakers as Speaker A, Speaker B, etc.
            
            Return the transcription in the following JSON format:
            {
                "segments": [
                    {
                        "text": "transcribed text",
                        "start_time": 0.0,
                        "end_time": 5.2,
                        "speaker": "Speaker A"
                    }
                ]
            }
            
            Provide accurate timestamps in seconds relative to the start of this audio clip.
            """
            
            print("Sending to Gemini for transcription...")
            response = self.text_model.generate_content([prompt, audio_file])
            
            # Parse response
            result = self._parse_json_response(response.text)
            
            segments = []
            if result and "segments" in result:
                for seg in result["segments"]:
                    segments.append(TranscriptSegment(
                        text=seg.get("text", ""),
                        start_time=start_time + seg.get("start_time", 0),
                        end_time=start_time + seg.get("end_time", 0),
                        speaker=seg.get("speaker"),
                        confidence=seg.get("confidence", 0.9)
                    ))
            
            return segments
        
        try:
            return retry_with_backoff(_transcribe, max_retries=3)
        except Exception as e:
            print(f"Error transcribing audio after retries: {e}")
            # Fallback: simple transcription without timestamps
            return await self._simple_transcribe(audio_path, start_time)
    
    async def _simple_transcribe(
        self, 
        audio_path: str, 
        start_time: float
    ) -> List[TranscriptSegment]:
        """Fallback simple transcription"""
        try:
            audio_file = genai.upload_file(audio_path)
            prompt = "Transcribe this audio verbatim. Identify different speakers if possible."
            
            response = self.text_model.generate_content([prompt, audio_file])
            
            return [TranscriptSegment(
                text=response.text,
                start_time=start_time,
                end_time=start_time + 300,  # Approximate
                speaker="Speaker A",
                confidence=0.8
            )]
        except Exception as e:
            print(f"Error in simple transcription: {e}")
            return []
    
    async def analyze_transcript(
        self, 
        transcript_text: str,
        duration: float,
        video_genre: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze transcript to extract topics, key moments, etc. with retry logic
        Split large transcripts into chunks to avoid timeouts
        
        Args:
            transcript_text: Full transcript text
            duration: Video duration in seconds
        
        Returns:
            Dictionary with topics, key moments, entities
        """
        # If transcript is very large, analyze in chunks but always provide full duration context
        max_chars = 50000  # ~12k tokens
        
        if len(transcript_text) > max_chars:
            print(f"Large transcript ({len(transcript_text)} chars), analyzing in chunks...")
            # Split into manageable chunks
            words = transcript_text.split()
            chunk_size = len(words) // 3  # Split into ~3 chunks
            chunks = [
                ' '.join(words[i:i+chunk_size]) 
                for i in range(0, len(words), chunk_size)
            ]
            
            # Analyze each chunk but provide full duration context
            all_topics = []
            all_entities = {"people": [], "companies": [], "concepts": [], "tools": []}
            all_takeaways = []
            all_visual_cues = []
            
            for idx, chunk in enumerate(chunks):
                print(f"Analyzing chunk {idx+1}/{len(chunks)} (full duration: {duration/60:.1f} min)...")
                # Calculate approximate time range for this chunk (for reference)
                chunk_start_approx = (idx / len(chunks)) * duration
                chunk_end_approx = ((idx + 1) / len(chunks)) * duration
                
                result = await self._analyze_transcript_chunk(
                    chunk, 
                    duration, 
                    idx, 
                    len(chunks),
                    chunk_start_approx,
                    chunk_end_approx,
                    video_genre=video_genre
                )
                if result:
                    # Always provide full duration context, so each chunk should generate topics for full video
                    # But we'll merge and deduplicate
                    chunk_topics = result.get("topics", [])
                    all_topics.extend(chunk_topics)
                    # Merge entities
                    for key in all_entities:
                        all_entities[key].extend(result.get("entities", {}).get(key, []))
                    all_takeaways.extend(result.get("key_takeaways", []))
                    all_visual_cues.extend(result.get("visual_cues", []))
            
            # Deduplicate entities
            for key in all_entities:
                all_entities[key] = list(set(all_entities[key]))
            
            # Deduplicate topics by timestamp range (merge overlapping/duplicate topics)
            deduplicated_topics = self._deduplicate_topics(all_topics, duration)
            
            return {
                "topics": deduplicated_topics,
                "entities": all_entities,
                "key_takeaways": list(set(all_takeaways)),
                "visual_cues": all_visual_cues
            }
        else:
            return await self._analyze_transcript_chunk(transcript_text, duration, 0, 1, 0.0, duration, video_genre=video_genre)
    
    def _deduplicate_topics(self, topics: List[Dict], duration: float) -> List[Dict]:
        """Deduplicate and merge overlapping topics"""
        if not topics:
            return []
        
        # Sort topics by start time
        sorted_topics = sorted(topics, key=lambda t: timestamp_to_seconds(t.get("timestamp_range", ["00:00:00"])[0]))
        
        deduplicated = []
        for topic in sorted_topics:
            ts_range = topic.get("timestamp_range", [])
            if len(ts_range) < 2:
                continue
            
            start_ts = timestamp_to_seconds(ts_range[0])
            end_ts = timestamp_to_seconds(ts_range[1])
            
            # Skip if this topic overlaps significantly with the last one (>70% overlap)
            if deduplicated:
                last_topic = deduplicated[-1]
                last_start = timestamp_to_seconds(last_topic.get("timestamp_range", [])[0])
                last_end = timestamp_to_seconds(last_topic.get("timestamp_range", [])[1])
                
                overlap_start = max(start_ts, last_start)
                overlap_end = min(end_ts, last_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                last_duration = last_end - last_start
                
                if last_duration > 0 and overlap_duration / last_duration > 0.7:
                    # Merge topics by keeping the longer one or the one with more key points
                    if len(topic.get("key_points", [])) > len(last_topic.get("key_points", [])):
                        deduplicated[-1] = topic
                    continue
            
            deduplicated.append(topic)
        
        return deduplicated
    
    async def _analyze_transcript_chunk(
        self,
        transcript_text: str,
        duration: float,
        chunk_idx: int,
        total_chunks: int,
        chunk_start_time: float = None,
        chunk_end_time: float = None,
        video_genre: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze a single transcript chunk"""
        def _analyze():
            chunk_info = f" (part {chunk_idx+1}/{total_chunks})" if total_chunks > 1 else ""
            time_info = ""
            if chunk_start_time is not None and chunk_end_time is not None:
                start_ts = seconds_to_timestamp(chunk_start_time)
                end_ts = seconds_to_timestamp(chunk_end_time)
                time_info = f"\n\nIMPORTANT: This transcript chunk covers video time {start_ts} to {end_ts} out of total duration {seconds_to_timestamp(duration)}."
            
            genre_snippet = self._genre_snippet(video_genre, "analysis")
            prompt = f"""
        Analyze this video transcript{chunk_info} (total video duration: {duration/60:.1f} minutes = {seconds_to_timestamp(duration)}) and extract topics that span the ENTIRE video duration.
        
        CRITICAL: You must analyze the transcript and generate topics with timestamps that cover the FULL video duration from 00:00:00 to {seconds_to_timestamp(duration)}. Do not stop at just the beginning or middle - ensure topics are distributed throughout the entire video.{time_info}

        {genre_snippet}
        
        Extract the following:
        
        1. Topic segmentation: Break the video into logical chapters/sections with start/end timestamps covering the ENTIRE duration (00:00:00 to {seconds_to_timestamp(duration)})
           - Each topic should have clear start and end timestamps
           - Topics should progress chronologically through the video
           - Ensure topics cover from the start to the end of the video
           - CRITICAL: Identify and label any sponsorship, advertisement, or pure promotional segments as "ad". Label educational/main content as "content".
        2. Key moments: Important phrases that likely reference visuals ("as shown", "this slide", etc.)
        3. Named entities: People, companies, tools, concepts mentioned
        4. Key takeaways: Main insights from the content
        
        Transcript:
        {transcript_text}
        
        Return analysis in this JSON format (ensure topics cover the full video duration):
        {{
            "topics": [
                {{
                    "title": "Topic title",
                    "timestamp_range": ["00:00:00", "00:15:30"],
                    "summary": "Brief summary",
                    "key_points": ["point 1", "point 2"],
                    "type": "content" // "content" or "ad"
                }}
            ],
            "visual_cues": [
                {{
                    "timestamp": "00:05:23",
                    "cue_text": "as you can see on this slide",
                    "context": "surrounding context"
                }}
            ],
            "entities": {{
                "people": ["name1", "name2"],
                "companies": ["company1"],
                "concepts": ["concept1", "concept2"],
                "tools": ["tool1"]
            }},
            "key_takeaways": ["takeaway 1", "takeaway 2"]
        }}
        
        Remember: Generate topics that span from 00:00:00 to {seconds_to_timestamp(duration)} to cover the entire video. Mark sponsorships as "ad".
        """
            
            print(f"Analyzing transcript chunk {chunk_idx+1}/{total_chunks} with Gemini...")
            response = self.text_model.generate_content(prompt)
            result = self._parse_json_response(response.text)
            return result or {}
        
        try:
            return retry_with_backoff(_analyze, max_retries=3)
        except Exception as e:
            print(f"Error analyzing transcript after retries: {e}")
            return {}
    
    async def analyze_frames(
        self, 
        frame_paths: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple frames with Gemini Vision
        
        Args:
            frame_paths: List of paths to image files
            context: Optional context about these frames
        
        Returns:
            List of frame analyses
        """
        results = []
        
        # Process in smaller batches of 2 for better reliability
        batch_size = 2
        for i in range(0, len(frame_paths), batch_size):
            batch = frame_paths[i:i+batch_size]
            
            try:
                batch_result = await self._analyze_frame_batch(batch, context)
                results.extend(batch_result)
            except Exception as e:
                print(f"Error analyzing frame batch at index {i}: {str(e)[:100]}")
                # Add placeholder results so job completes
                for path in batch:
                    results.append({
                        "frame_path": path,
                        "description": "Analysis failed but processing continued",
                        "ocr_text": "",
                        "type": "unknown",
                        "insights": ""
                    })
        
        return results
    
    async def _analyze_frame_batch(
        self,
        frame_paths: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Analyze a batch of frames together with retry logic"""
        def _analyze():
            images = []
            for path in frame_paths:
                img = Image.open(path)
                images.append(img)
            
            context_text = f"\nContext: {context}" if context else ""
            
            prompt = f"""
        Analyze these video frames and for each frame provide:
        1. Semantic description (what's shown - slides, diagrams, people, demos, etc.)
        2. OCR: Extract all visible text
        3. Type: Classify as "slide", "diagram", "chart", "demo", "person", "other"
        4. Key insights: What information does this frame convey?
        
        {context_text}
        
        Return analysis in this JSON format (avoid trailing commas):
        {{
            "frames": [
                {{
                    "frame_index": 0,
                    "description": "Slide showing framework diagram",
                    "ocr_text": "extracted text here",
                    "type": "slide",
                    "insights": "Key concepts being presented"
                }}
            ]
        }}
        """
            
            content = [prompt] + images
            print(f"Analyzing batch of {len(images)} frames...")
            response = self.vision_model.generate_content(content)
            
            result = self._parse_json_response(response.text)
            
            analyses = []
            if result and "frames" in result:
                for i, frame_data in enumerate(result["frames"]):
                    if i < len(frame_paths):
                        analyses.append({
                            "frame_path": frame_paths[i],
                            "description": frame_data.get("description", ""),
                            "ocr_text": frame_data.get("ocr_text", ""),
                            "type": frame_data.get("type", "other"),
                            "insights": frame_data.get("insights", "")
                        })
            
            return analyses
        
        try:
            return retry_with_backoff(_analyze, max_retries=3)
        except Exception as e:
            print(f"Error analyzing frame batch after retries: {e}")
            # Return placeholder results
            return [{
                "frame_path": path,
                "description": "Analysis failed",
                "ocr_text": "",
                "type": "unknown",
                "insights": ""
            } for path in frame_paths]
    
    async def analyze_frame_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """
        Analyze clusters of frames to select hero frame and extract topic.
        
        Args:
            clusters: List of clusters from ImageProcessor
            
        Returns:
            List of visual subtopics (title, summary, hero_frame_path)
        """
        results = []
        
        print(f"Analyzing {len(clusters)} visual clusters with Gemini Vision...")
        
        for i, cluster in enumerate(clusters):
            # Limit candidates to top 5 sharpest frames
            candidates = cluster.get('candidates', [])[:5] 
            frame_paths = [c['path'] for c in candidates]
            
            # Prepare images
            image_parts = []
            valid_candidates = []
            
            for idx, path in enumerate(frame_paths):
                try:
                    img = Image.open(path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    image_parts.append(img)
                    valid_candidates.append(candidates[idx])
                except:
                    continue
            
            if not image_parts:
                continue
            
            # Use timestamps for context
            start_ts = seconds_to_timestamp(cluster.get('start_time', 0))
            end_ts = seconds_to_timestamp(cluster.get('end_time', 0))
                
            prompt = f"""
            I am providing you with {len(image_parts)} frames captured within a processing window from {start_ts} to {end_ts}. These likely represent the same slide or visual element, potentially with slight animations or cursor movements.
            
            Task 1: Select the "Hero Frame". This is the frame that is most focused, least blurry, and contains the most complete information (e.g., the full list is revealed, or the slide build is complete).
            Task 2: Extract the title or main heading from that frame.
            Task 3: Summarize the specific data or concept shown in that frame (do not summarize the audio, only what is VISIBLE).

            Return JSON:
            {{
              "hero_frame_index": 0, // The index of the selected best image (0 to {len(image_parts)-1})
              "sub_topic_title": "Slide Title",
              "visual_summary": "Description of the visual content (chart trends, code purpose, diagram flow)",
              "ocr_keywords": ["keyword1", "keyword2"]
            }}
            """
            
            def _analyze_single_cluster():
                # print(f"Analyzing cluster {i+1}/{len(clusters)} with {len(image_parts)} frames...")
                content = [prompt] + image_parts
                response = self.vision_model.generate_content(content)
                return self._parse_json_response(response.text)

            try:
                # Use retry logic for robustness
                parsed = retry_with_backoff(_analyze_single_cluster, max_retries=2, initial_delay=1)
                
                if parsed:
                    idx = parsed.get("hero_frame_index", 0)
                    # Validate index
                    if not isinstance(idx, int) or idx < 0 or idx >= len(valid_candidates):
                        idx = 0
                        
                    hero_frame = valid_candidates[idx]
                    
                    results.append({
                        "timestamp": hero_frame['timestamp'],
                        "hero_frame_path": hero_frame['path'],
                        "sub_topic_title": parsed.get("sub_topic_title", "Visual Topic"),
                        "visual_summary": parsed.get("visual_summary", ""),
                        "ocr_keywords": parsed.get("ocr_keywords", []),
                        "frame_count": cluster.get('frame_count', 1),
                        "cluster_idx": i
                    })
                    print(f"Cluster {i+1}/{len(clusters)}: Hero Frame Selected (Index {idx}). Title: {parsed.get('sub_topic_title')}")
                else:
                    print(f"Cluster {i+1}: Failed to parse response")
                    
            except Exception as e:
                print(f"Error analyzing cluster {i+1}: {e}")
                continue
                
        return results

    async def map_visuals_to_topics(
        self,
        main_topics: List[Dict],
        visual_subtopics: List[Dict]
    ) -> List[Dict]:
        """
        Intelligently map visual sub-topics to main transcript topics using LLM.
        """
        if not main_topics or not visual_subtopics:
            return main_topics
            
        print(f"Mapping {len(visual_subtopics)} visual subtopics to {len(main_topics)} main topics...")
        
        # Prepare minimal data for prompt
        simple_main_topics = [{
            "title": t.get("title"),
            "timestamp_range": t.get("timestamp_range")
        } for t in main_topics]
        
        simple_visuals = [{
            "title": v.get("sub_topic_title"),
            "visual_summary": v.get("visual_summary"),
            "timestamp": seconds_to_timestamp(v.get("timestamp", 0)),
            "original_index": i
        } for i, v in enumerate(visual_subtopics)]
        
        prompt = f"""
        You are a Report Structuring Engine. I have a list of "Main Topics" derived from the audio transcript, and a list of "Visual Sub-Topics" derived from analyzing screenshots.

        Your task is to nest the Visual Sub-Topics under the correct Main Topic based on their timestamps.

        Rules:
        1. A Visual Sub-Topic belongs to a Main Topic if its timestamp falls within the Main Topic's start/end range.
        2. If a Main Topic has more than 3 visual sub-topics, select the 3 most distinct ones based on their titles and summaries to avoid repetition.
        3. If a visual doesn't fit any main topic perfectly, fit it to the nearest logical topic.
        4. CRITICAL: If a Visual Sub-Topic is clearly an advertisement, sponsorship, or unrelated promotion (e.g. brand logos, 'buy now', 'subscribe'), DISCARD IT. Do not map it to any topic.

        Input Data:
        Main Topics: {json.dumps(simple_main_topics, indent=2)}
        Visual Sub-Topics: {json.dumps(simple_visuals, indent=2)}

        Return the Final JSON Structure:
        {{
          "topics": [
            {{
              "title": "Main Topic Title", 
              "sub_topics": [
                {{
                  "title": "Visual Sub-Topic Title", 
                  "visual_summary": "Summary...", 
                  "timestamp": "HH:MM:SS",
                  "original_index": 0 
                }}
              ]
            }}
          ]
        }}
        """
        
        def _map_topics():
            response = self.text_model.generate_content(prompt)
            return self._parse_json_response(response.text)
            
        try:
            mapped_result = retry_with_backoff(_map_topics, max_retries=2)
            
            if mapped_result and "topics" in mapped_result:
                # Merge logic
                # We want to preserve the FULL original main_topic data (which has summaries etc)
                # and attach sub_topics to it.
                
                # Clone main_topics to avoid mutating original list during iteration
                import copy
                final_topics = copy.deepcopy(main_topics)
                
                # Create lookup for result topics
                result_topic_map = {t.get("title"): t for t in mapped_result["topics"]}
                
                for topic in final_topics:
                    mapped = result_topic_map.get(topic.get("title"))
                    if mapped:
                        sub_topics_data = []
                        for sub in mapped.get("sub_topics", []):
                            idx = sub.get("original_index")
                            if idx is not None and 0 <= idx < len(visual_subtopics):
                                # Get full visual data
                                visual = visual_subtopics[idx]
                                sub_topics_data.append({
                                    "title": sub.get("title", visual.get("sub_topic_title")),
                                    "visual_summary": sub.get("visual_summary", visual.get("visual_summary")),
                                    "timestamp": sub.get("timestamp", seconds_to_timestamp(visual.get("timestamp", 0))),
                                    "image_url": None, # Will be filled by pipeline using frame_path map
                                    "frame_timestamp": visual.get("timestamp", 0) # Keep seconds for linking
                                })
                        topic["sub_topics"] = sub_topics_data
                    else:
                        topic["sub_topics"] = []
                
                return final_topics
            else:
                return self._fallback_map_topics(main_topics, visual_subtopics)
                
        except Exception as e:
            print(f"Error mapping topics: {e}")
            return self._fallback_map_topics(main_topics, visual_subtopics)

    def _fallback_map_topics(self, main_topics, visual_subtopics):
        """Simple timestamp-based mapping fallback"""
        print("Using fallback timestamp mapping...")
        import copy
        final_topics = copy.deepcopy(main_topics)
        
        for topic in final_topics:
            topic["sub_topics"] = []
            
            ts_start = topic.get("timestamp_range", ["00:00:00"])[0]
            ts_end = topic.get("timestamp_range", ["00:00:00", "23:59:59"])[1]
            
            start_time = timestamp_to_seconds(ts_start)
            end_time = timestamp_to_seconds(ts_end)
            
            for v in visual_subtopics:
                ts = v.get("timestamp", 0)
                if start_time <= ts <= end_time:
                     topic["sub_topics"].append({
                        "title": v.get("sub_topic_title"),
                        "visual_summary": v.get("visual_summary"),
                        "timestamp": seconds_to_timestamp(ts),
                        "frame_timestamp": ts,
                        "image_url": None
                    })
            
            # Limit to 3 per topic
            topic["sub_topics"] = topic["sub_topics"][:3]
            
        return final_topics

    async def synthesize_results(
        self,
        transcript_analysis: Dict[str, Any],
        frame_analyses: List[Dict[str, Any]],
        duration: float,
        video_genre: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize transcript and frame analyses into final output with retry logic
        
        Args:
            transcript_analysis: Analysis from transcript
            frame_analyses: Analyses from frames
            duration: Video duration
        
        Returns:
            Structured output with topics, summary, etc.
        """
        def _synthesize():
            # Get all topics from transcript analysis - preserve ALL of them
            all_topics = transcript_analysis.get("topics", [])
            
            # Ensure topics cover full duration - if not, keep original topics from analysis
            topics_covering_full_duration = all_topics
            
            # Create a compact version for synthesis summary generation
            topics_preview = json.dumps(all_topics[:10])[:3000] if len(all_topics) > 10 else json.dumps(all_topics)[:3000]
            frames_preview = json.dumps(frame_analyses[:15])[:3000] if len(frame_analyses) > 15 else json.dumps(frame_analyses)[:3000]
            
            genre_snippet = self._genre_snippet(video_genre, "synthesis")
            prompt = f"""
        You are synthesizing analysis of a {duration/60:.1f}-minute video (duration: {seconds_to_timestamp(duration)}).
        
        IMPORTANT: You must preserve ALL topics from the transcript analysis. Do not filter, remove, or skip any topics. 
        All topics should cover the full video duration from 00:00:00 to {seconds_to_timestamp(duration)}.

        {genre_snippet}
        
        Transcript Topics ({len(all_topics)} total - preserve ALL of them):
        {topics_preview}
        
        Key Frames ({len(frame_analyses)} total):
        {frames_preview}
        
        Your task:
        1. Generate an executive summary (3-5 sentences) covering the ENTIRE video
        2. PRESERVE ALL topics from transcript analysis - do not filter or remove any
        3. Ensure topics span the full video duration (00:00:00 to {seconds_to_timestamp(duration)})
        4. Extract actionable insights and key takeaways
        5. List entities mentioned (companies, concepts, tools)
        
        Return ONLY valid JSON (no trailing commas or newlines in strings):
        {{
            "executive_summary": "Clear summary covering the entire video...",
            "topics": [
                {{
                    "title": "Topic title",
                    "timestamp_range": ["00:00:00", "00:15:30"],
                    "summary": "Single line summary",
                    "key_points": ["point 1", "point 2"]
                }}
            ],
            "key_takeaways": ["takeaway 1", "takeaway 2"],
            "entities": {{
                "companies": ["name1"],
                "concepts": ["concept1"],
                "tools": ["tool1"]
            }}
        }}
        
        CRITICAL: Include ALL {len(all_topics)} topics in your response. Topics must cover from 00:00:00 to {seconds_to_timestamp(duration)}.
        """
            
            print(f"Synthesizing results with Gemini (preserving {len(all_topics)} topics)...")
            response = self.text_model.generate_content(prompt)
            result = self._parse_json_response(response.text)
            
            # If synthesis returns fewer topics than original, prefer original topics
            synthesized_topics = result.get("topics", []) if result else []
            if len(synthesized_topics) < len(all_topics) * 0.8:  # If we lost >20% of topics
                print(f"Warning: Synthesis returned {len(synthesized_topics)} topics but original had {len(all_topics)}. Using original topics.")
                synthesized_topics = all_topics
            
            # Merge: use synthesized topics if they cover full duration, otherwise use original
            final_topics = synthesized_topics if synthesized_topics else all_topics
            
            return {
                "executive_summary": result.get("executive_summary", "") if result else "Video processing completed.",
                "topics": final_topics,
                "key_takeaways": result.get("key_takeaways", transcript_analysis.get("key_takeaways", [])) if result else transcript_analysis.get("key_takeaways", []),
                "entities": result.get("entities", transcript_analysis.get("entities", {})) if result else transcript_analysis.get("entities", {})
            }
        
        try:
            return retry_with_backoff(_synthesize, max_retries=2)
        except Exception as e:
            print(f"Error synthesizing results: {e}")
            # Return fallback with ALL original topics from analysis
            return {
                "executive_summary": "Video processing completed but synthesis had errors.",
                "topics": transcript_analysis.get("topics", []),
                "key_takeaways": transcript_analysis.get("key_takeaways", []),
                "entities": transcript_analysis.get("entities", {})
            }
    
    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Extract and parse JSON from model response with aggressive error recovery"""
        import re
        import json
        
        def _repair_json(json_str: str) -> str:
            """Repair common JSON syntax errors"""
            # Remove comments
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            # Remove trailing commas
            json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
            
            # Fix unescaped quotes inside strings (this is tricky and heuristics-based)
            # We assume keys are always double-quoted and followed by a colon
            # This is a dangerous regex but handles many common cases
            # It looks for "key": "val" patterns and tries to identify unescaped quotes in val
            
            return json_str

        try:
            # 1. Try to find JSON block in markdown
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # 2. Try to find the first '{' and last '}'
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    json_str = text[start:end+1]
                else:
                    json_str = text

            # 3. Clean and Parse
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 4. Repair and Retry
                repaired = _repair_json(json_str)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    # 5. Last resort: specific fix for newlines in strings
                    # LLMs often put real newlines inside strings which is invalid JSON
                    repaired_newlines = re.sub(r'(?<=["\w])\n(?=["\w])', '\\n', json_str)
                    return json.loads(repaired_newlines)
                    
        except Exception as e:
            print(f"JSON parsing failed: {str(e)}")
            # print(f"Failed JSON text subset: {text[:200]}...")
            return None


# Singleton instance
gemini_service = GeminiService()
