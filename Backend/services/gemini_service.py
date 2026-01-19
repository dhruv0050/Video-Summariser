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
                    "key_points": ["point 1", "point 2"]
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
        
        Remember: Generate topics that span from 00:00:00 to {seconds_to_timestamp(duration)} to cover the entire video.
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
        
        def clean_json(s: str) -> str:
            """Clean common JSON formatting issues"""
            # Remove leading/trailing whitespace
            s = s.strip()
            
            # Remove comments (not standard JSON but LLMs sometimes add them)
            s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
            s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
            
            # Remove trailing commas before closing braces/brackets
            s = re.sub(r',\s*}', '}', s)
            s = re.sub(r',\s*]', ']', s)
            
            # Handle multiple spaces in strings
            s = re.sub(r' {2,}', ' ', s)
            
            # Escape unescaped newlines and tabs within string values
            lines = []
            in_string = False
            escaped = False
            i = 0
            while i < len(s):
                char = s[i]
                
                if char == '"' and not escaped:
                    in_string = not in_string
                    lines.append(char)
                elif char == '\\' and in_string:
                    escaped = not escaped
                    lines.append(char)
                elif in_string and not escaped:
                    # Handle special characters in strings
                    if char == '\n':
                        lines.append(' ')
                    elif char == '\r':
                        if i + 1 < len(s) and s[i+1] == '\n':
                            i += 1
                        lines.append(' ')
                    elif char == '\t':
                        lines.append(' ')
                    else:
                        lines.append(char)
                    escaped = False
                else:
                    lines.append(char)
                    escaped = False
                
                i += 1
            
            return ''.join(lines)
        
        try:
            # Try direct parse first
            return json.loads(text)
        except json.JSONDecodeError:
            # Extract JSON from markdown code block
            json_text = text
            
            # Try multiple extraction patterns - prioritize markdown code blocks
            patterns = [
                (r'```json\s*(.*?)\s*```', re.DOTALL),  # ```json ... ```
                (r'```\s*(.*?)\s*```', re.DOTALL),  # ``` ... ```
                (r'```\s*(.*?)\s*```', re.MULTILINE | re.DOTALL),  # Alternative markdown
            ]
            
            json_text = None
            for pattern, flags in patterns:
                try:
                    match = re.search(pattern, text, flags)
                    if match:
                        json_text = match.group(1).strip()
                        print(f"Extracted JSON from markdown code block (pattern: {pattern[:20]}...)")
                        break
                except Exception as e:
                    continue
            
            # If no markdown block found, try to find JSON object directly
            if not json_text:
                # Find first { ... } block
                start = text.find('{')
                if start >= 0:
                    depth = 0
                    for i, char in enumerate(text[start:], start):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                json_text = text[start:i+1]
                                print(f"Extracted JSON object directly")
                                break
            
            # Try cleaning and parsing
            if json_text:
                try:
                    cleaned = clean_json(json_text)
                    return json.loads(cleaned)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed after extraction: {e}")
                    # Fall through to last resort
            else:
                # No JSON found in markdown blocks, try direct extraction
                json_text = text
            
            # Last resort: try to parse cleaned original text
            try:
                cleaned = clean_json(json_text)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Last resort: try to find valid JSON subset
                try:
                    # Find first '{' and match to '}'
                    start = cleaned.find('{')
                    if start >= 0:
                        depth = 0
                        for i, char in enumerate(cleaned[start:], start):
                            if char == '{':
                                depth += 1
                            elif char == '}':
                                depth -= 1
                                if depth == 0:
                                    subset = cleaned[start:i+1]
                                    return json.loads(subset)
                except:
                    pass
                
                print(f"Failed to parse JSON after all attempts")
                print(f"Response preview (first 400 chars): {text[:400]}")
                return None


# Singleton instance
gemini_service = GeminiService()
