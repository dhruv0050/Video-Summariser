import subprocess
import os
from typing import List, Tuple
import config


class FFmpegUtils:
    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if FFmpeg is installed"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """Get video duration in seconds"""
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    
    @staticmethod
    def extract_audio(
        video_path: str, 
        output_path: str,
        sample_rate: int = config.AUDIO_SAMPLE_RATE
    ) -> str:
        """
        Extract audio from video as mono WAV file
        
        Args:
            video_path: Path to input video
            output_path: Path to output audio file
            sample_rate: Audio sample rate (default: 16000 Hz)
        
        Returns:
            Path to extracted audio file
        """
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    
    @staticmethod
    def split_audio(
        audio_path: str,
        chunk_duration: int = config.MAX_AUDIO_CHUNK_DURATION,
        overlap: int = config.AUDIO_OVERLAP_DURATION
    ) -> List[Tuple[str, float, float]]:
        """
        Split audio into overlapping chunks
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap duration in seconds
        
        Returns:
            List of tuples: (chunk_path, start_time, end_time)
        """
        # Get total duration
        duration = FFmpegUtils.get_video_duration(audio_path)
        
        chunks = []
        current_time = 0
        chunk_index = 0
        
        base_name = os.path.splitext(audio_path)[0]
        
        while current_time < duration:
            end_time = min(current_time + chunk_duration, duration)
            chunk_path = f"{base_name}_chunk_{chunk_index}.wav"
            
            # Extract chunk
            cmd = [
                'ffmpeg',
                '-i', audio_path,
                '-ss', str(current_time),
                '-t', str(end_time - current_time),
                '-acodec', 'copy',
                '-y',
                chunk_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            chunks.append((chunk_path, current_time, end_time))
            
            # Move forward, accounting for overlap
            current_time += chunk_duration - overlap
            chunk_index += 1
        
        return chunks
    
    @staticmethod
    def extract_keyframes(
        video_path: str,
        output_dir: str,
        interval: int = config.KEYFRAME_INTERVAL
    ) -> List[Tuple[str, float]]:
        """
        Extract keyframes at regular intervals
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            interval: Interval in seconds between frames
        
        Returns:
            List of tuples: (frame_path, timestamp)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        duration = FFmpegUtils.get_video_duration(video_path)
        frames = []
        
        current_time = 0
        frame_index = 0
        
        while current_time <= duration:
            frame_path = os.path.join(output_dir, f"frame_{frame_index:04d}.jpg")
            
            cmd = [
                'ffmpeg',
                '-ss', str(current_time),
                '-i', video_path,
                '-frames:v', '1',
                '-q:v', '2',  # High quality
                '-y',
                frame_path
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                frames.append((frame_path, current_time))
                frame_index += 1
            except subprocess.CalledProcessError:
                # Skip if frame extraction fails
                pass
            
            current_time += interval
        
        return frames
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# Check FFmpeg availability on module load
if not FFmpegUtils.check_ffmpeg():
    print("WARNING: FFmpeg not found. Please install FFmpeg to process videos.")
