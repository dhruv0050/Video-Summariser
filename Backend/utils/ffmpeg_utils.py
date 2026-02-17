import subprocess
import os
import re
from typing import List, Tuple
import config

# Use imageio-ffmpeg's bundled FFmpeg binary
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"✅ Using imageio-ffmpeg binary: {FFMPEG_PATH}")
except (ImportError, RuntimeError):
    FFMPEG_PATH = 'ffmpeg'
    print("⚠️  imageio-ffmpeg not found, falling back to system ffmpeg")


class FFmpegUtils:
    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run([FFMPEG_PATH, '-version'], 
                         capture_output=True, 
                         check=True,
                         timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """Get video duration in seconds using ffmpeg"""
        # Convert to absolute path
        video_path = os.path.abspath(video_path)
        
        try:
            # Use ffmpeg to get duration from the file
            cmd = [
                FFMPEG_PATH,
                '-i', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stderr  # ffmpeg outputs to stderr
            
            # Parse duration from output: Duration: HH:MM:SS.ms
            match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', output)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = float(match.group(3))
                total_seconds = hours * 3600 + minutes * 60 + seconds
                return total_seconds
            
            # Fallback: return default if parsing fails
            print(f"Warning: Could not parse duration from {video_path}")
            return 0.0
            
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return 0.0
    
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
            FFMPEG_PATH,
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
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
                FFMPEG_PATH,
                '-i', audio_path,
                '-ss', str(current_time),
                '-t', str(end_time - current_time),
                '-acodec', 'copy',
                '-y',
                chunk_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            
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
                FFMPEG_PATH,
                '-ss', str(current_time),
                '-i', video_path,
                '-frames:v', '1',
                '-q:v', '2',  # High quality
                '-y',
                frame_path
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=60)
                frames.append((frame_path, current_time))
                frame_index += 1
            except subprocess.CalledProcessError:
                # Skip if frame extraction fails
                pass
            
            current_time += interval
        
        return frames
    
    @staticmethod
    def extract_dense_frames(
        video_path: str,
        output_dir: str,
        time_windows: List[Tuple[float, float]],
        fps: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Extract frames densely (e.g., 1 FPS) only within specific time windows.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            time_windows: List of (start, end) tuples in seconds
            fps: Frames per second to extract (default: 1)
        
        Returns:
            List of tuples: (frame_path, timestamp)
        """
        os.makedirs(output_dir, exist_ok=True)
        all_frames = []
        
        print(f"Extracting dense frames for {len(time_windows)} windows...")
        
        for i, (start, end) in enumerate(time_windows):
            duration = end - start
            if duration <= 0:
                continue
                
            # Create a pattern that includes the timestamp to avoid collisions
            # We'll use a unique prefix for each window to ensure no overwrites
            # output pattern: window_{i}_frame_%04d.jpg
            
            # Use filter_complex to trim and fps
            # ffmpeg -ss start -t duration -i input -vf fps=1 out_%d.jpg
            
            # We want exact timestamps? It's harder with batch. 
            # Let's do a loop for now or use the fps filter efficiently.
            # Using -vf fps=1 is efficient.
            
            window_output_pattern = os.path.join(output_dir, f"win_{i}_%04d.jpg")
            
            cmd = [
                FFMPEG_PATH,
                '-ss', str(start),
                '-t', str(duration),
                '-i', video_path,
                '-vf', f'fps={fps}',
                '-q:v', '2',
                '-y',
                window_output_pattern
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=120)
                
                # Now collect the generated files and map them to timestamps
                # The files will be win_0_0001.jpg, win_0_0002.jpg etc.
                # Timestamp = start + (index-1)/fps
                
                for filename in sorted(os.listdir(output_dir)):
                    if filename.startswith(f"win_{i}_") and filename.endswith(".jpg"):
                        # Parse frame number
                        try:
                            # win_0_0001.jpg -> 1
                            frame_num = int(filename.split('_')[-1].split('.')[0])
                            
                            # Calculate timestamp: start + (frame_num - 1) / fps
                            # FFmpeg usually starts at 0 or 1 depending on version/settings
                            # but with fps filter, the first frame is usually at relative t=0 or t=0.5/fps
                            # Approximation is fine for now.
                            
                            timestamp = start + (frame_num - 1) / fps
                            full_path = os.path.join(output_dir, filename)
                            
                            # Rename to something more semantic: frame_{timestamp}s.jpg
                            # to match the rest of the pipeline if needed, OR just keep it.
                            # Let's rename for clarity: frame_00000_12s.jpg (5 digits for seconds)
                            
                            safe_ts_str = f"{int(timestamp):05d}"
                            new_name = f"frame_{safe_ts_str}_{i}_{frame_num}.jpg"
                            new_path = os.path.join(output_dir, new_name)
                            
                            if not os.path.exists(new_path): # Avoid double rename if running multiple times
                                os.rename(full_path, new_path)
                                all_frames.append((new_path, timestamp))
                            else:
                                all_frames.append((new_path, timestamp))
                                
                        except ValueError:
                            continue
                            
            except subprocess.CalledProcessError as e:
                print(f"Error extracting dense frames for window {start}-{end}: {e}")
                
        return sorted(all_frames, key=lambda x: x[1])

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# Check FFmpeg availability on module load
if not FFmpegUtils.check_ffmpeg():
    print("WARNING: FFmpeg not found. Please ensure imageio-ffmpeg is installed or FFmpeg is in PATH.")
