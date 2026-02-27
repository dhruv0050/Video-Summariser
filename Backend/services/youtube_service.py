import os
import re
from typing import Optional
import yt_dlp
import config


class YouTubeService:
    """Service for downloading YouTube videos"""
    
    @staticmethod
    def _resolve_cookies_path():
        """Resolve cookies file path with multiple fallback strategies"""
        if not config.YOUTUBE_COOKIES_PATH:
            return None
        
        potential_paths = []
        
        # 1. Use as-is if absolute
        if os.path.isabs(config.YOUTUBE_COOKIES_PATH):
            potential_paths.append(config.YOUTUBE_COOKIES_PATH)
        else:
            # 2. Relative to current working directory
            potential_paths.append(os.path.abspath(config.YOUTUBE_COOKIES_PATH))
            
            # 3. Relative to Backend directory (where main.py is)
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            potential_paths.append(os.path.join(script_dir, config.YOUTUBE_COOKIES_PATH))
            
            # 4. Relative to utils directory
            utils_dir = os.path.dirname(os.path.abspath(__file__))
            potential_paths.append(os.path.join(utils_dir, 'cookies.txt'))
            
            # 5. Try just the filename in Backend directory
            potential_paths.append(os.path.join(script_dir, os.path.basename(config.YOUTUBE_COOKIES_PATH)))
        
        # Try each potential path
        for path in potential_paths:
            if os.path.exists(path) and os.path.isfile(path):
                return os.path.abspath(path)
        
        return None
    
    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        # If no match, assume the URL is just the video ID
        if len(youtube_url) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', youtube_url):
            return youtube_url
        
        raise ValueError(f"Invalid YouTube URL format: {youtube_url}")
    
    @staticmethod
    def get_video_info(video_id: str) -> dict:
        """Get video metadata without downloading"""
        proxy_url = os.getenv('PROXY_URL')
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'verbose': False,
        }
        
        if proxy_url:
            ydl_opts['proxy'] = proxy_url
            print(f"✅ Using proxy for metadata extraction")
        
        # Add cookies if configured (use same path resolution as download_video)
        cookies_path = YouTubeService._resolve_cookies_path()
        if cookies_path:
            ydl_opts['cookies'] = cookies_path
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Untitled Video'),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', '')
                }
            except Exception as e:
                raise Exception(f"Failed to fetch video info: {str(e)}")
    
    @staticmethod
    def extract_playlist_info(playlist_url: str) -> dict:
        """Extract all video metadata from a YouTube playlist without downloading.
        
        Args:
            playlist_url: YouTube playlist URL
            
        Returns:
            Dict with playlist title, description, channel, and list of videos
        """
        proxy_url = os.getenv('PROXY_URL')
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',  # Get metadata only, don't resolve each video
            'skip_download': True,
            'verbose': False,
        }
        
        if proxy_url:
            ydl_opts['proxy'] = proxy_url
            print(f"✅ Using proxy for playlist extraction")
        
        cookies_path = YouTubeService._resolve_cookies_path()
        if cookies_path:
            ydl_opts['cookies'] = cookies_path
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(playlist_url, download=False)
                
                entries = info.get('entries', [])
                videos = []
                for i, entry in enumerate(entries):
                    if entry is None:
                        continue
                    video_id = entry.get('id', '')
                    videos.append({
                        'video_url': f"https://www.youtube.com/watch?v={video_id}",
                        'video_id': video_id,
                        'video_title': entry.get('title', f'Video {i+1}'),
                        'duration': entry.get('duration') or 0,
                        'order': i
                    })
                
                return {
                    'title': info.get('title', 'Untitled Playlist'),
                    'description': info.get('description', ''),
                    'channel': info.get('uploader', info.get('channel', 'Unknown')),
                    'video_count': len(videos),
                    'videos': videos
                }
            except Exception as e:
                raise Exception(f"Failed to extract playlist info: {str(e)}")
    
    @staticmethod
    def download_video(
        youtube_url: str,
        output_path: str,
        video_id: Optional[str] = None
    ) -> str:
        """
        Download YouTube video to specified path
        
        Args:
            youtube_url: YouTube video URL
            output_path: Path where video should be saved
            video_id: Optional video ID (if already extracted)
        
        Returns:
            Path to downloaded video file
        """
        if video_id is None:
            video_id = YouTubeService.extract_video_id(youtube_url)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure yt-dlp options for best quality and format
        # Remove extension from output_path for yt-dlp template
        base_output_path = output_path.rsplit('.', 1)[0] if '.' in os.path.basename(output_path) else output_path
        
        # Format preference: avoid HLS (m3u8) which can have fragment issues, prefer progressive mp4
        # Try multiple format strategies with fallbacks
        format_selectors = [
            # First try: Progressive MP4 (single file, most reliable)
            'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            # Second try: Any progressive format (no HLS)
            'best[protocol!=m3u8_native][ext=mp4]/best[protocol!=m3u8_native]',
            # Third try: Best available (including HLS as last resort)
            'best'
        ]
        
        proxy_url = os.getenv('PROXY_URL')
        
        ydl_opts = {
            'format': format_selectors[0],  # Start with best MP4
            'outtmpl': base_output_path + '.%(ext)s',  # yt-dlp will add extension
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [YouTubeService._progress_hook],
            'no_check_certificate': False,
            'prefer_insecure': False,
            'verbose': False,
            'fragment_retries': 3,
            'retries': 3,
        }
        
        if proxy_url:
            ydl_opts['proxy'] = proxy_url
            print(f"✅ Using proxy for video download")
        
        # Add cookies if configured (prioritize cookies file for server environments)
        cookies_configured = False
        cookies_path = YouTubeService._resolve_cookies_path()
        
        if cookies_path:
            ydl_opts['cookies'] = cookies_path
            file_size = os.path.getsize(cookies_path)
            print(f"✅ Using cookies from file: {cookies_path} ({file_size} bytes)")
            cookies_configured = True
        elif config.YOUTUBE_COOKIES_PATH:
            print(f"❌ Cookies file not found at: {config.YOUTUBE_COOKIES_PATH}")
            print(f"   Current working directory: {os.getcwd()}")
            print(f"   Script directory: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
        elif config.YOUTUBE_COOKIES_FROM_BROWSER:
            # Try cookies_from_browser, but it may fail on servers
            try:
                ydl_opts['cookies_from_browser'] = (config.YOUTUBE_COOKIES_FROM_BROWSER,)
                print(f"Attempting to use cookies from browser: {config.YOUTUBE_COOKIES_FROM_BROWSER}")
                cookies_configured = True
            except Exception as e:
                print(f"Warning: cookies_from_browser not available: {e}")
                print("Note: On servers (like Render), use YOUTUBE_COOKIES_PATH with a cookies.txt file instead")
        
        # If no cookies configured, try mobile clients which sometimes bypass bot detection
        if not cookies_configured:
            print("⚠️ No cookies configured - using default clients (may still trigger bot detection)")
            print("   Recommendation: Set YOUTUBE_COOKIES_PATH to a valid cookies.txt file")
        else:
            print(f"✅ Cookies configured - using authenticated requests")
        
        # If the URL is just a video ID, construct full URL
        if not youtube_url.startswith('http'):
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Try download with multiple format strategies
        last_error = None
        for format_idx, format_selector in enumerate(format_selectors):
            try:
                ydl_opts['format'] = format_selector
                print(f"Attempting download with format selector {format_idx + 1}/{len(format_selectors)}: {format_selector[:50]}...")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
                    
                    # yt-dlp may have added an extension, check for the actual file
                    possible_extensions = ['.mp4', '.webm', '.mkv', '.flv', '.m4a']
                    
                    for ext in possible_extensions:
                        potential_path = base_output_path + ext
                        if os.path.exists(potential_path) and os.path.getsize(potential_path) > 0:
                            # Verify file is not empty
                            file_size = os.path.getsize(potential_path)
                            print(f"Downloaded file found: {potential_path} ({file_size} bytes)")
                            
                            # If the extension doesn't match what we want, rename it
                            if potential_path != output_path:
                                os.rename(potential_path, output_path)
                            return output_path
                    
                    # If no file found with extensions, check if output_path already exists
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        return output_path
                    
                    # Last resort: check if base_output_path exists
                    if os.path.exists(base_output_path) and os.path.getsize(base_output_path) > 0:
                        os.rename(base_output_path, output_path)
                        return output_path
                    
                    # If we got here, file is empty or doesn't exist
                    if format_idx < len(format_selectors) - 1:
                        print(f"Download attempt {format_idx + 1} failed: file empty or not found. Trying next format...")
                        continue
                    else:
                        raise Exception(f"Downloaded file is empty or not found after all attempts")
                        
            except Exception as e:
                last_error = str(e)
                print(f"Download attempt {format_idx + 1} failed: {last_error}")
                
                # If this was the last attempt, raise the error
                if format_idx == len(format_selectors) - 1:
                    # Clean up any empty files
                    for ext in ['.mp4', '.webm', '.mkv', '.flv', '.m4a']:
                        potential_path = base_output_path + ext
                        if os.path.exists(potential_path) and os.path.getsize(potential_path) == 0:
                            try:
                                os.remove(potential_path)
                            except:
                                pass
                    
                    raise Exception(f"Failed to download YouTube video after {len(format_selectors)} attempts. Last error: {last_error}")
                
                # Continue to next format selector
                continue
        
        # Should not reach here, but just in case
        raise Exception(f"Failed to download YouTube video: {last_error or 'Unknown error'}")
    
    @staticmethod
    def _progress_hook(d):
        """Progress hook for yt-dlp download"""
        if d['status'] == 'downloading':
            p = d.get('downloaded_bytes', 0)
            t = d.get('total_bytes') or d.get('total_bytes_estimate')
            if t:
                percent = int((p / t) * 100)
                # Only log every 10% and avoid double logging same %
                last_p = getattr(YouTubeService, '_last_percent', -1)
                if percent % 10 == 0 and percent != last_p:
                    YouTubeService._last_percent = percent
                    speed = d.get('_speed_str', 'N/A')
                    print(f"Download Progress: {percent}% at {speed}")
        elif d['status'] == 'finished':
            print(f"Download complete: {d['filename']}")


# Singleton instance
youtube_service = YouTubeService()
