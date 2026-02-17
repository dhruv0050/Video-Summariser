from typing import List, Dict, Tuple
import math

def merge_time_windows(
    audio_cues: List[Dict], 
    visual_rois: List[Dict], 
    total_duration: float,
    buffer_seconds: float = 10.0,
    min_gap: float = 5.0
) -> List[Tuple[float, float]]:
    """
    Merge audio and visual cues into continuous processing windows.
    
    Args:
        audio_cues: List of dicts with 'timestamp' (str or float)
        visual_rois: List of dicts with 'timestamp' (float)
        total_duration: Total video duration in seconds
        buffer_seconds: Padding around each point (e.g. T-10s to T+10s)
        min_gap: Merge windows if they are closer than this
        
    Returns:
        List of (start, end) tuples in seconds
    """
    
    # 1. Collect all interesting timestamps
    timestamps = []
    
    # Process audio cues
    for cue in audio_cues:
        ts = cue.get("timestamp")
        if isinstance(ts, str):
            # Parse HH:MM:SS
            try:
                parts = ts.split(':')
                if len(parts) == 3:
                    h, m, s = map(float, parts)
                    seconds = h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    m, s = map(float, parts)
                    seconds = m * 60 + s
                else:
                    seconds = float(ts)
                timestamps.append(seconds)
            except:
                continue
        elif isinstance(ts, (int, float)):
            timestamps.append(float(ts))
            
    # Process visual ROIs (Gatekeeper hits)
    for roi in visual_rois:
        ts = roi.get("timestamp")
        if isinstance(ts, (int, float)):
            timestamps.append(float(ts))
            
    if not timestamps:
        return []
        
    # 2. Create initial windows [T - buffer, T + buffer]
    windows = []
    for ts in timestamps:
        start = max(0, ts - buffer_seconds)
        end = min(total_duration, ts + buffer_seconds)
        windows.append((start, end))
        
    # Sort by start time
    windows.sort(key=lambda x: x[0])
    
    # 3. Merge overlapping or close windows
    if not windows:
        return []
        
    merged = [windows[0]]
    
    for current_start, current_end in windows[1:]:
        last_start, last_end = merged[-1]
        
        # If overlap or gap is small enough, merge
        if current_start <= last_end + min_gap:
            new_end = max(last_end, current_end)
            merged[-1] = (last_start, new_end)
        else:
            merged.append((current_start, current_end))
            
    return merged
