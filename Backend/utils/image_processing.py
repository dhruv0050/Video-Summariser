from typing import List, Dict, Tuple, Optional
import os
# import cv2 # Removed to avoid numpy version conflict
# import numpy as np
from PIL import Image, ImageFilter, ImageStat

class ImageProcessor:
    """Utilities for image processing, deduplication, and quality assessment."""
    
    @staticmethod
    def calculate_phash(image_path: str, hash_size: int = 8) -> Optional[str]:
        """
        Calculate perceptual hash of an image using dHash (Difference Hash).
        Returns hex string of the hash.
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return None
                
            # Use PIL to open and convert to grayscale
            with Image.open(image_path) as img:
                img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
                pixels = list(img.getdata())
                
            # Compare adjacent pixels
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = pixels[row * (hash_size + 1) + col]
                    pixel_right = pixels[row * (hash_size + 1) + col + 1]
                    diff.append(pixel_left > pixel_right)
                    
            # Convert binary array to hex string
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(diff):
                if value:
                    decimal_value += 2**(index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
                    decimal_value = 0
                    
            return "".join(hex_string)
            
        except Exception as e:
            print(f"Error calculating hash for {image_path}: {e}")
            return None

    @staticmethod
    def calculate_blur(image_path: str) -> float:
        """
        Calculate sharpness score using PIL (variance of edges).
        Higher is sharper.
        """
        try:
            with Image.open(image_path) as img:
                img = img.convert("L")
                edges = img.filter(ImageFilter.FIND_EDGES)
                stat = ImageStat.Stat(edges)
                return stat.var[0]
        except Exception as e:
            print(f"Error calculating blur for {image_path}: {e}")
            return 0.0

    @staticmethod
    def cluster_frames(frames: List[Tuple[str, float]], threshold: int = 10) -> List[Dict]:
        """
        Cluster similar frames based on perceptual hash.
        
        Args:
            frames: List of (path, timestamp) tuples
            threshold: Hamming distance threshold for similarity (0-64). Lower = stricter.
            
        Returns:
            List of clusters (dicts with 'frames', 'start_time', 'end_time', 'key_frame')
        """
        if not frames:
            return []
            
        clusters = []
        current_cluster = []
        last_hash = None
        
        # Helper to calculate hamming distance
        def hamming_distance(h1, h2):
            if not h1 or not h2: return 64
            bin1 = bin(int(h1, 16))[2:].zfill(64)
            bin2 = bin(int(h2, 16))[2:].zfill(64)
            return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))

        # Calculate hashes for all frames
        frame_hashes = []
        for path, ts in frames:
            phash = ImageProcessor.calculate_phash(path)
            if phash:
                frame_hashes.append({'path': path, 'timestamp': ts, 'hash': phash, 'blur_score': 0})
        
        if not frame_hashes:
            return []

        # Start clustering
        current_cluster = [frame_hashes[0]]
        
        for i in range(1, len(frame_hashes)):
            frame = frame_hashes[i]
            prev_frame = frame_hashes[i-1]
            
            dist = hamming_distance(frame['hash'], prev_frame['hash'])
            
            # If similar, add to current cluster
            if dist <= threshold:
                current_cluster.append(frame)
            else:
                # Close current cluster and start new one
                clusters.append(current_cluster)
                current_cluster = [frame]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(current_cluster)
            
        # Process clusters to be ready for Gemini
        processed_clusters = []
        for cluster in clusters:
            # Calculate blur scores for ranking if cluster is large
            # We only need to calculate scores for the top candidates or all if cluster is small
            # For simplicity, calculate for all in clusters > 1
            if len(cluster) > 1:
                for f in cluster:
                    f['blur_score'] = ImageProcessor.calculate_blur(f['path'])
            else:
                cluster[0]['blur_score'] = 100.0 # Default high score for single frame

            # Sort by sharpness (descending)
            cluster.sort(key=lambda x: x['blur_score'], reverse=True)
            
            # Select top K candidates (max 5)
            candidates = cluster[:5]
            
            processed_clusters.append({
                "start_time": min(f['timestamp'] for f in cluster),
                "end_time": max(f['timestamp'] for f in cluster),
                "frame_count": len(cluster),
                "candidates": candidates # List of dicts with path, timestamp, blur_score
            })
            
        return processed_clusters
