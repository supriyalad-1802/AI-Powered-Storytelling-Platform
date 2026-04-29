import os
from moviepy import VideoFileClip
import uuid

def extract_highlight(video_path, timestamp_str, static_folder, duration=10):
    """
    Extracts a highlight clip from a video file based on a given timestamp MM:SS.
    The clip will start 5 seconds before the timestamp (if possible) and last for `duration` seconds.
    """
    try:
        parts = timestamp_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            total_seconds = minutes * 60 + seconds
        else:
            return None
            
        start_time = max(0, total_seconds - 5)
        end_time = start_time + duration
        
        # We need a unique output name
        clip_id = str(uuid.uuid4())[:8]
        output_filename = f"highlight_{clip_id}.mp4"
        
        # Save directly to the static folder's results subdirectory
        results_dir = os.path.join(static_folder, 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, output_filename)
        
        with VideoFileClip(video_path) as video:
            # Ensure we don't go past the video duration
            end_time = min(end_time, video.duration)
            if start_time >= video.duration:
                start_time = max(0, video.duration - duration)
                end_time = video.duration
                
            highlight = video.subclip(start_time, end_time)
            # Remove audio from the highlight clip as we have TTS narrative
            highlight = highlight.without_audio()
            highlight.write_videofile(
                output_path, 
                codec="libx264", 
                audio=False,
                logger=None # Suppress moviepy output logs
            )
            
        return output_filename
    except Exception as e:
        print(f"Error extracting video highlight: {e}")
        return None
