import yt_dlp
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import logging

# Configure basic logging to avoid cluttering console too much
logging.basicConfig(filename='downloader_errors.log', level=logging.ERROR)

def get_video_path(video_id, category):
    """Returns the expected path of a video if it exists."""
    base_dir = os.path.join(config.VIDEOS_DIR, category)
    if not os.path.exists(base_dir):
        return None
        
    for filename in os.listdir(base_dir):
        if filename.startswith(video_id + "."):
            return os.path.join(base_dir, filename)
            
    # Fallback checks (slower but safer)
    for ext in ['.mp4', '.mkv', '.webm']:
        potential_path = os.path.join(base_dir, f"{video_id}{ext}")
        if os.path.exists(potential_path):
            return potential_path
            
    return None

def download_video_sync(video_info):
    """
    Downloads a single video. Designed to be run in a thread pool.
    video_info: dict or tuple containing (video_id, category)
    """
    video_id = video_info['video_id']
    category = video_info['category']
    
    # Fast check if already exists
    existing_path = get_video_path(video_id, category)
    if existing_path:
        return existing_path

    category_dir = os.path.join(config.VIDEOS_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Use a fresh options dict for each thread to be safe
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(category_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        # 'download_archive': ... concurrent writing to archive file is risky
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        # Verify success
        final_path = get_video_path(video_id, category)
        return final_path
    except Exception as e:
        logging.error(f"Failed to download {video_id}: {e}")
        return None

def download_videos(metadata_file, max_workers=5):
    """
    Parallel version of download_videos.
    """
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found.")
        return

    df = pd.read_csv(metadata_file)
    
    # Load failed IDs
    failed_ids = set()
    if os.path.exists(config.FAILED_VIDEOS_FILE):
        try:
            with open(config.FAILED_VIDEOS_FILE, 'r') as f:
                failed_ids = set(line.strip() for line in f if line.strip())
        except Exception:
            pass

    # Filter out failed videos
    videos_to_process = [v for v in df.to_dict('records') if str(v['video_id']) not in failed_ids]
    
    print(f"Loaded {len(df)} videos. Skipping {len(df) - len(videos_to_process)} failed. Downloading {len(videos_to_process)} videos with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_video_sync, vid): vid for vid in videos_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), unit="video"):
            try:
                result = future.result()
            except Exception as e:
                logging.error(f"Thread error: {e}")

if __name__ == "__main__":
    download_videos(config.METADATA_FILE)
