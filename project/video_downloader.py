import yt_dlp
import os
import pandas as pd
from tqdm import tqdm
import config

def download_videos(metadata_file):
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found.")
        return

    df = pd.read_csv(metadata_file)
    print(f"Loaded {len(df)} videos for downloading.")

    # Group by category to manage directories
    grouped = df.groupby('category')

    for category, group in grouped:
        category_dir = os.path.join(config.VIDEOS_DIR, category)
        os.makedirs(category_dir, exist_ok=True)
        
        print(f"Downloading {len(group)} videos for category: {category}")
        
        # Prepare list of URLs and output templates
        ydl_opts = {
            # Prefer single file downloads to avoid needing ffmpeg for merging
            'format': 'best[ext=mp4]/best',
            'outtmpl': os.path.join(category_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True, # Skip individual video errors
            'download_archive': os.path.join(config.VIDEOS_DIR, 'downloaded.txt'), # Track downloaded files
        }

        urls = [f"https://www.youtube.com/watch?v={vid}" for vid in group['video_id']]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # ydl.download handles the list, but for progress bar we might want to iterate manually
            # However, ydl has its own internal progress. Let's use manual iteration to wrap with tqdm for our own counter
            
            pbar = tqdm(total=len(urls), unit="video")
            for url in urls:
                try:
                    ydl.download([url])
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
                pbar.update(1)
            pbar.close()

if __name__ == "__main__":
    download_videos(config.METADATA_FILE)
