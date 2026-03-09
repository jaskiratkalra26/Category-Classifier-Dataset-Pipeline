import pandas as pd
import os
import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor
import yt_dlp
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
try:
    from clip_embedder import ClipEmbedder
    from frame_sampler import sample_frames
except ImportError:
    # If standard import fails, try relative or assume user fixes environment
    pass

# Categories from user request
TARGET_CATEGORIES = [
    "Gaming", "Education", "Technology", "Finance", "Fitness",
    "Cooking", "Travel", "Music", "Comedy", "News", "Sports", "Beauty"
]

# Mapping from dataset categories to target categories
CATEGORY_MAP = {
    "Gaming": "Gaming",
    "Education": "Education",
    "Tech": "Technology",
    "Science": "Technology", # Map Science to Tech if needed
    "Finance": "Finance",
    "Fitness": "Fitness",
    "Food": "Cooking",
    "Travel": "Travel",
    "Music": "Music",
    "Comedy": "Comedy",
    "News": "News",
    "Sports": "Sports",
    "Beauty": "Beauty",
    "Fashion": "Beauty", # Map Fashion to Beauty
    # Add others if needed
}

def search_and_download_video(row, output_dir):
    """
    Searches for a video on YouTube using title and author, and downloads it.
    Returns the path to the downloaded file or None.
    """
    title = row['title']
    author = row['author_handle']
    search_query = f"{title} {author}"
    
    # Clean search query
    search_query = "".join([c if c.isalnum() or c.isspace() else "" for c in search_query])
    
    # Configure yt-dlp
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'max_filesize': 50 * 1024 * 1024, # 50MB limit
        'match_filter': yt_dlp.utils.match_filter_func("duration > 10 & duration < 60"),
        'default_search': 'ytsearch1',
        'ignoreerrors': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # We search and download in one go
            # This might be slow if we search many times
            # Note: yt-dlp search returns a playlist-like object
            info = ydl.extract_info(f"ytsearch1:{search_query}", download=True)
            
            if 'entries' in info:
                video_info = info['entries'][0]
            else:
                video_info = info

            if not video_info:
                return None
                
            video_id = video_info.get('id')
            ext = video_info.get('ext', 'mp4')
            path = os.path.join(output_dir, f"{video_id}.{ext}")
            
            if os.path.exists(path):
                return path
            
    except Exception as e:
        # print(f"Error downloading {search_query}: {e}")
        return None
        
    return None

def main():
    print("Loading youtube_shorts_tiktok_trends_2025.csv...")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'youtube_shorts_tiktok_trends_2025.csv')
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter Platform
    print("Filtering for YouTube platform...")
    df = df[df['platform'].str.lower() == 'youtube']
    
    # Filter Duration
    print("Filtering duration 10-60s...")
    df = df[(df['duration_sec'] >= 10) & (df['duration_sec'] <= 60)]
    
    # Map Categories
    print("Mapping categories...")
    df['mapped_category'] = df['category'].map(CATEGORY_MAP)
    df = df.dropna(subset=['mapped_category'])
    
    # Filter Targets
    df = df[df['mapped_category'].isin(TARGET_CATEGORIES)]
    
    # Equalize
    counts = df['mapped_category'].value_counts()
    print("Counts available per category:")
    print(counts)
    
    if counts.empty:
        print("No videos found.")
        return

    # User asked for "equal records". Since we rely on search which is slow/uncertain,
    # let's cap at a reasonable number for a test run, e.g., 20.
    # If the user wants ALL 800+, they can increase this limit.
    # Given "a 1.5 gb video is being dowloaded" complaint, smaller batches are better to start.
    target_count = 20
    print(f"Equalizing to {target_count} records per category for processing...")
    
    sampled_df = df.groupby('mapped_category').sample(n=target_count, replace=False, random_state=42)
    
    # Initialize Embedder
    print("Initializing ClipEmbedder...")
    try:
        embedder = ClipEmbedder()
    except Exception as e:
        print(f"Failed to init embedder: {e}")
        return

    # Output file
    output_path = config.OUTPUT_DATASET_FILE
    if os.path.exists(output_path):
        # Backup old one
        os.rename(output_path, output_path + ".bak")
        
    results = []
    
    print("Processing videos (Search -> Download -> Embed)...")
    
    # We process sequentially or parallel?
    # Search is network bound. Embedding is CPU/GPU bound.
    # Let's do sequential to avoid "1.5 gb" parallel surprises and easier debugging.
    
    failed_downloads = 0
    
    for idx, row in tqdm.tqdm(sampled_df.iterrows(), total=len(sampled_df)):
        category = row['mapped_category']
        cat_dir = os.path.join(config.VIDEOS_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        # Download
        video_path = search_and_download_video(row, cat_dir)
        
        if not video_path:
            failed_downloads += 1
            continue
            
        # Extract Frames
        frames = sample_frames(video_path, num_frames=config.FRAME_SAMPLE_COUNT)
        if not frames:
            continue
            
        # Embed
        embedding = embedder.get_embedding(frames)
        if embedding is None:
            continue
            
        # Store
        res = {
            "video_id": os.path.basename(video_path).split('.')[0], # Use filename as ID since we don't have original ID
            "category": category,
            "original_title": row['title'],
            "original_author": row['author_handle']
        }
        for i, val in enumerate(embedding):
            res[f"embedding_{i}"] = val
            
        results.append(res)
        
        # Save incrementally
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    # Final Save
    if results:
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"Done. Saved {len(results)} records to {output_path}.")
        print(f"Failed downloads: {failed_downloads}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
