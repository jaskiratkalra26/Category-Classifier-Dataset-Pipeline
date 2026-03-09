import yt_dlp
import os
import pandas as pd
import config
from tqdm import tqdm
from frame_sampler import sample_frames
from clip_embedder import ClipEmbedder
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Configuration for new dataset
DATASET_CSV = 'youtube_shorts_tiktok_trends_2025.csv'
VIDEO_EMBEDDINGS_FILE = 'video_embeddings.csv' # Or modify config
TARGET_DURATION_MIN = 10
TARGET_DURATION_MAX = 60
TARGET_CATEGORIES = [
    "gaming", "education", "technology", "finance", "fitness",
    "cooking", "travel", "music", "comedy", "news", "sports", "beauty"
]

CATEGORY_MAP = {
    # New CSV category -> Target
    "Gaming": "gaming",
    "Food": "cooking",
    "News": "news",
    "Beauty": "beauty",
    "Fitness": "fitness",
    "Comedy": "comedy",
    "Travel": "travel",
    "Tech": "technology",
    "Music": "music",
    "Sports": "sports",
    "Education": "education",
    "Finance": "finance",
    "Fashion": "beauty",
    "Science": "education", # or technology?
    "Automotive": "technology" # maybe?
    # "Lifestyle", "Art", "Pets", "DIY" left out unless needed
}

def search_video(title, author_handle):
    """
    Search for video on YouTube using yt-dlp.
    Returns the video ID if found.
    """
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': True,
        'noplaylist': True,
        'match_filter': yt_dlp.utils.match_filter_func("!is_live"),
    }
    
    query = f"ytsearch1:{title} {author_handle}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if 'entries' in info and len(info['entries']) > 0:
                return info['entries'][0]['id']
            # Sometimes single result is returned directly
            if 'id' in info:
                return info['id']
    except Exception as e:
        # print(f"Search failed for {title}: {e}")
        pass
    return None

def download_video_search_and_fetch(row):
    """
    Finds video ID via search, then downloads if not exists.
    Returns: (video_id, video_path) or (None, None)
    """
    title = row['title']
    author = row['author_handle'] # Check column name
    category = row['mapped_category']
    
    # 1. Search for ID
    video_id = search_video(title, author)
    if not video_id:
        return None, None
        
    # check exists
    base_dir = os.path.join(config.VIDEOS_DIR, category)
    os.makedirs(base_dir, exist_ok=True)
    
    # Check if file exists
    # reusing logic from video_downloader
    # simple check:
    expected_path = os.path.join(base_dir, f"{video_id}.mp4")
    if os.path.exists(expected_path):
        return video_id, expected_path
    
    # Wildcard check
    for f in os.listdir(base_dir):
        if f.startswith(video_id + "."):
            return video_id, os.path.join(base_dir, f)
            
    # Download
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(base_dir, f'%(id)s.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Verify download
        for f in os.listdir(base_dir):
            if f.startswith(video_id + "."):
                return video_id, os.path.join(base_dir, f)
    except Exception as e:
        # print(f"Download failed for {video_id}: {e}")
        pass
        
    return None, None

def main():
    print("Loading youtube_shorts_tiktok_trends_2025.csv...")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_CSV)
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # 1. Filter Platform: YouTube
    # Column 'platform'
    print("Filtering for YouTube platform...")
    df = df[df['platform'].str.contains('YouTube', case=False, na=False)]
    
    # 2. Filter Duration (10-60s)
    print("Filtering duration (10-60s)...")
    # Clean duration column if needed (e.g. '18' -> 18)
    df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce')
    df = df.dropna(subset=['duration_sec'])
    df = df[(df['duration_sec'] >= TARGET_DURATION_MIN) & (df['duration_sec'] <= TARGET_DURATION_MAX)]
    
    # 3. Map Categories
    print("Mapping categories...")
    def get_mapped_category(cat):
        if not isinstance(cat, str): return None
        return CATEGORY_MAP.get(cat, None)
        
    df['mapped_category'] = df['category'].apply(get_mapped_category)
    df = df.dropna(subset=['mapped_category'])
    
    # 4. Equalize Records
    counts = df['mapped_category'].value_counts()
    print("Counts per category (after mapping):")
    print(counts)
    
    if counts.empty:
        print("No matches found.")
        return
        
    min_count = counts.min()
    print(f"Equalizing to {min_count} records per category.")
    sampled_df = df.groupby('mapped_category').sample(n=min_count, random_state=42)
    
    # 5. Process Videos
    print("Processing videos (Search -> Download -> Embed)...")
    
    # Initialize Embedder
    embedder = ClipEmbedder()
    
    # Results container
    results = []
    
    # Using ThreadPoolExecutor only for search/download to speed up IO
    # Embedding is CPU/GPU bound, better sequential or separate process?
    # Mixed: search/download in parallel, embed sequentially to avoid memory issues with CLIP
    
    # Prepare tasks
    tasks = []
    for _, row in sampled_df.iterrows():
        tasks.append(row)
        
    print(f"Total videos to process: {len(tasks)}")
    
    # Phase 1: Download all first (parallel)
    # Mapping tasks to results
    downloaded_videos = [] # List of (video_id, category, path)
    
    print("Phase 1: Searching and Downloading Videos...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(download_video_search_and_fetch, row): row for row in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(tasks)):
            row = future_to_row[future]
            try:
                vid_id, vid_path = future.result()
                if vid_id and vid_path:
                    downloaded_videos.append({
                        'video_id': vid_id,
                        'category': row['mapped_category'],
                        'path': vid_path,
                        'original_title': row['title']
                    })
            except Exception as e:
                # print(f"Task failed: {e}")
                pass
                
    print(f"Phase 1 Complete. {len(downloaded_videos)} videos ready for embedding.")
    
    # Phase 2: Embed
    print("Phase 2: Generating Embeddings...")
    processed_count = 0
    
    output_path = config.OUTPUT_DATASET_FILE
    
    # Creating final DataFrame
    final_data = []
    
    for item in tqdm(downloaded_videos):
        try:
            frames = sample_frames(item['path'], num_frames=config.FRAME_SAMPLE_COUNT)
            if not frames:
                continue
            
            embedding = embedder.get_embedding(frames)
            if embedding is None:
                continue
                
            res = {
                "video_id": item['video_id'],
                "category": item['category'],
                # "title": item['original_title'] # Optional
            }
            for i, val in enumerate(embedding):
                res[f"embedding_{i}"] = val
            
            final_data.append(res)
            processed_count += 1
            
        except Exception as e:
            print(f"Embedding error {item['video_id']}: {e}")
            continue

    if final_data:
        print(f"Saving {len(final_data)} records to {output_path}...")
        pd.DataFrame(final_data).to_csv(output_path, index=False)
        print("Done.")
    else:
        print("No embeddings generated.")

if __name__ == "__main__":
    main()
