import pandas as pd
import os
import sys
import tqdm
import cv2
from PIL import Image
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from clip_embedder import ClipEmbedder
from frame_sampler import sample_frames
from video_downloader import download_video_sync

def match_keywords(text, keywords):
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(k in text for k in keywords)

KEYWORD_MAP = {
    "finance": ["finance", "money", "invest", "stock", "economy", "business", "crypto", "trading", "market", "wealth"],
    "fitness": ["fitness", "workout", "gym", "exercise", "bodybuilding", "yoga", "health", "train", "muscle", "weight"],
    "cooking": ["cooking", "recipe", "food", "kitchen", "bake", "baking", "chef", "meal", "cook", "delicious", "eat"],
    "beauty": ["beauty", "makeup", "skincare", "cosmetics", "fashion", "hair", "tutorial", "style", "look"]
}

CATEGORY_MAP = {
    "Gaming": "gaming",
    "Education": "education",
    "Science & Technology": "technology",
    "Comedy": "comedy",
    "Music": "music",
    "News & Politics": "news",
    "Sports": "sports",
    "Travel & Events": "travel"
}

def classify_row(row):
    cat = row['category']
    if cat in CATEGORY_MAP:
        return CATEGORY_MAP[cat]
    
    text_blob = f"{row['title']} {row['description']} {row['hashtags']}"
    matched = []
    for target_cat, keywords in KEYWORD_MAP.items():
        if match_keywords(text_blob, keywords):
            matched.append(target_cat)
    
    if len(matched) == 1:
        return matched[0]
    return None

def main():
    print("Loading youtube_data.csv...")
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'youtube_data.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Filter duration 10-60s
    print("Filtering by duration (10-60s)...")
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna(subset=['duration'])
    df = df[(df['duration'] >= 10) & (df['duration'] <= 60)]
    
    # Classify
    print("Classifying videos...")
    df['mapped_category'] = df.apply(classify_row, axis=1)
    
    # Filter targets
    TARGET_CATEGORIES = [
        "gaming", "education", "technology", "finance", "fitness",
        "cooking", "travel", "music", "comedy", "news", "sports", "beauty"
    ]
    df = df[df['mapped_category'].isin(TARGET_CATEGORIES)]
    
    # Equalize records
    counts = df['mapped_category'].value_counts()
    print("Counts per category:")
    print(counts)
    
    if counts.empty:
        print("No videos found matching criteria.")
        return

    min_count = counts.min()
    print(f"Equalizing to {min_count} records per category.")
    
    sampled_df = df.groupby('mapped_category').sample(n=min_count, random_state=42)
    
    # Prepare list of videos
    video_tasks = []
    for _, row in sampled_df.iterrows():
        video_tasks.append({'video_id': row['video_id'], 'category': row['mapped_category']})

    # Parallel Download
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures
    
    print(f"Downloading {len(video_tasks)} videos in parallel...")
    max_workers = 8 # Adjust based on network
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_video_sync, task): task for task in video_tasks}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(video_tasks)):
            task = futures[future]
            try:
                future.result()
            except Exception as e:
                pass # video_downloader handles errors individually usually

    # Process Embeddings
    print("Initializing ClipEmbedder...")
    embedder = ClipEmbedder()
    
    output_path = config.OUTPUT_DATASET_FILE
    
    # Check existing to resume
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            if 'video_id' in existing_df.columns:
                processed_ids = set(existing_df['video_id'].astype(str))
        except:
            pass
            
    # Open file in append mode if exists, else write header
    write_header = not os.path.exists(output_path)
    
    print("Processing embeddings...")
    processed_count = 0
    
    # We open the file and append line by line or batch
    # To use pandas efficiently, we probably want to collect a batch and write
    
    results_batch = []
    BATCH_SIZE = 10
    
    for idx, row in tqdm.tqdm(sampled_df.iterrows(), total=len(sampled_df)):
        video_id = row['video_id']
        if str(video_id) in processed_ids:
            continue
            
        category = row['mapped_category']
        
        # 1. Check/Get Video Path (should satisfy by now from download step)
        video_path = None
        # Logic to find path (reuse get_video_path logic or blind guess)
        # We can implement a helper or assume structure
        # video_downloader stores in videos/category/video_id.extension
        
        # Try to find file
        exts = ['.mp4', '.mkv', '.webm']
        found_path = None
        cat_dir = os.path.join(config.VIDEOS_DIR, category)
        if os.path.exists(cat_dir):
            for e in exts:
                p = os.path.join(cat_dir, f"{video_id}{e}")
                if os.path.exists(p):
                    found_path = p
                    break
            # wildcard check if not found
            if not found_path:
                for f in os.listdir(cat_dir):
                    if f.startswith(video_id + "."):
                        found_path = os.path.join(cat_dir, f)
                        break
        
        if not found_path:
            # print(f"Video not found for {video_id}")
            continue
            
        try:
            # 2. Extract Frames
            frames = sample_frames(found_path, num_frames=config.FRAME_SAMPLE_COUNT)
            if not frames:
                continue
            
            # 3. Generate Embedding
            embedding = embedder.get_embedding(frames)
            if embedding is None:
                continue
                
            # 4. Store Result
            res = {
                "video_id": video_id,
                "category": category,
            }
            for i, val in enumerate(embedding):
                res[f"embedding_{i}"] = val
            
            results_batch.append(res)
            processed_count += 1
            
            if len(results_batch) >= BATCH_SIZE:
                 batch_df = pd.DataFrame(results_batch)
                 batch_df.to_csv(output_path, mode='a', header=write_header, index=False)
                 write_header = False # Only write header once
                 results_batch = []
                 
        except Exception as e:
            print(f"Error {video_id}: {e}")
            continue

    # Flush remaining
    if results_batch:
        batch_df = pd.DataFrame(results_batch)
        batch_df.to_csv(output_path, mode='a', header=write_header, index=False)
    
    print(f"Done. Processed {processed_count} new videos.")

if __name__ == "__main__":
    main()
