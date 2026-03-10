import pandas as pd
import os
import tqdm
import sys
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
import yt_dlp
import numpy as np
import traceback

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

def process_category_batch(category, target_count, output_dir, embedder, results_list, output_path, known_ids):
    """
    Process a single category by performing a BATCH search and processing results.
    Returns the number of successfully processed videos.
    """
    # print(f"\nProcessing category: {category}")
    cat_dir = os.path.join(config.VIDEOS_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)
    
    # Try multiple search variations to find enough videos
    search_queries = [
        f"{category} shorts",
        f"best {category} shorts",
        f"viral {category} shorts",
        f"new {category} shorts",
        f"{category} compilation shorts",
        f"top {category} shorts",
        f"trending {category} shorts",
        f"funny {category} moments",
        f"{category} 2024",
        f"{category} 2025",
    ]

    # Add specific queries for hard-to-fill categories
    if category.lower() == "gaming":
        search_queries.extend([
            "minecraft shorts", "roblox shorts", "fortnite shorts", 
            "gta 5 shorts", "valorant shorts", "call of duty shorts",
            "gaming memes", "league of legends shorts", "among us shorts"
        ])
    elif category.lower() == "news":
        search_queries.extend(["breaking news shorts", "world news shorts", "daily news shorts", "politics shorts"])
    elif category.lower() == "cooking":
        search_queries.extend(["easy recipes shorts", "street food shorts", "food hacks shorts", "baking shorts"])
    elif category.lower() == "sports":
        search_queries.extend(["nba shorts", "football shorts", "soccer shorts", "ufc shorts", "cricket shorts", "gymnastics shorts"])
    elif category.lower() == "technology":
        search_queries.extend(["tech reviews shorts", "new gadgets shorts", "pc build shorts", "smartphone review shorts", "future tech shorts", "coding shorts"])
    elif category.lower() == "finance":
        search_queries.extend(["investing tips shorts", "stock market shorts", "crypto news shorts", "passive income shorts", "money saving tips shorts"])
    elif category.lower() == "fitness":
        search_queries.extend(["workout routine shorts", "gym motivation shorts", "calisthenics shorts", "yoga shorts", "weight loss tips shorts"])
    elif category.lower() == "travel":
        search_queries.extend(["travel guide shorts", "beautiful places shorts", "luxury travel shorts", "solo travel shorts", "backpacking shorts"])
    elif category.lower() == "music":
        search_queries.extend(["live concert shorts", "music cover shorts", "guitar solo shorts", "piano shorts", "rap freestyle shorts", "singer shorts"])
    elif category.lower() == "comedy":
        search_queries.extend(["skit shorts", "stand up comedy shorts", "prank shorts", "relatable shorts", "funny animals shorts"])
    elif category.lower() == "beauty":
        search_queries.extend(["makeup tutorial shorts", "skincare routine shorts", "fashion trends shorts", "hair styling shorts", "outfit ideas shorts"])
    elif category.lower() == "education":
        search_queries.extend(["science facts shorts", "history facts shorts", "learn english shorts", "math hacks shorts", "psychology facts shorts", "did you know shorts"])
    
    total_processed_for_category = 0

    for idx, search_query in enumerate(search_queries):
        # Check how many we already have for this category in the current session
        current_count = sum(1 for r in results_list if r['category'] == category)
        needed = target_count - current_count
        
        if needed <= 0:
            break

        # Search for more candidates than needed
        # Ensure we ask for enough to cover failures
        search_limit = int(needed * 3) + 20
        
        # Hard cap to prevent extreme queries, but must be at least needed + buffer
        if search_limit > 500: search_limit = 500
        if search_limit < 50: search_limit = 50
        
        # print(f"Searching for up to {search_limit} candidates for '{search_query}'...")

        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': os.path.join(cat_dir, '%(id)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'max_filesize': 50 * 1024 * 1024,
            'match_filter': yt_dlp.utils.match_filter_func("duration >= 5 & duration <= 65"),
            'ignoreerrors': True,
            'socket_timeout': 30,
            'retries': 10,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 1. BATCH SEARCH
                try:
                    result = ydl.extract_info(f"ytsearch{search_limit}:{search_query}", download=False)
                except Exception as e:
                    print(f"Search failed for '{search_query}': {e}")
                    continue

                if not result:
                    continue
                    
                entries = result.get('entries', [])
                if not entries:
                    continue
                
                entries = list(entries)

                # 2. Iterate and Process
                processed_in_query = 0
                for video_info in entries:
                    if sum(1 for r in results_list if r['category'] == category) >= target_count:
                        break
                        
                    if not video_info: continue
                    video_id = video_info.get('id')
                    if not video_id: continue
                    
                    # Check if this ID is already in our results
                    if video_id in known_ids or any(r['video_id'] == video_id for r in results_list):
                        continue

                    # Prepare download path
                    video_path = None
                    base_path = os.path.join(cat_dir, video_id)
                    
                    # Check if file exists
                    for ext in ['.mp4', '.mkv', '.webm']:
                         if os.path.exists(base_path + ext):
                             video_path = base_path + ext
                             break
                    
                    # Download if not exists
                    if not video_path:
                        try:
                            ydl.download([video_info.get('webpage_url', f'https://youtu.be/{video_id}')])
                            for ext in ['.mp4', '.mkv', '.webm']:
                                if os.path.exists(base_path + ext):
                                    video_path = base_path + ext
                                    break
                            time.sleep(random.uniform(2, 5))
                        except Exception as e:
                            print(f"Download failed for {video_id}: {e}")
                            time.sleep(5)
                            continue
                    
                    if not video_path:
                        continue

                    # Sample Frames
                    try:
                        frames = sample_frames(video_path, num_frames=config.FRAME_SAMPLE_COUNT)
                    except Exception as e:
                        print(f"Frame error {video_id} (deleting): {e}")
                        try: os.remove(video_path) 
                        except: pass
                        continue
                        
                    if not frames:
                        print(f"No frames from {video_id} (deleting)")
                        try: os.remove(video_path) 
                        except: pass
                        continue

                    # Embed
                    try:
                        embedding = embedder.get_embedding(frames)
                    except Exception as e:
                         print(f"Embed error {video_id}: {e}")
                         continue

                    if embedding is None:
                        continue

                    # Add to results
                    res = {
                        "video_id": video_id,
                        "category": category,
                        "original_title": video_info.get('title', 'Unknown'),
                        "dataset_title": f"SEARCH_{category}", 
                        "original_author": video_info.get('uploader', 'Unknown')
                    }
                    for i, val in enumerate(embedding):
                        res[f"embedding_{i}"] = val
                    
                    results_list.append(res)
                    total_processed_for_category += 1
                    processed_in_query += 1
                    
                    # Update known_ids so duplicates within session (diff queries) are caught
                    known_ids.add(video_id)
                    
                    # Checkpoint save
                    if len(results_list) % 5 == 0:
                        pd.DataFrame(results_list).to_csv(output_path, index=False)
        
        except Exception as e:
            print(f"Error processing query '{search_query}': {e}")
            traceback.print_exc()

    return total_processed_for_category

def main():
    print("Starting optimized dataset collection (Batch Search Mode)...")
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Download and process shorts dataset.")
    parser.add_argument("--count", type=int, default=50, help="Target number of videos per category (default: 50)")
    parser.add_argument("--category", type=str, help="Specific category to process (optional)")
    args = parser.parse_args()
    
    # Initialize Embedder
    print("Initializing ClipEmbedder...")
    try:
        embedder = ClipEmbedder()
    except Exception as e:
        print(f"Failed to init embedder: {e}")
        return

    # Output file
    output_path = config.OUTPUT_DATASET_FILE
    
    results = []
    if os.path.exists(output_path):
        try:
            print(f"Resuming from existing file: {output_path}")
            # Ensure video_id is read as string
            df_existing = pd.read_csv(output_path, dtype={'video_id': str})
            results = df_existing.to_dict('records')
            print(f"Loaded {len(results)} existing records.")
        except Exception as e:
            print(f"Error reading existing file: {e}. Starting fresh.")
            # If read fails, maybe backup
            backup_path = output_path + ".corrupt"
            try:
                os.rename(output_path, backup_path)
                print(f"Backed up corrupt file to {backup_path}")
            except: pass
    else:
        print("Starting fresh dataset.")
    
    # Target count per category (from CLI or default)
    target_count = args.count 
    print(f"Targeting {target_count} videos per category. Filling gaps in existing dataset...")
    
    total_processed = 0
    
    # Pre-compute set of known IDs to avoid re-processing ANY video
    known_video_ids = set(r['video_id'] for r in results if 'video_id' in r)
    print(f"Tracking {len(known_video_ids)} unique video IDs to skip.")
    
    # Determine categories to process
    categories_to_process = TARGET_CATEGORIES
    if args.category:
        if args.category in TARGET_CATEGORIES:
            categories_to_process = [args.category]
            print(f"Filtering for single category: {args.category}")
        else:
            print(f"Warning: Category '{args.category}' not in standard list. Processing anyway.")
            categories_to_process = [args.category]

    try:
        # Iterate over CATEGORIES directly with progress bar
        pbar = tqdm.tqdm(categories_to_process)
        for category in pbar:
            pbar.set_description(f"Processing {category}")
            count = process_category_batch(category, target_count, config.VIDEOS_DIR, embedder, results, output_path, known_video_ids)
            total_processed += count
            # print(f"Finished {category}: Added {count} videos.")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nCRITICAL ERROR in main loop: {e}")
        traceback.print_exc()
    finally:
        # Final Save
        if results:
            print(f"Saving {len(results)} total records to {output_path}")
            pd.DataFrame(results).to_csv(output_path, index=False)
        print("Done.")

if __name__ == "__main__":
    main()
