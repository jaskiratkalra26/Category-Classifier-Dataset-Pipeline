import time
import pandas as pd
import config
import os
from youtube_collector import YouTubeCollector
from video_downloader import download_videos
from dataset_builder import build_dataset
from googleapiclient.errors import HttpError

BATCH_SIZE = 10  # Number of videos to collect per cycle per category
SLEEP_TIME = 10  # Short sleep between cycles if API is healthy
API_COOLDOWN = 600 # 10 minutes sleep for API if quota/rate limit hit

def run_incremental():
    print("=== Incremental Pipeline Started ===")
    
    # Initialize collector
    try:
        collector = YouTubeCollector()
    except Exception as e:
        print(f"Failed to initialize collector: {e}")
        return

    api_paused = False
    api_resume_time = 0

    # Track consecutive empty results per category to avoid infinite API hammering
    consecutive_empty_cycles = {cat: 0 for cat in config.CATEGORIES}
    stalled_categories = set()
    MAX_CONSECUTIVE_EMPTY = 3

    while True:
        videos_collected_this_cycle = 0
        all_categories_complete = True # Assume complete until proven otherwise

        # 1. Collection Phase (Only if API is available)
        if api_paused:
            remaining = int(api_resume_time - time.time())
            if remaining > 0:
                print(f"\n[API Paused] Skipping collection due to quota/rate limit. Resuming in {remaining}s...")
            else:
                print("\n[API Resumed] Attempting collection again...")
                api_paused = False

        if not api_paused:
            print("\n--- Phase 1: Collection ---")
            try:
                # Let's inspect current metadata state
                if os.path.exists(config.METADATA_FILE):
                    try:
                        df = pd.read_csv(config.METADATA_FILE)
                    except:
                        df = pd.DataFrame(columns=['video_id', 'title', 'category', 'duration'])
                else:
                    df = pd.DataFrame(columns=['video_id', 'title', 'category', 'duration'])

                for category in config.CATEGORIES:
                    if category in stalled_categories:
                        print(f"Skipping stalled category '{category}' (search exhausted).")
                        continue

                    if 'category' in df.columns:
                        current_count = len(df[df['category'] == category])
                    else:
                        current_count = 0
                    
                    if current_count >= config.VIDEOS_PER_CATEGORY:
                        print(f"Category '{category}' is complete ({current_count}/{config.VIDEOS_PER_CATEGORY}).")
                        continue
                    
                    all_categories_complete = False
                    needed = config.VIDEOS_PER_CATEGORY - current_count
                    to_collect = min(needed, BATCH_SIZE)
                    
                    print(f"Collection needed for '{category}': {to_collect} more videos in this batch.")
                    
                    # Fetch
                    try:
                        existing_ids = set(df['video_id'].unique()) if not df.empty else set()
                        new_videos = collector.collect_videos_for_category(
                            category, 
                            max_needed=to_collect, 
                            existing_ids=existing_ids
                        )
                        
                        if new_videos:
                            videos_collected_this_cycle += len(new_videos)
                            new_df = pd.DataFrame(new_videos)
                            
                            # Save immediately
                            if not os.path.exists(config.METADATA_FILE):
                                new_df.to_csv(config.METADATA_FILE, index=False)
                            else:
                                new_df.to_csv(config.METADATA_FILE, mode='a', header=False, index=False)
                            
                            # Update local df for next iteration in loop
                            df = pd.concat([df, new_df], ignore_index=True)
                            print(f"Saved {len(new_videos)} videos for {category}.")
                            consecutive_empty_cycles[category] = 0 # Reset counter on success
                        else:
                            print(f"No videos found for {category} in this attempt.")
                            consecutive_empty_cycles[category] += 1
                            if consecutive_empty_cycles[category] >= MAX_CONSECUTIVE_EMPTY:
                                print(f"WARNING: Category '{category}' has returned no videos for {MAX_CONSECUTIVE_EMPTY} consecutive cycles. Marking as stalled.")
                                stalled_categories.add(category)
                            
                    except HttpError as e:
                        if e.resp.status in [403, 429]:
                            print(f"API Quota/Rate Limit hit: {e}")
                            print(f"Pausing API collection for {API_COOLDOWN} seconds.")
                            api_paused = True
                            api_resume_time = time.time() + API_COOLDOWN
                            break # Break category loop
                        else:
                            print(f"API Error for {category}: {e}")
                    except Exception as e:
                        print(f"Error collecting {category}: {e}")

            except Exception as e:
                print(f"Unexpected error during collection phase: {e}")
        
        # Check if we are totally done with collection
        if all_categories_complete and not api_paused:
             print("\nAll categories collected! Proceeding to final processing...")
             # We don't break yet, we ensure everything is processed first

        # 2. Download Phase (Always run, even if collection skipped)
        # Check if we have videos to download
        # (Naive check: total metadata rows > 0)
        try:
             if os.path.exists(config.METADATA_FILE):
                df = pd.read_csv(config.METADATA_FILE)
                print(f"\n--- Phase 2: Downloading (Total Metadata: {len(df)}) ---")
                download_videos(config.METADATA_FILE)
        except Exception as e:
             print(f"Download error: {e}")

        # 3. Build/Embed Phase (Always run)
        print("\n--- Phase 3: Building Dataset ---")
        try:
             build_dataset(config.METADATA_FILE, config.OUTPUT_DATASET_FILE)
        except Exception as e:
             print(f"Build error: {e}")

        # Final Exit Condition
        # If all collected AND all processed (implied by build_dataset completing fully), we can stop.
        # But build_dataset might not signal completion easily here without reading its output again.
        # For now, let's keep looping until user interrupts, or maybe check if dataset == metadata length
        
        if all_categories_complete and not api_paused:
             # Check if processing is done
             if os.path.exists(config.OUTPUT_DATASET_FILE) and os.path.exists(config.METADATA_FILE):
                 meta_df = pd.read_csv(config.METADATA_FILE)
                 out_df = pd.read_csv(config.OUTPUT_DATASET_FILE)
                 if len(out_df) >= len(meta_df):
                     print("\nAll videos collected and processed! Pipeline finished.")
                     break

        # Sleep Logic
        if api_paused:
            print(f"\nCycle complete. API is paused. Sleeping short ({SLEEP_TIME}s) before checking processing again...")
        else:
            print(f"\nCycle complete. Sleeping for {SLEEP_TIME} seconds...")
            
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    run_incremental()