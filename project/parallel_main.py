import pandas as pd
import config
from youtube_collector import YouTubeCollector
from video_downloader import download_video_sync
from dataset_builder import process_single_video, _save_chunk
from clip_embedder import ClipEmbedder
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import queue
import time
import threading
from tqdm import tqdm

def download_manager(videos_list, q, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(download_video_sync, vid): vid 
            for vid in videos_list
        }
        
        for future in as_completed(future_to_video):
            vid_info = future_to_video[future]
            try:
                path = future.result()
                if path:
                    # Success: Put into queue for embedding
                    msg = vid_info.copy()
                    msg['path'] = path
                    q.put(msg)
                else:
                    # Download failed
                    # Pass a special "skipped" message so the embedder progress bar updates
                    q.put({"status": "skipped"})
            except Exception as e:
                print(f"Download exception: {e}")
                q.put({"status": "error"})
        
        # Signal done
        q.put(None)

def run_pipeline_parallel(max_download_workers=4):
    print("=== Pipeline Started (Concurrent Mode) ===")
    
    # Check if API key is set
    if config.YOUTUBE_API_KEY == "YOUR_API_KEY_HERE" and not os.path.exists(config.METADATA_FILE):
         print("Please set your YOUTUBE_API_KEY in config.py")
         return

    # 1. Collection (Must be serial to get the list first)
    if not os.path.exists(config.METADATA_FILE):
        print("Collecting metadata...")
        collector = YouTubeCollector(config.YOUTUBE_API_KEY)
        df_metadata = collector.collect_all_categories()
    else:
        df_metadata = pd.read_csv(config.METADATA_FILE)

    # 2. Setup processing queues and trackers
    
    # Filter out already processed videos in the final dataset
    processed_ids = set()
    if os.path.exists(config.OUTPUT_DATASET_FILE):
        try:
             existing = pd.read_csv(config.OUTPUT_DATASET_FILE)
             if not existing.empty:
                processed_ids = set(existing['video_id'].unique())
        except:
            pass

    # Items to process: Videos in metadata that are NOT in dataset
    videos_to_process = df_metadata[~df_metadata['video_id'].isin(processed_ids)].to_dict('records')
    
    if not videos_to_process:
        print("All videos processed!")
        return

    print(f"Total videos to process: {len(videos_to_process)}")

    # Initialize Embedder (Main Thread - GPU/CPU)
    print("Initializing CLIP Embedder...")
    embedder = ClipEmbedder()

    # Queue for videos that are downloaded and ready for embedding
    # We will pass dicts here: {"video_id": "...", "path": "..."} OR {"status": "skipped"}
    ready_queue = queue.Queue()
    
    # Progress Bar
    pbar = tqdm(total=len(videos_to_process), unit="video", desc="Processing")

    # Start the download manager in a background thread
    manager_thread = threading.Thread(target=download_manager, args=(videos_to_process, ready_queue, max_download_workers))
    manager_thread.start()

    # Main Thread: Embedding Loop (Consumer)
    buffer = []
    CHUNK_SIZE = 50
    
    while True:
        try:
            # Block until an item is available
            item = ready_queue.get(timeout=2)
            
            if item is None:
                # Sentinel received, downloads done
                break

            # Handle status messages for skipped/failed downloads
            if "status" in item and item["status"] in ["skipped", "error", "failed"]:
                pbar.update(1)
                continue

            # Process (Generate Embedding)
            # Use imported process_single_video
            result = process_single_video(item['video_id'], item['category'], embedder)
            
            if result:
                buffer.append(result)
            
            pbar.update(1)
            
            if len(buffer) >= CHUNK_SIZE:
                _save_chunk(buffer, config.OUTPUT_DATASET_FILE)
                buffer = []
                
        except queue.Empty:
            if not manager_thread.is_alive():
                # If download manager is dead and queue is empty, we are done
                break
            continue
        except Exception as e:
            print(f"Error in embedding loop: {e}")

    # Save remaining
    if buffer:
        _save_chunk(buffer, config.OUTPUT_DATASET_FILE)
        
    pbar.close()
    manager_thread.join()
    print("Pipeline completed.")

if __name__ == "__main__":
    run_pipeline_parallel()