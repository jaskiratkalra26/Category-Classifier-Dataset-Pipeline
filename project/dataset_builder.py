import pandas as pd
import numpy as np
import os
import tqdm
import config
from frame_sampler import sample_frames
from clip_embedder import ClipEmbedder
from PIL import Image

def _mark_failed(video_id):
    """Marks a video as failed so it's skipped in future runs."""
    try:
        with open(config.FAILED_VIDEOS_FILE, "a") as f:
            f.write(video_id + "\n")
    except Exception as e:
        print(f"Warning: Could not write to failed_videos.txt: {e}")

def process_single_video(video_id, category, embedder):
    """
    Processes a single video: extracting frames, generating embedding, returning dict.
    Returns None if failed.
    """
    # Find video
    video_path = None
    category_dir = os.path.join(config.VIDEOS_DIR, category)
    if os.path.exists(category_dir):
        # Check by ID
        for f in os.listdir(category_dir):
            if f.startswith(video_id + "."):
                video_path = os.path.join(category_dir, f)
                break
    
    if not video_path:
        print(f"[DEBUG] Video path not found for ID: {video_id} in {category_dir}")
        _mark_failed(video_id)
        return None

    try:
        frames = sample_frames(video_path, num_frames=config.FRAME_SAMPLE_COUNT)
        if not frames:
            print(f"[DEBUG] No frames extracted for {video_id} from {video_path}")
            return None
            
        embedding = embedder.get_embedding(frames)
        
        if embedding is None:
            print(f"[DEBUG] Embedding generation returned None for {video_id}")
            return None
            
        print(f"[DEBUG] Success processing {video_id}. Embedding shape: {embedding.shape}")
        
        # Create result dictionary
        res = {
            "video_id": video_id,
            "category": category
        }
        # Add embedding fields
        for i, val in enumerate(embedding):
            res[f"embedding_{i}"] = val
        return res

    except Exception as e:
        print(f"Error processing {video_id}: {e}")
        return None
    
    return None

def build_dataset(metadata_file=config.METADATA_FILE, output_file=config.OUTPUT_DATASET_FILE):
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found.")
        return

    # Load metadata
    try:
        df = pd.read_csv(metadata_file, dtype={'video_id': str})
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Check for existing dataset to resume
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            # Enforce string type for video_id to ensure matching works correctly
            existing_df = pd.read_csv(output_file, dtype={'video_id': str})
            processed_ids = set(existing_df['video_id'].unique())
            print(f"Found existing dataset with {len(processed_ids)} processed videos. Resuming...")
        except Exception as e:
            print(f"Error reading existing dataset (might be empty or corrupted), starting fresh: {e}")

    # Load failed IDs
    failed_ids = set()
    if os.path.exists(config.FAILED_VIDEOS_FILE):
        try:
            with open(config.FAILED_VIDEOS_FILE, 'r') as f:
                failed_ids = set(line.strip() for line in f if line.strip())
            print(f"Found {len(failed_ids)} previously failed videos. Skipping...")
        except Exception as e:
            print(f"Error reading failed videos list: {e}")

    # Identify videos to process
    # Ensure strict string comparison
    df['video_id'] = df['video_id'].astype(str)
    # Filter out both processed AND failed videos
    videos_to_process = df[~df['video_id'].isin(processed_ids) & ~df['video_id'].isin(failed_ids)]

    if videos_to_process.empty:
        print("All videos in metadata have been processed!")
        return

    print(f"Processing {len(videos_to_process)} videos...")

    # Initialize Embedder (this loads the model and might take time/memory)
    embedder = ClipEmbedder()

    # Iterate and process
    results = []
    CHUNK_SIZE = 1  # Save after every video to prevent data loss

    pbar = tqdm.tqdm(total=len(videos_to_process), unit="video")
    
    for index, row in videos_to_process.iterrows():
        res = process_single_video(row['video_id'], row['category'], embedder)
        
        if res:
             results.append(res)
        
        # Write chunk immediately if we have results
        if len(results) >= CHUNK_SIZE:
             _save_chunk(results, output_file)
             results = []
             
        pbar.update(1)

    # Save remaining
    if results:
        _save_chunk(results, output_file)

    pbar.close()
    print(f"Dataset building complete. Saved to {output_file}")


def _save_chunk(data, filepath):
    """Appends a list of dicts to CSV."""
    df_chunk = pd.DataFrame(data)
    if not os.path.exists(filepath):
        df_chunk.to_csv(filepath, index=False)
    else:
        df_chunk.to_csv(filepath, mode='a', header=False, index=False)

if __name__ == "__main__":
    build_dataset()
