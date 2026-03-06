import pandas as pd
import numpy as np
import os
import tqdm
import config
from frame_sampler import sample_frames
from clip_embedder import ClipEmbedder
from PIL import Image

def build_dataset(metadata_file=config.METADATA_FILE, output_file=config.OUTPUT_DATASET_FILE):
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found.")
        return

    # Load metadata
    try:
        df = pd.read_csv(metadata_file)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Check for existing dataset to resume
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            processed_ids = set(existing_df['video_id'].unique())
            print(f"Found existing dataset with {len(processed_ids)} processed videos. Resuming...")
        except Exception as e:
            print(f"Error reading existing dataset, starting fresh: {e}")

    # Identify videos to process
    videos_to_process = df[~df['video_id'].isin(processed_ids)]

    if videos_to_process.empty:
        print("All videos in metadata have been processed!")
        return

    print(f"Processing {len(videos_to_process)} videos...")

    # Initialize Embedder (this loads the model and might take time/memory)
    embedder = ClipEmbedder()

    # Iterate and process
    # We will buffer results and write in chunks to avoid memory issues and data loss
    results = []
    CHUNK_SIZE = 100

    pbar = tqdm.tqdm(total=len(videos_to_process), unit="video")
    
    for index, row in videos_to_process.iterrows():
        video_id = row['video_id']
        category = row['category']
        
        # Construct video path
        video_path = os.path.join(config.VIDEOS_DIR, category, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            # Try checking without extension or other extensions if strict path fails
            # But downloader forced mp4, so let's stick to that or Log warning
            # Actually, sometimes yt-dlp merges into mkv if mp4 merge fails. 
            # Let's check a few extensions.
            found = False
            for ext in ['.mp4', '.mkv', '.webm']:
                 temp_path = os.path.join(config.VIDEOS_DIR, category, f"{video_id}{ext}")
                 if os.path.exists(temp_path):
                     video_path = temp_path
                     found = True
                     break
            
            if not found:
                pbar.write(f"Video file not found for ID {video_id} in {category}. Skipping.")
                pbar.update(1)
                continue

        try:
            frames = sample_frames(video_path, num_frames=config.FRAME_SAMPLE_COUNT)
            if not frames:
                pbar.write(f"No frames extracted for {video_id}. Skipping.")
                pbar.update(1)
                continue
                
            embedding = embedder.get_embedding(frames)
            
            if embedding is not None:
                # Create result dictionary
                # Flatten embedding to columns embedding_0, embedding_1...
                res = {
                    "video_id": video_id,
                    "category": category
                }
                for i, val in enumerate(embedding):
                    res[f"embedding_{i}"] = val
                
                results.append(res)

        except Exception as e:
            pbar.write(f"Error processing video {video_id}: {e}")

        pbar.update(1)

        # Write chunk
        if len(results) >= CHUNK_SIZE:
             _save_chunk(results, output_file)
             results = []

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
