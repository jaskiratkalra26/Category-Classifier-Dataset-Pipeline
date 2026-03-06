import argparse
import config
from youtube_collector import YouTubeCollector
from video_downloader import download_videos
from dataset_builder import build_dataset
import os

def main():
    parser = argparse.ArgumentParser(description="YouTube Category Classification Dataset Pipeline")
    parser.add_argument("--collect", action="store_true", help="Run video collection (YouTube API)")
    parser.add_argument("--download", action="store_true", help="Run video downloading")
    parser.add_argument("--build", action="store_true", help="Build dataset (extract frames & generate embeddings)")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    # Default to all if no specific step is requested
    if not (args.collect or args.download or args.build):
        args.all = True

    print("=== Pipeline Started ===")

    # Step 1: Collection
    if args.all or args.collect:
        print("\n--- Step 1: Collecting Video Metadata ---")
        if config.YOUTUBE_API_KEY == "YOUR_API_KEY_HERE":
            print("ERROR: Please set YOUTUBE_API_KEY in config.py before running collection.")
            return
        
        collector = YouTubeCollector(config.YOUTUBE_API_KEY)
        collector.collect_all_categories()

    # Step 2: Download
    if args.all or args.download:
        print("\n--- Step 2: Downloading Videos ---")
        download_videos(config.METADATA_FILE)

    # Step 3: Build Dataset
    if args.all or args.build:
        print("\n--- Step 3: Building Dataset (Frames -> CLIP -> CSV) ---")
        build_dataset(config.METADATA_FILE, config.OUTPUT_DATASET_FILE)

    print("\n=== Pipeline Completed ===")
    print(f"Dataset location: {config.OUTPUT_DATASET_FILE}")

if __name__ == "__main__":
    main()
