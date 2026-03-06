import os
import shutil
import pandas as pd
import config
from youtube_collector import YouTubeCollector
from video_downloader import download_videos
from dataset_builder import build_dataset

def run_test():
    print("=== STARTING PIPELINE TEST ===")
    
    # 1. Monkey-patch configuration for testing
    print("[TEST] Configuring test parameters...")
    original_videos_per_cat = config.VIDEOS_PER_CATEGORY
    original_categories = config.CATEGORIES
    
    # Use a separate test directory
    TEST_DIR = os.path.join(config.BASE_DIR, "test_output")
    os.makedirs(TEST_DIR, exist_ok=True)
    
    config.VIDEOS_PER_CATEGORY = 1  # Only 1 video
    config.CATEGORIES = ["gaming"]  # Only 1 category
    
    TEST_METADATA_FILE = os.path.join(TEST_DIR, "test_metadata.csv")
    TEST_DATASET_FILE = os.path.join(TEST_DIR, "test_dataset.csv")
    
    # Temporarily override the global metadata file path in config if needed by collector
    # The collector class uses config.METADATA_FILE in collect_all_categories
    config.METADATA_FILE = TEST_METADATA_FILE
    
    try:
        # 2. Test Collector
        print("\n[TEST] 1. Testing YouTube Collector...")
        collector = YouTubeCollector(config.YOUTUBE_API_KEY)
        # We need to ensure we don't load an existing real metadata file by accident
        if os.path.exists(TEST_METADATA_FILE):
             os.remove(TEST_METADATA_FILE)
             
        df = collector.collect_all_categories()
        
        if df.empty:
            print("[FAILED] Collector returned empty dataframe. Check API Key or Quota.")
            return

        print(f"[SUCCESS] Collected {len(df)} videos.")
        print(df.head())

        # 3. Test Downloader
        print("\n[TEST] 2. Testing Video Downloader...")
        # Point the downloader to save in a test folder or just cleanup later
        # The downloader uses config.VIDEOS_DIR. Let's redirect it.
        TEST_VIDEOS_DIR = os.path.join(TEST_DIR, "videos")
        config.VIDEOS_DIR = TEST_VIDEOS_DIR
        
        download_videos(TEST_METADATA_FILE)
        
        # Verify video exists
        video_id = df.iloc[0]['video_id']
        category = df.iloc[0]['category']
        expected_video_path = os.path.join(TEST_VIDEOS_DIR, category, f"{video_id}.mp4")
        
        # Video might have merged to mkv or webm, check broadly
        found_video = False
        for ext in ['.mp4', '.mkv', '.webm']:
            if os.path.exists(os.path.join(TEST_VIDEOS_DIR, category, f"{video_id}{ext}")):
                found_video = True
                expected_video_path = os.path.join(TEST_VIDEOS_DIR, category, f"{video_id}{ext}")
                break
                
        if not found_video:
            print(f"[FAILED] Video file not found at {expected_video_path}")
            # Depending on network/yt-dlp, this might fail. We continue to see if we can handle it.
            return
        
        print(f"[SUCCESS] Video downloaded to {expected_video_path}")

        # 4. Test Dataset Builder (Frames + CLIP)
        print("\n[TEST] 3. Testing Dataset Builder (Frame Extraction + CLIP)...")
        # We need to make sure dataset_builder uses our TEST_VIDEOS_DIR
        # dataset_builder imports config, and we already monkey-patched config.VIDEOS_DIR
        
        if os.path.exists(TEST_DATASET_FILE):
            os.remove(TEST_DATASET_FILE)
            
        build_dataset(TEST_METADATA_FILE, TEST_DATASET_FILE)
        
        if not os.path.exists(TEST_DATASET_FILE):
            print("[FAILED] Dataset CSV not created.")
            return
            
        result_df = pd.read_csv(TEST_DATASET_FILE)
        print(f"[SUCCESS] Dataset created with shape: {result_df.shape}")
        print("Columns:", result_df.columns.tolist()[:5], "...")
        
        # Validation
        if result_df.shape[0] != 1:
            print(f"[WARNING] Expected 1 row, got {result_df.shape[0]}")
        
        if 'embedding_0' not in result_df.columns:
            print("[FAILED] Embeddings not found in output CSV.")
            return

        print("\n=== TEST PASSED SUCCESSFULLY ===")
        print(f"Test artifacts are in: {TEST_DIR}")
        print("You can inspect the CSV and downloaded video there.")

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
