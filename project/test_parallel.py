import os
import pandas as pd
import config
from youtube_collector import YouTubeCollector
from parallel_main import run_pipeline_parallel
import shutil

def run_test():
    print("=== STARTING PARALLEL PIPELINE TEST ===")
    
    # 1. Monkey-patch configuration for testing
    print("[TEST] Configuring test parameters...")
    
    # Use a separate test directory
    TEST_DIR = os.path.join(config.BASE_DIR, "test_parallel_output")
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Monkey-patch config settings BEFORE they are used by any imported module functions
    config.VIDEOS_PER_CATEGORY = 2 # Test with 2 videos
    config.CATEGORIES = ["gaming"] # Just test one category
    
    TEST_METADATA_FILE = os.path.join(TEST_DIR, "test_metadata.csv")
    TEST_DATASET_FILE = os.path.join(TEST_DIR, "test_dataset.csv")
    TEST_VIDEOS_DIR = os.path.join(TEST_DIR, "videos")
    
    # Override global config paths
    config.METADATA_FILE = TEST_METADATA_FILE
    config.OUTPUT_DATASET_FILE = TEST_DATASET_FILE
    config.VIDEOS_DIR = TEST_VIDEOS_DIR
    
    # 2. collect dummy metadata
    print("\n[TEST] 1. Creating dummy metadata (or real collection if needed)...")
    if config.YOUTUBE_API_KEY == "YOUR_API_KEY_HERE":
        print("Please set your YOUTUBE_API_KEY in config.py")
        return

    collector = YouTubeCollector(config.YOUTUBE_API_KEY)
    df = collector.collect_all_categories()
    
    if df.empty:
        print("[FAILED] Collector returned empty dataframe.")
        return

    print(f"[SUCCESS] Collected {len(df)} videos.")

    # 3. Run Parallel Pipeline
    print("\n[TEST] 2. Running Parallel Pipeline...")
    # This function uses the config we patched
    run_pipeline_parallel(max_download_workers=2)

    # 4. Validate
    print("\n[TEST] 3. Validating Results...")
    if not os.path.exists(TEST_DATASET_FILE):
        print("[FAILED] Dataset CSV not created.")
        return
        
    result_df = pd.read_csv(TEST_DATASET_FILE)
    print(f"[SUCCESS] Dataset created with shape: {result_df.shape}")
    print("Columns:", result_df.columns.tolist()[:5], "...")
    
    if result_df.shape[0] > 0:
        print("\n=== TEST PASSED SUCCESSFULLY ===")
    else:
        print("\n[FAILED] Dataset is empty.")

if __name__ == "__main__":
    run_test()
