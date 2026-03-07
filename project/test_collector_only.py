import sys
import os
import pandas as pd

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config
    # Patch config for testing
    config.VIDEOS_PER_CATEGORY = 1
    config.CATEGORIES = ["gaming"]
    # Ensure directory exists or use current dir properly
    if not os.path.exists("test_output"):
        os.makedirs("test_output")
    config.METADATA_FILE = os.path.join("test_output", "test_collector_metadata.csv")

    from youtube_collector import YouTubeCollector
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_collector():
    print("Testing YouTubeCollector...")
    
    if os.path.exists(config.METADATA_FILE):
        os.remove(config.METADATA_FILE)
        
    collector = YouTubeCollector()
    print("Collector initialized.")
    
    # We will test collect_all_categories
    print("Running collect_all_categories...")
    df = collector.collect_all_categories()
    
    print(f"Collection finished. DataFrame shape: {df.shape}")
    print(df)
    
    if not df.empty:
        print("Success! Videos collected.")
    else:
        print("Warning: No videos collected.")

    # Clean up
    if os.path.exists(config.METADATA_FILE):
        os.remove(config.METADATA_FILE)

if __name__ == "__main__":
    test_collector()