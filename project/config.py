import os

# API Configuration
YOUTUBE_API_KEY = "AIzaSyBV5VN40HOWGO9sPUSj3rsESMzRvxs79EQ"

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
METADATA_FILE = os.path.join(DATASET_DIR, "videos_metadata.csv")
OUTPUT_DATASET_FILE = os.path.join(DATASET_DIR, "video_embeddings.csv")

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Dataset Configuration
CATEGORIES = [
    "gaming",
    "education",
    "technology",
    "finance",
    "fitness",
    "cooking",
    "travel",
    "music",
    "comedy",
    "news",
    "sports",
    "beauty"
]

# Map categories to specific search queries for better relevance
SEARCH_QUERIES = {
    "gaming": "gaming gameplay",
    "education": "education tutorial",
    "technology": "technology review",
    "finance": "finance tips",
    "fitness": "fitness workout",
    "cooking": "cooking recipe",
    "travel": "travel vlog",
    "music": "music video",
    "comedy": "comedy sketch",
    "news": "news report",
    "sports": "sports highlights",
    "beauty": "beauty makeup tutorial"
}

VIDEOS_PER_CATEGORY = 1000
FRAME_SAMPLE_COUNT = 12
MIN_DURATION = 10  # seconds
MAX_DURATION = 60  # seconds
