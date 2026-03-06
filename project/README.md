# YouTube Category Classification Dataset Pipeline

## Overview
This project automates the creation of a video classification dataset. It collects YouTube videos from 12 categories, downloads them, extracts frames, and generates CLIP embeddings.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **API Key Configuration:**
    Open `config.py` and set your `YOUTUBE_API_KEY`.
    
    ```python
    YOUTUBE_API_KEY = "YOUR_ACTUAL_API_KEY"
    ```

## Usage

Run the entire pipeline:
```bash
python main.py --all
```

Or run individual steps:

1.  **Collect Metadata** (Search YouTube API):
    ```bash
    python main.py --collect
    ```

2.  **Download Videos**:
    ```bash
    python main.py --download
    ```

3.  **Build Dataset** (Extract frames & Generate Embeddings):
    ```bash
    python main.py --build
    ```

## Testing
To verify the pipeline with a minimal test (1 video), run:
```bash
python test_pipeline.py
```

## Output

-   **Videos**: Stored in `videos/<category>/`
-   **Metadata**: `dataset/videos_metadata.csv`
-   **Final Dataset**: `dataset/video_embeddings.csv`

## Notes
-   The pipeline uses `openai/clip-vit-base-patch32` for embeddings (512 dimensions).
-   Video downloading is handled by `yt-dlp`.
-   **ffmpeg**: If you require higher quality video merging (1080p+), install `ffmpeg` and add it to your system PATH. The default configuration uses standard MP4 downloads that work without ffmpeg.
-   If the script is interrupted, it can be resumed (it checks for existing files).
