import cv2
import numpy as np
import config
import os
import tqdm

def sample_frames(video_path, num_frames=config.FRAME_SAMPLE_COUNT):
    """
    Extracts frames uniformly from a video.
    Returns a list of numpy arrays (frames).
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If video is extremely short or corrupt, return empty
    if total_frames <= 0:
        print(f"Empty or corrupted video: {video_path}")
        cap.release()
        return []

    # Calculate indices for uniform sampling
    if total_frames < num_frames:
        # If fewer frames than requested, take all and pad? Or just take all available.
        # Repeating frames is safer to keep consistent count if strict size is needed.
        # But for embedding average, taking all available is fine.
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            print(f"Failed to read frame at index {idx} in {video_path}")

    cap.release()
    return frames
