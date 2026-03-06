import pandas as pd
from googleapiclient.discovery import build
import isodate
import time
from tqdm import tqdm
import config
import os

class YouTubeCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=self.api_key)

    def search_videos_by_category(self, category):
        """
        Searches for videos in a category that meet the duration criteria.
        Returns a list of dictionaries with metadata.
        """
        videos = []
        query = config.SEARCH_QUERIES.get(category, category)
        print(f"Collecting videos for '{category}' (Query: '{query}')...")
        
        next_page_token = None
        pbar = tqdm(total=config.VIDEOS_PER_CATEGORY, unit="video")

        while len(videos) < config.VIDEOS_PER_CATEGORY:
            try:
                # 1. Search for videos
                search_response = self.youtube.search().list(
                    q=query,
                    type="video",
                    part="id,snippet",
                    maxResults=50,
                    videoDuration="short", # Optimization: filter by short first, though 'short' is < 4 mins
                    pageToken=next_page_token
                ).execute()

                video_ids = [item['id']['videoId'] for item in search_response['items']]
                if not video_ids:
                    print(f"No more videos found for {category}.")
                    break

                # 2. Get content details to check precise duration
                details_response = self.youtube.videos().list(
                    part="contentDetails",
                    id=",".join(video_ids)
                ).execute()

                for item in details_response['items']:
                    video_id = item['id']
                    duration_iso = item['contentDetails']['duration']
                    duration_seconds = isodate.parse_duration(duration_iso).total_seconds()

                    if config.MIN_DURATION <= duration_seconds <= config.MAX_DURATION:
                        # Find the title from the search response
                        # Note: This is slightly inefficient as we iterate the search response for each detail item
                        # but given batch size of 50, it is negligible.
                        title = next((s['snippet']['title'] for s in search_response['items'] if s['id']['videoId'] == video_id), "Unknown")
                        
                        videos.append({
                            "video_id": video_id,
                            "title": title,
                            "category": category,
                            "duration": duration_seconds
                        })
                        pbar.update(1)

                        if len(videos) >= config.VIDEOS_PER_CATEGORY:
                            break
                
                next_page_token = search_response.get("nextPageToken")
                if not next_page_token:
                    print(f"End of results for {category}.")
                    break
                
                # Check quota or rate limits if necessary, sleep a bit to be polite
                # time.sleep(0.1) 

            except Exception as e:
                print(f"Error during API call for {category}: {e}")
                time.sleep(5)  # Wait and retry on error
                
        pbar.close()
        return videos

    def collect_all_categories(self):
        all_videos = []
        
        # Check if metadata file already exists to avoid restarting
        if os.path.exists(config.METADATA_FILE):
             print(f"Found existing metadata file at {config.METADATA_FILE}. Loading...")
             try:
                 df = pd.read_csv(config.METADATA_FILE)
                 existing_ids = set(df['video_id'].unique())
                 print(f"Loaded {len(df)} existing videos.")
                 # Decide if we want to add to it or overwrite. For safety, let's load it and return.
                 # If the user wants to resume, logic would be more complex.
                 # Simple approach: If it exists, return it, let user delete it to restart.
                 return df
             except Exception as e:
                 print(f"Error reading existing metadata: {e}. Starting fresh.")

        for category in config.CATEGORIES:
            videos = self.search_videos_by_category(category)
            all_videos.extend(videos)
            print(f"Total valid videos for {category}: {len(videos)}")

        df = pd.DataFrame(all_videos)
        if not df.empty:
            df.to_csv(config.METADATA_FILE, index=False)
            print(f"Saved metadata to {config.METADATA_FILE}")
        else:
            print("No videos collected.")
        
        return df

if __name__ == "__main__":
    if config.YOUTUBE_API_KEY == "YOUR_API_KEY_HERE":
        print("Please set your YOUTUBE_API_KEY in config.py")
    else:
        collector = YouTubeCollector(config.YOUTUBE_API_KEY)
        collector.collect_all_categories()
