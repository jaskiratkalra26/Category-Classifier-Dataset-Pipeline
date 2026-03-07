import pandas as pd
import time
import os
import config
# Try importing tqdm, fall back to simple range if missing
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import isodate

class YouTubeCollector: 
    def __init__(self, api_key=None): 
        # Check if API key is in config if not passed
        self.api_key = api_key if api_key else config.YOUTUBE_API_KEY
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            print("Warning: API Key not set properly.")
        else:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)

    def _get_video_durations(self, video_ids):
        """
        Fetches duration for a list of video IDs (max 50).
        Returns a dict {video_id: duration_in_seconds}
        """
        durations = {}
        if not video_ids:
            return durations
            
        try:
            # Join IDs for batch request
            request = self.youtube.videos().list(
                part="contentDetails",
                id=",".join(video_ids)
            )
            response = request.execute()

            for item in response.get("items", []):
                vid = item["id"]
                iso_duration = item["contentDetails"]["duration"]
                # Convert ISO 8601 duration to seconds
                try:
                    duration_td = isodate.parse_duration(iso_duration)
                    durations[vid] = duration_td.total_seconds()
                except:
                    durations[vid] = 0
        except HttpError as e:
            print(f"API Error fetching details: {e}")
            if e.resp.status in [403, 429]:
                raise e # Propagate quota errors
        
        return durations

    def collect_videos_for_category(self, category, max_needed, existing_ids=set()):
        """
        Collects metadata using YouTube API for a specific category.
        Stops when max_needed valid videos are found or API limits reached.
        """
        videos = []
        query_term = config.SEARCH_QUERIES.get(category, category)
        print(f"\n[Search] Collecting videos for '{category}' (Query: '{query_term}') using API...")
        
        next_page_token = None
        consecutive_errors = 0
        
        # We request 'short' videos (<4 mins) to increase hit rate for 10-60s target
        
        while len(videos) < max_needed:
            try:
                # Search Request
                # Request roughly 2x needed to account for duplicates and duration filtering
                results_per_page = min(50, max(10, (max_needed - len(videos)) * 3))
                
                search_request = self.youtube.search().list(
                    part="id,snippet",
                    q=query_term,
                    type="video",
                    maxResults=min(50, results_per_page),
                    pageToken=next_page_token,
                    videoDuration='short' 
                )
                search_response = search_request.execute()
                
                # Extract IDs
                items = search_response.get("items", [])
                if not items:
                    print("No more items found.")
                    break
                    
                candidate_ids = []
                candidate_titles = {}
                
                for item in items:
                    vid = item["id"]["videoId"]
                    if vid not in existing_ids and vid not in [v['video_id'] for v in videos]:
                        candidate_ids.append(vid)
                        candidate_titles[vid] = item["snippet"]["title"]
                
                if not candidate_ids:
                    print("No new unique videos in this batch.")
                    next_page_token = search_response.get("nextPageToken")
                    if not next_page_token:
                        break
                    continue

                # Fetch Durations (batch of 50 max)
                # Split candidates into chunks of 50 just in case logic above allows >50
                for i in range(0, len(candidate_ids), 50):
                    batch_ids = candidate_ids[i:i+50]
                    durations_map = self._get_video_durations(batch_ids)
                    
                    # Filter
                    for vid, duration in durations_map.items():
                        if config.MIN_DURATION <= duration <= config.MAX_DURATION:
                            videos.append({
                                "video_id": vid,
                                "title": candidate_titles.get(vid, "Unknown"),
                                "category": category,
                                "duration": duration
                            })
                            if len(videos) >= max_needed:
                                break
                    
                    if len(videos) >= max_needed:
                        break
                
                print(f"  Collected {len(videos)}/{max_needed} valid videos so far...")
                
                next_page_token = search_response.get("nextPageToken")
                if not next_page_token:
                    break
                    
                consecutive_errors = 0
                    
            except HttpError as e:
                print(f"API Error during search: {e}")
                if e.resp.status in [403, 429]:  # Quota or Rate Limit
                    print("Quota exceeded or rate limited.")
                    raise e
                consecutive_errors += 1
                if consecutive_errors > 3:
                     print("Too many consecutive errors. Stopping.")
                     break
                time.sleep(2) # Backoff
            except Exception as e:
                print(f"Error: {e}")
                break
                
        return videos

    def collect_all_categories(self):
        # Ensure metadata file exists with header
        if not os.path.exists(config.METADATA_FILE):
             # Ensure directory exists first
             os.makedirs(os.path.dirname(config.METADATA_FILE), exist_ok=True)
             pd.DataFrame(columns=['video_id', 'title', 'category', 'duration']).to_csv(config.METADATA_FILE, index=False)
             print(f"Created new metadata file at {config.METADATA_FILE}")
            
        # Load existing
        try:
            df = pd.read_csv(config.METADATA_FILE)
            existing_ids = set(df['video_id'].unique())
            print(f"Loaded {len(df)} existing videos.")
        except Exception as e:
            print(f"Error reading metadata: {e}")
            existing_ids = set()
            df = pd.DataFrame(columns=['video_id', 'title', 'category', 'duration'])

        for category in config.CATEGORIES:
            # Count current valid videos for this category
            if 'category' not in df.columns:
                 cat_count = 0
            else:
                 cat_count = len(df[df['category'] == category])

            if cat_count >= config.VIDEOS_PER_CATEGORY:
                print(f"Category '{category}' is complete ({cat_count}/{config.VIDEOS_PER_CATEGORY}).")
                continue
                
            print(f"Category '{category}' needs more videos ({cat_count}/{config.VIDEOS_PER_CATEGORY}).")
            
            # Fetch new videos
            needed = config.VIDEOS_PER_CATEGORY - cat_count
            try:
                new_videos = self.collect_videos_for_category(category, needed, existing_ids)
                
                if new_videos:
                    # Append to CSV
                    new_df = pd.DataFrame(new_videos)
                    new_df.to_csv(config.METADATA_FILE, mode='a', header=False, index=False)
                    
                    # Update local tracking
                    existing_ids.update([v['video_id'] for v in new_videos])
                    # Update df locally
                    df = pd.concat([df, new_df], ignore_index=True)
                    
                    print(f"Added {len(new_videos)} new videos to metadata.")
                else:
                    print(f"No new videos found for {category}.")
            except HttpError as e:
                if e.resp.status in [403, 429]:
                    print("Stopping collection due to API limits. Resume later.")
                    break

        # Final load
        if os.path.exists(config.METADATA_FILE):
             return pd.read_csv(config.METADATA_FILE)
        return pd.DataFrame()

if __name__ == "__main__":
    collector = YouTubeCollector()
    collector.collect_all_categories()
