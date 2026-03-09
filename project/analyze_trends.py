import pandas as pd
import os

# Define the file path
file_path = 'project/youtube_shorts_tiktok_trends_2025.csv'

# Check if file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

try:
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter for platform='YouTube'
    youtube_df = df[df['platform'] == 'YouTube']
    
    # Filter for duration between 10 and 60 (inclusive)
    # Using 'duration_sec' as it matches the context of video duration better than 'trend_duration_days'
    start_duration = 10
    end_duration = 60
    
    filtered_df = youtube_df[
        (youtube_df['duration_sec'] >= start_duration) & 
        (youtube_df['duration_sec'] <= end_duration)
    ]
    
    print(f"Total records after filtering: {len(filtered_df)}")
    
    # Print unique values in 'category' and their counts
    print("\n--- Category Counts ---")
    category_counts = filtered_df['category'].value_counts()
    print(category_counts)
    
    # Print unique values in 'genre' if it exists
    if 'genre' in filtered_df.columns:
        print("\n--- Unique Genres ---")
        unique_genres = filtered_df['genre'].unique()
        for genre in unique_genres:
            print(genre)
    else:
        print("\n'genre' column not found.")

except Exception as e:
    print(f"An error occurred: {e}")
