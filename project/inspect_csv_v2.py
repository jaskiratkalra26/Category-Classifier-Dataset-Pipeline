import pandas as pd
import re

file_path = 'project/youtube_shorts_tiktok_trends_2025.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at {file_path}")
    exit()

print("--- 1. Unique Platform Values ---")
print(df['platform'].unique())
print()

columns_with_ids = []
columns_with_urls = []

youtube_id_pattern = re.compile(r'^[a-zA-Z0-9_-]{11}$')
url_pattern = re.compile(r'^http')

# Only check first 50 rows to be fast but thorough enough
sample_df = df.head(50)

print("--- 2. Checking for YouTube ID-like (11 chars) or URL columns ---")

for col in df.columns:
    col_matches = 0
    col_urls = 0
    # Process column as string
    vals = sample_df[col].dropna().astype(str)
    
    for val in vals:
        if youtube_id_pattern.match(val):
            # Print the first match found to confirm valid ID look
            if col_matches == 0:
                print(f"Potential ID in '{col}': {val}")
            col_matches += 1
        if url_pattern.match(val):
            if col_urls == 0:
                print(f"Potential URL in '{col}': {val}")
            col_urls += 1
            
    if col_matches > 0:
        columns_with_ids.append(col)
    if col_urls > 0:
        columns_with_urls.append(col)

print(f"Columns with probable IDs: {columns_with_ids}")
print(f"Columns with probable URLs: {columns_with_urls}")
print()

print("--- 3. First 3 rows where platform is 'YouTube Shorts' or 'YouTube' ---")
# Filter where platform contains 'YouTube'
yt_df = df[df['platform'].isin(['YouTube Shorts', 'YouTube'])]

if not yt_df.empty:
    # Print dictionary of first row to see all fields clearly
    print("First row details (as dict):")
    row_dict = yt_df.iloc[0].to_dict()
    for k, v in row_dict.items():
        print(f"  {k}: {v}")
        
    print("\nFirst 3 rows (platform, title, row_id):")
    print(yt_df[['platform', 'title', 'row_id']].head(3).to_string())
else:
    print("No YouTube rows found.")
print()

print("--- 4. Columns available ---")
print(df.columns.tolist())
