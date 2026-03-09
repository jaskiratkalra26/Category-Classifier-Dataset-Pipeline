import pandas as pd
import re

# Load the dataset
file_path = 'project/youtube_shorts_tiktok_trends_2025.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at {file_path}")
    exit()

print("--- 1. Unique Platform Values ---")
print(df['platform'].unique())
print("\n")

print("--- 2. Checking for YouTube ID-like (11 chars) or URL columns ---")
youtube_id_pattern = re.compile(r'^[a-zA-Z0-9_-]{11}$')
url_pattern = re.compile(r'^http')

for col in df.columns:
    # Check first non-null value if possible
    sample = df[col].dropna().astype(str)
    if not sample.empty:
        first_val = sample.iloc[0]
        match_id = youtube_id_pattern.match(first_val)
        match_url = url_pattern.match(first_val)
        
        # Check a few more to be sure it's not a coincidence (e.g. "Gaming" has 6 chars, but maybe some other word has 11)
        # But specifically looking for the ID pattern.
        # Let's count how many match the pattern in a sample of 10
        matches = sample.head(10).apply(lambda x: bool(youtube_id_pattern.match(x))).sum()
        is_url = sample.head(10).apply(lambda x: bool(url_pattern.match(x))).sum()

        if matches > 0:
            print(f"Column '{col}' has values that look like YouTube IDs (example: {first_val}) - Matches in first 10: {matches}")
        if is_url > 0:
            print(f"Column '{col}' has values that look like URLs (example: {first_val})")

print("\n")

print("--- 3. First 3 rows where platform is 'YouTube Shorts' or 'YouTube' ---")
# Filter where platform contains 'YouTube' to be safe, or exact match
yt_df = df[df['platform'].isin(['YouTube Shorts', 'YouTube'])]
if not yt_df.empty:
    print(yt_df.head(3).to_markdown(index=False))
else:
    print("No YouTube rows found.")

print("\n")

print("--- 4. Columns available for link construction ---")
print(df.columns.tolist())
