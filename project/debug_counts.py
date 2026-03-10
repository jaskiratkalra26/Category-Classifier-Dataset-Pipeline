import pandas as pd
import os

csv_path = 'project/youtube_shorts_tiktok_trends_2025.csv'
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    exit()

df = pd.read_csv(csv_path)

CATEGORY_MAP = {
    "Gaming": "Gaming",
    "Education": "Education",
    "Tech": "Technology",
    "Science": "Technology",
    "Finance": "Finance",
    "Fitness": "Fitness",
    "Food": "Cooking",
    "Travel": "Travel",
    "Music": "Music",
    "Comedy": "Comedy",
    "News": "News",
    "Sports": "Sports",
    "Beauty": "Beauty",
    "Fashion": "Beauty",
}
TARGET_CATEGORIES = [
    "Gaming", "Education", "Technology", "Finance", "Fitness",
    "Cooking", "Travel", "Music", "Comedy", "News", "Sports", "Beauty"
]

print("Filtering for YouTube platform...")
df = df[df['platform'].str.lower() == 'youtube']
print(f"Count after platform filter: {len(df)}")

print("Filtering duration 10-60s...")
df = df[(df['duration_sec'] >= 10) & (df['duration_sec'] <= 60)]
print(f"Count after duration filter: {len(df)}")

print("Mapping categories...")
df['mapped_category'] = df['category'].map(CATEGORY_MAP)
df = df.dropna(subset=['mapped_category'])
df = df[df['mapped_category'].isin(TARGET_CATEGORIES)]
print(f"Count after category map and filter: {len(df)}")

counts = df['mapped_category'].value_counts()
print(counts)

target_count = 20
print(f"\nChecking if any category has < {target_count}:")
for cat, count in counts.items():
    if count < target_count:
        print(f"  {cat}: {count} (Sample will FAIL)")
    else:
        print(f"  {cat}: {count} (OK)")
