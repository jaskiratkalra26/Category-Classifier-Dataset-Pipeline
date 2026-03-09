import pandas as pd
import os

# Load the data
csv_path = 'youtube_data.csv'
try:
    if not os.path.exists(csv_path):
        # try parent
        csv_path = '../youtube_data.csv'

    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading csv: {e}")
    exit()

# Filter duration
# Duration is likely in seconds based on 180, 930 examples.
df = df.dropna(subset=['duration'])
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
df_filtered = df[(df['duration'] >= 10) & (df['duration'] <= 60)]

def match_keywords(text, keywords):
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(k in text for k in keywords)

KEYWORD_MAP = {
    "finance": ["finance", "money", "invest", "stock", "economy", "business", "crypto", "trading"],
    "fitness": ["fitness", "workout", "gym", "exercise", "bodybuilding", "yoga", "health"],
    "cooking": ["cooking", "recipe", "food", "kitchen", "bake", "baking", "chef", "meal"],
    "beauty": ["beauty", "makeup", "skincare", "cosmetics", "fashion", "hair", "tutorial"]
}

CATEGORY_MAP = {
    "Gaming": "gaming",
    "Education": "education",
    "Science & Technology": "technology",
    "Comedy": "comedy",
    "Music": "music",
    "News & Politics": "news",
    "Sports": "sports",
    "Travel & Events": "travel"
}

def classify_row(row):
    cat = row['category']
    
    # Try direct map first
    if cat in CATEGORY_MAP:
        return CATEGORY_MAP[cat]
    
    # If not direct mapped, try keywords on Howto & Style or just generally if not mapped?
    # Let's try generally on the text fields for the 4 missing categories
    text_blob = f"{row['title']} {row['description']} {row['hashtags']}"
    
    # Priority? maybe single pass
    matched = []
    for target_cat, keywords in KEYWORD_MAP.items():
        if match_keywords(text_blob, keywords):
            matched.append(target_cat)
    
    if len(matched) == 1:
        return matched[0]
    
    # If multiple matches or no matches, return None (or 'unknown')
    return None

df_filtered['mapped_category'] = df_filtered.apply(classify_row, axis=1)

# Filter for only our target categories
TARGET_CATEGORIES = [
    "gaming", "education", "technology", "finance", "fitness",
    "cooking", "travel", "music", "comedy", "news", "sports", "beauty"
]
df_final = df_filtered[df_filtered['mapped_category'].isin(TARGET_CATEGORIES)]

# Count per category
counts = df_final['mapped_category'].value_counts()
print("Counts per category:")
print(counts)

min_count = counts.min() if not counts.empty else 0
print(f"Minimum count (limit for equal records): {min_count}")
