import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
try:
    df = pd.read_csv('Global YouTube Statistics.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'Global YouTube Statistics.csv' not found. Please ensure the file is in the same directory.")
    exit()

# 2. Data Cleaning
# Fill missing categorical values
df['Country'] = df['Country'].fillna('Unknown')
df['category'] = df['category'].fillna('Unknown')

# Drop rows where creation year is missing for temporal analysis
df = df.dropna(subset=['created_year'])
df['created_year'] = df['created_year'].astype(int)

# Set global style
sns.set_theme(style="whitegrid")

# --- Visualization 1: Top 10 YouTubers by Subscribers ---
plt.figure(figsize=(12, 6))
top_10_subs = df.nlargest(10, 'subscribers')
sns.barplot(data=top_10_subs, x='subscribers', y='Youtuber', palette='viridis')
plt.title('Top 10 YouTubers by Subscribers', fontsize=16)
plt.xlabel('Subscribers (Billions)', fontsize=12)
plt.ylabel('YouTuber', fontsize=12)
plt.tight_layout()
plt.savefig('eda_top_10_subscribers.png')
print("Saved: eda_top_10_subscribers.png")

# --- Visualization 2: Category Distribution ---
plt.figure(figsize=(12, 8))
category_counts = df['category'].value_counts()
sns.barplot(x=category_counts.values, y=category_counts.index, palette='magma')
plt.title('Distribution of YouTube Categories', fontsize=16)
plt.xlabel('Number of Channels', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.tight_layout()
plt.savefig('eda_category_distribution.png')
print("Saved: eda_category_distribution.png")

# --- Visualization 3: Subscribers vs Views (Correlation) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='video views', y='subscribers', hue='category', alpha=0.6, legend=False)
plt.title('Correlation: Subscribers vs Video Views', fontsize=16)
plt.xlabel('Video Views (Log Scale)', fontsize=12)
plt.ylabel('Subscribers (Log Scale)', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('eda_subs_vs_views.png')
print("Saved: eda_subs_vs_views.png")

# --- Visualization 4: Temporal Trend (Channels Created per Year) ---
plt.figure(figsize=(12, 6))
yearly_counts = df['created_year'].value_counts().sort_index()
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', linewidth=2.5, color='teal')
plt.title('Growth of YouTube: Channels Created Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('New Channels Created', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('eda_temporal_trend.png')
print("Saved: eda_temporal_trend.png")

print("\nEDA Completed. Images saved.")