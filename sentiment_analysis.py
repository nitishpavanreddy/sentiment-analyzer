# ============================================================
#  AI-POWERED SENTIMENT ANALYZER - Full Project
#  Portfolio Project | NLP + Text Analytics
# ============================================================
#
#  HOW TO RUN:
#  C:/Users/nitis/AppData/Local/Python/pythoncore-3.14-64/python.exe sentiment_analysis.py
#
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

print("=" * 55)
print("   AI-POWERED SENTIMENT ANALYZER")
print("=" * 55)

# ============================================================
#  STEP 1: CREATE REALISTIC REVIEW DATASET
# ============================================================

reviews_data = [
    # Positive reviews
    ("This product is absolutely amazing! Best purchase I made this year.", 5, "Electronics"),
    ("Excellent quality, fast shipping. Highly recommend to everyone!", 5, "Electronics"),
    ("Works perfectly, exactly as described. Very happy with it.", 5, "Electronics"),
    ("Outstanding product! Exceeded my expectations completely.", 5, "Electronics"),
    ("Great value for money. Will definitely buy again from this seller.", 4, "Electronics"),
    ("Really good product, easy to use and great battery life.", 4, "Electronics"),
    ("Love this product! Works great and looks beautiful.", 5, "Electronics"),
    ("Fantastic quality. Arrived quickly and packaged very well.", 4, "Electronics"),
    ("Perfect for my needs. Solid build quality and great performance.", 5, "Electronics"),
    ("Very satisfied with this purchase. Highly recommend!", 4, "Electronics"),
    ("Impressive product. Does exactly what it promises.", 5, "Electronics"),
    ("Good product overall. Setup was easy and it works well.", 4, "Electronics"),
    # Negative reviews
    ("Terrible product. Broke after just two days of use.", 1, "Electronics"),
    ("Very disappointed. Does not work as advertised at all.", 1, "Electronics"),
    ("Worst purchase ever. Complete waste of money.", 1, "Electronics"),
    ("Poor quality. Stopped working after one week.", 1, "Electronics"),
    ("Not worth the price. Very cheap materials used.", 2, "Electronics"),
    ("Disappointed with the quality. Expected much better.", 2, "Electronics"),
    ("Product arrived damaged. Customer service was unhelpful.", 1, "Electronics"),
    ("Does not match the description. Very misleading product.", 2, "Electronics"),
    ("Bad experience overall. Would not recommend to anyone.", 1, "Electronics"),
    ("Overpriced for the quality you get. Very dissatisfied.", 2, "Electronics"),
    # Neutral reviews
    ("Product is okay. Nothing special but does the job.", 3, "Electronics"),
    ("Average product. Some good features but also some issues.", 3, "Electronics"),
    ("Decent quality for the price. Not amazing but acceptable.", 3, "Electronics"),
    ("Works as expected. No major complaints but nothing impressive.", 3, "Electronics"),
    ("Mediocre product. Has potential but needs improvement.", 3, "Electronics"),
    # Furniture reviews
    ("Beautiful furniture! Transforms the entire room look.", 5, "Furniture"),
    ("Easy to assemble and looks great in my living room.", 4, "Furniture"),
    ("Sturdy and well built. Great addition to my home.", 5, "Furniture"),
    ("Chair is comfortable and looks exactly like the picture.", 4, "Furniture"),
    ("Good desk for the price. Solid construction overall.", 4, "Furniture"),
    ("Wobbly and cheaply made. Very disappointed with quality.", 1, "Furniture"),
    ("Instructions unclear. Took forever to put together.", 2, "Furniture"),
    ("Decent furniture but nothing exceptional about it.", 3, "Furniture"),
    ("Missing parts in the box. Very frustrating experience.", 1, "Furniture"),
    ("Looks nice but not as sturdy as I expected.", 3, "Furniture"),
    # Office Supplies reviews
    ("Perfect pens! Smooth writing and great ink quality.", 5, "Office Supplies"),
    ("Good paper quality. Printer loves it, no jams at all.", 4, "Office Supplies"),
    ("Amazing binder. Holds everything perfectly and durable.", 5, "Office Supplies"),
    ("Ink runs out too fast. Not worth the price at all.", 2, "Office Supplies"),
    ("Average quality paper. Does the job but nothing more.", 3, "Office Supplies"),
    ("Great office supplies. Fast delivery and good packaging.", 4, "Office Supplies"),
    ("Poor quality stapler. Jams constantly and very frustrating.", 1, "Office Supplies"),
    ("Decent value for money. Works fine for everyday use.", 3, "Office Supplies"),
    ("Excellent pens for the price. Very smooth and reliable.", 5, "Office Supplies"),
    ("Disappointed with quality. Expected better from this brand.", 2, "Office Supplies"),
]

df = pd.DataFrame(reviews_data, columns=['Review', 'Rating', 'Category'])
print(f"\nDataset created: {len(df)} reviews across {df['Category'].nunique()} categories")

# ============================================================
#  STEP 2: SENTIMENT ANALYSIS
# ============================================================

print("\n--- Running Sentiment Analysis ---")

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # VADER scores
    scores = sia.polarity_scores(text)
    compound = scores['compound']

    # TextBlob scores
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Combined score
    combined = (compound + polarity) / 2

    # Classify sentiment
    if combined >= 0.05:
        sentiment = 'Positive'
    elif combined <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return pd.Series({
        'VADER_Score': round(compound, 3),
        'TextBlob_Score': round(polarity, 3),
        'Combined_Score': round(combined, 3),
        'Sentiment': sentiment
    })

# Apply sentiment analysis to all reviews
sentiment_results = df['Review'].apply(analyze_sentiment)
df = pd.concat([df, sentiment_results], axis=1)

# ============================================================
#  STEP 3: ANALYSIS RESULTS
# ============================================================

print("\n--- Sentiment Distribution ---")
sent_counts = df['Sentiment'].value_counts()
total = len(df)
for sent, count in sent_counts.items():
    print(f"{sent:10} : {count:3} reviews ({count/total*100:.1f}%)")

print("\n--- Sentiment by Category ---")
cat_sent = df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
print(cat_sent.to_string())

print("\n--- Average Scores ---")
print(f"Average VADER score    : {df['VADER_Score'].mean():.3f}")
print(f"Average TextBlob score : {df['TextBlob_Score'].mean():.3f}")
print(f"Average Combined score : {df['Combined_Score'].mean():.3f}")

print("\n--- Top 3 Most Positive Reviews ---")
top_pos = df.nlargest(3, 'Combined_Score')[['Review', 'Combined_Score']]
for _, row in top_pos.iterrows():
    print(f"Score {row['Combined_Score']:.2f}: {row['Review'][:60]}...")

print("\n--- Top 3 Most Negative Reviews ---")
top_neg = df.nsmallest(3, 'Combined_Score')[['Review', 'Combined_Score']]
for _, row in top_neg.iterrows():
    print(f"Score {row['Combined_Score']:.2f}: {row['Review'][:60]}...")

# ============================================================
#  STEP 4: VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('AI-Powered Review Sentiment Dashboard',
             fontsize=16, fontweight='bold')
plt.subplots_adjust(hspace=0.4, wspace=0.35)

colors = {'Positive': '#10B981', 'Neutral': '#F59E0B', 'Negative': '#EF4444'}

# Chart 1: Sentiment Distribution Donut
ax = axes[0, 0]
sent_data = df['Sentiment'].value_counts()
pie_colors = [colors[s] for s in sent_data.index]
wedges, texts, autotexts = ax.pie(
    sent_data.values, labels=sent_data.index,
    autopct='%1.1f%%', colors=pie_colors,
    wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax.set_title('Overall Sentiment Distribution', fontweight='bold')

# Chart 2: Sentiment by Category
ax = axes[0, 1]
cat_sent_plot = df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
bar_colors = [colors.get(c, '#888') for c in cat_sent_plot.columns]
cat_sent_plot.plot(kind='bar', ax=ax, color=bar_colors,
                   edgecolor='white', linewidth=0.5)
ax.set_title('Sentiment by Category', fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Number of Reviews')
ax.legend(title='Sentiment', fontsize=9)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

# Chart 3: Sentiment Score Distribution
ax = axes[1, 0]
pos_scores = df[df['Sentiment']=='Positive']['Combined_Score']
neg_scores = df[df['Sentiment']=='Negative']['Combined_Score']
neu_scores = df[df['Sentiment']=='Neutral']['Combined_Score']
ax.hist(pos_scores, bins=10, alpha=0.7, color='#10B981', label='Positive')
ax.hist(neu_scores, bins=5,  alpha=0.7, color='#F59E0B', label='Neutral')
ax.hist(neg_scores, bins=10, alpha=0.7, color='#EF4444', label='Negative')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
ax.set_title('Sentiment Score Distribution', fontweight='bold')
ax.set_xlabel('Combined Sentiment Score')
ax.set_ylabel('Number of Reviews')
ax.legend()

# Chart 4: Word Cloud of all reviews
ax = axes[1, 1]
stop_words = set(stopwords.words('english'))
all_text = ' '.join(df['Review'].tolist())
all_text = re.sub(r'[^a-zA-Z\s]', '', all_text.lower())

wordcloud = WordCloud(
    width=600, height=300,
    background_color='white',
    stopwords=stop_words,
    colormap='RdYlGn',
    max_words=50
).generate(all_text)

ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Most Frequent Words in Reviews', fontweight='bold')

plt.tight_layout()
plt.savefig('sentiment_dashboard.png', dpi=150, bbox_inches='tight')
print("\nDashboard saved as: sentiment_dashboard.png")

# ============================================================
#  STEP 5: BUSINESS INSIGHTS
# ============================================================

print("\n" + "="*55)
print("  BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*55)

pos_pct = (df['Sentiment']=='Positive').mean() * 100
neg_pct = (df['Sentiment']=='Negative').mean() * 100
neu_pct = (df['Sentiment']=='Neutral').mean() * 100

print(f"\n1. OVERALL SENTIMENT:")
print(f"   Positive : {pos_pct:.1f}% of customers are happy")
print(f"   Negative : {neg_pct:.1f}% of customers are unhappy")
print(f"   Neutral  : {neu_pct:.1f}% of customers are neutral")

best_cat = df.groupby('Category')['Combined_Score'].mean().idxmax()
worst_cat = df.groupby('Category')['Combined_Score'].mean().idxmin()
print(f"\n2. BEST CATEGORY    : {best_cat} — highest customer satisfaction")
print(f"   PROBLEM CATEGORY : {worst_cat} — needs immediate attention")

print(f"\n3. KEY COMPLAINT THEMES (from negative reviews):")
print("   → Product quality not matching description")
print("   → Items breaking after short use")
print("   → Damaged or missing parts on arrival")
print("   → Poor value for money")

print(f"\n4. RECOMMENDATIONS:")
print("   → Prioritize quality control for negative categories")
print("   → Respond to all 1-star reviews within 24 hours")
print("   → Use positive review keywords in product descriptions")
print("   → Investigate shipping damage complaints")
print("="*55)
