"""
Dataset Statistics - Word-level sentiment distribution analysis
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from bias_snapshot import preprocess_text, analyze_sentiment, download_nltk_data


def load_all_articles(json_path):
    """Load all articles from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    articles = []
    for topic_key, topic_data in data['topics'].items():
        for article_key in ['article_a', 'article_b']:
            article = topic_data[article_key].copy()
            article['topic'] = topic_data['name']
            article['article_label'] = article_key.replace('article_', '').upper()
            articles.append(article)
    
    return articles


def analyze_corpus(articles):
    """Analyze all words across corpus."""
    download_nltk_data()
    
    all_scores = []
    word_freq = Counter()
    
    for article in articles:
        tokens = preprocess_text(article['content'])
        df = analyze_sentiment(tokens)
        all_scores.extend(df['score'].tolist())
        word_freq.update(tokens)
    
    return all_scores, word_freq


def create_distribution_plot(word_scores):
    """Create word sentiment distribution visualization."""
    non_neutral = [s for s in word_scores if s != 0.0]
    neutral_count = len([s for s in word_scores if s == 0.0])
    positive = [s for s in non_neutral if s > 0]
    negative = [s for s in non_neutral if s < 0]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Overlapping histograms
    ax.hist(negative, bins=25, alpha=0.75, color='#DC143C', 
            edgecolor='black', label=f'Negative ({len(negative)})')
    ax.hist(positive, bins=25, alpha=0.75, color='#2E8B57', 
            edgecolor='black', label=f'Positive ({len(positive)})')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Sentiment Score', fontweight='bold', fontsize=12)
    ax.set_ylabel('Word Count', fontweight='bold', fontsize=12)
    ax.set_title('Word Sentiment Distribution (Excluding Neutral)', 
                 fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    # Stats box
    stats = f'Total: {len(word_scores):,}\n'
    stats += f'Neutral: {neutral_count:,} ({neutral_count/len(word_scores)*100:.1f}%)\n'
    stats += f'Non-Neutral: {len(non_neutral):,}\n'
    stats += f'Mean: {np.mean(non_neutral):.3f}'
    
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=10, family='monospace')
    
    plt.tight_layout()
    return fig


def create_breakdown_plot(word_scores):
    """Create sentiment category breakdown."""
    # Granular categorization
    strong_positive = sum(1 for s in word_scores if s > 0.3)
    positive = sum(1 for s in word_scores if 0.05 < s <= 0.3)
    weak_positive = sum(1 for s in word_scores if 0 < s <= 0.05)
    neutral = sum(1 for s in word_scores if s == 0)
    weak_negative = sum(1 for s in word_scores if -0.05 <= s < 0)
    negative = sum(1 for s in word_scores if -0.3 <= s < -0.05)
    strong_negative = sum(1 for s in word_scores if s < -0.3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Pie chart - 3 categories
    positive_total = strong_positive + positive + weak_positive
    negative_total = strong_negative + negative + weak_negative
    
    sizes = [positive_total, negative_total, neutral]
    labels = [f'Positive\n{positive_total:,}', f'Negative\n{negative_total:,}', f'Neutral\n{neutral:,}']
    colors = ['#2E8B57', '#DC143C', '#808080']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('Overall Sentiment Distribution', fontweight='bold', fontsize=14)
    
    # Right: Granular bar chart - 6 categories
    categories = ['Strong\nPositive\n(>0.3)', 'Positive\n(0.05-0.3)', 
                  'Weak Pos\n(0-0.05)', 'Weak Neg\n(0 to -0.05)',
                  'Negative\n(-0.05 to -0.3)', 'Strong\nNegative\n(<-0.3)']
    counts = [strong_positive, positive, weak_positive, 
              weak_negative, negative, strong_negative]
    colors_bar = ['#006400', '#2E8B57', '#90EE90', 
                  '#FFB6C1', '#DC143C', '#8B0000']
    
    bars = ax2.bar(range(len(categories)), counts, color=colors_bar, alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_ylabel('Word Count', fontweight='bold', fontsize=12)
    ax2.set_title('Granular Sentiment Breakdown', fontweight='bold', fontsize=14)
    ax2.grid(alpha=0.3, axis='y')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        if count > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    return fig


def print_statistics(word_scores, word_freq, articles):
    """Print corpus statistics."""
    print("\n" + "="*70)
    print("CORPUS STATISTICS")
    print("="*70)
    
    positive = sum(1 for s in word_scores if s > 0.05)
    negative = sum(1 for s in word_scores if s < -0.05)
    neutral = sum(1 for s in word_scores if -0.05 <= s <= 0.05)
    
    print(f"\nDataset: {len(articles)} articles, {len(word_scores):,} words, {len(word_freq):,} unique")
    print(f"\nSentiment Distribution:")
    print(f"  Positive: {positive:,} ({positive/len(word_scores)*100:.1f}%)")
    print(f"  Negative: {negative:,} ({negative/len(word_scores)*100:.1f}%)")
    print(f"  Neutral: {neutral:,} ({neutral/len(word_scores)*100:.1f}%)")
    print(f"\nStats: Mean={np.mean(word_scores):.3f}, StdDev={np.std(word_scores):.3f}")
    print(f"\nTop Words: {', '.join([f'{w}({c})' for w, c in word_freq.most_common(8)])}")
    print("="*70 + "\n")


def create_article_table(articles):
    """Create article comparison table."""
    download_nltk_data()
    
    rows = []
    for article in articles:
        tokens = preprocess_text(article['content'])
        df = analyze_sentiment(tokens)
        
        rows.append({
            'Topic': article['topic'],
            'Article': article['article_label'],
            'Stance': article.get('stance', 'N/A'),
            'Avg': f"{df['score'].mean():.3f}",
            'Pos': len(df[df['score'] > 0.05]),
            'Neg': len(df[df['score'] < -0.05])
        })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv('article_stats.csv', index=False)
    print(f"\n💾 Saved: article_stats.csv\n")


def main():
    """Main function."""
    import sys
    
    json_file = sys.argv[1] if len(sys.argv) > 1 else "test_articles.json"
    
    print(f"📁 Loading: {json_file}")
    articles = load_all_articles(json_file)
    
    print(f"🔍 Analyzing corpus...")
    word_scores, word_freq = analyze_corpus(articles)
    
    print_statistics(word_scores, word_freq, articles)
    create_article_table(articles)
    
    print("📈 Creating visualizations...")
    
    fig1 = create_distribution_plot(word_scores)
    fig1.savefig('word_distribution.png', dpi=300, bbox_inches='tight')
    print("   💾 Saved: word_distribution.png")
    
    fig2 = create_breakdown_plot(word_scores)
    fig2.savefig('sentiment_breakdown.png', dpi=300, bbox_inches='tight')
    print("   💾 Saved: sentiment_breakdown.png")
    
    plt.show()
    print("\n✅ Complete!\n")


if __name__ == "__main__":
    main()
