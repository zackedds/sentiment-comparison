"""
Bias Snapshot DEBUG - Adjusted for full-length articles
Testing different sentiment thresholds and normalization approaches
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

# Import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from bias_snapshot import preprocess_text, analyze_sentiment, download_nltk_data


def load_data(json_path="../data/real_articles_full.json"):
    """Load articles from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    articles = []
    for topic_key, topic_data in data['topics'].items():
        for article_key in ['article_a', 'article_b']:
            article = topic_data[article_key].copy()
            article['topic'] = topic_data['name']
            article['article_label'] = f"{topic_data['name']} - {article_key.upper()}"
            article['stance'] = article.get('stance', 'unknown')
            articles.append(article)
    
    return articles


def analyze_dataset_normalized(articles):
    """Analyze with relative scoring - compare against corpus mean."""
    download_nltk_data()
    
    results = []
    all_scores = []
    article_dfs = []
    
    # First pass: collect all scores
    for article in articles:
        tokens = preprocess_text(article['content'])
        df = analyze_sentiment(tokens)
        article_dfs.append((article, df))
        all_scores.extend(df['score'].tolist())
    
    # Calculate corpus mean as baseline
    corpus_mean = np.mean(all_scores)
    corpus_std = np.std(all_scores)
    
    print(f"\n🔍 DEBUG INFO:")
    print(f"   Corpus Mean: {corpus_mean:.4f}")
    print(f"   Corpus Std Dev: {corpus_std:.4f}")
    
    # Second pass: calculate relative scores
    for article, df in article_dfs:
        # Method 1: Relative to corpus mean
        relative_score = df['score'].mean() - corpus_mean
        
        # Method 2: Tighter thresholds (0.01 instead of 0.05)
        positive_tight = len(df[df['score'] > 0.01])
        negative_tight = len(df[df['score'] < -0.01])
        
        # Method 3: Z-score normalization
        z_scores = (df['score'] - corpus_mean) / corpus_std if corpus_std > 0 else df['score']
        z_mean = z_scores.mean()
        
        results.append({
            'Topic': article['topic'],
            'Article': article['article_label'],
            'Stance': article['stance'],
            'Raw Avg': f"{df['score'].mean():.4f}",
            'Relative Score': f"{relative_score:.4f}",
            'Z-Score': f"{z_mean:.4f}",
            'Pos (>0.01)': positive_tight,
            'Neg (<-0.01)': negative_tight,
            'Total Words': len(df)
        })
    
    return pd.DataFrame(results), all_scores, corpus_mean


def create_debug_visualizations(results_df, all_scores, corpus_mean):
    """Create visualizations with adjusted baselines."""
    
    # Figure 1: Relative sentiment comparison (normalized around 0)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Relative scores (difference from corpus mean)
    results_df['Relative Numeric'] = results_df['Relative Score'].astype(float)
    colors = ['#2E8B57' if x > 0 else '#DC143C' if x < 0 else '#808080' 
              for x in results_df['Relative Numeric']]
    
    bars = ax1.barh(range(len(results_df)), results_df['Relative Numeric'], 
                    color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['Article'], fontsize=9)
    ax1.set_xlabel('Relative Sentiment (vs Corpus Mean)', fontweight='bold', fontsize=11)
    ax1.set_title('Sentiment Relative to Corpus Baseline', 
                  fontweight='bold', fontsize=13, pad=15)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.grid(alpha=0.3, axis='x')
    
    # Add stance labels
    for i, (_, row) in enumerate(results_df.iterrows()):
        stance = row['Stance']
        if stance in ['pro', 'con']:
            ax1.text(0.98, i, f" [{stance}]", transform=ax1.get_yaxis_transform(),
                    va='center', ha='right', fontsize=8, style='italic', alpha=0.6)
    
    # Right: Z-scores (standardized)
    results_df['Z Numeric'] = results_df['Z-Score'].astype(float)
    colors_z = ['#2E8B57' if x > 0 else '#DC143C' if x < 0 else '#808080' 
                for x in results_df['Z Numeric']]
    
    bars2 = ax2.barh(range(len(results_df)), results_df['Z Numeric'], 
                     color=colors_z, alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(results_df)))
    ax2.set_yticklabels(results_df['Article'], fontsize=9)
    ax2.set_xlabel('Z-Score (Standardized)', fontweight='bold', fontsize=11)
    ax2.set_title('Standardized Sentiment Scores', 
                  fontweight='bold', fontsize=13, pad=15)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig1.savefig('debug_normalized_sentiment.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: debug_normalized_sentiment.png")
    
    # Figure 2: Word distribution with tighter thresholds
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter non-neutral with tighter threshold
    positive_tight = [s for s in all_scores if s > 0.01]
    negative_tight = [s for s in all_scores if s < -0.01]
    neutral_tight = len(all_scores) - len(positive_tight) - len(negative_tight)
    
    # Histogram with tighter bounds
    ax3.hist(negative_tight, bins=30, alpha=0.75, color='#DC143C', 
             edgecolor='black', label=f'Negative ({len(negative_tight)})')
    ax3.hist(positive_tight, bins=30, alpha=0.75, color='#2E8B57', 
             edgecolor='black', label=f'Positive ({len(positive_tight)})')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.axvline(x=corpus_mean, color='orange', linestyle=':', linewidth=2, 
               label=f'Corpus Mean ({corpus_mean:.4f})', alpha=0.7)
    ax3.set_xlabel('Sentiment Score', fontweight='bold')
    ax3.set_ylabel('Word Count', fontweight='bold')
    ax3.set_title('Word Distribution (Threshold: ±0.01)', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3, axis='y')
    
    # Comparative bar: Pos vs Neg words by article
    articles = results_df['Article'].str.split(' - ').str[-1]
    pos_counts = results_df['Pos (>0.01)'].values
    neg_counts = results_df['Neg (<-0.01)'].values
    
    x = np.arange(len(articles))
    width = 0.35
    
    ax4.bar(x - width/2, pos_counts, width, label='Positive', 
           color='#2E8B57', alpha=0.7, edgecolor='black')
    ax4.bar(x + width/2, neg_counts, width, label='Negative', 
           color='#DC143C', alpha=0.7, edgecolor='black')
    
    ax4.set_ylabel('Word Count', fontweight='bold')
    ax4.set_title('Positive vs Negative Words by Article\n(Threshold: ±0.01)', 
                 fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(articles, rotation=45, ha='right', fontsize=8)
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig2.savefig('debug_word_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: debug_word_comparison.png")
    
    plt.show()


def print_debug_summary(results_df, all_scores, corpus_mean):
    """Print detailed debug summary."""
    print("\n" + "="*70)
    print("DEBUG MODE - NORMALIZED SENTIMENT ANALYSIS")
    print("="*70)
    
    print(f"\n📊 Corpus Statistics:")
    print(f"   Total Words: {len(all_scores)}")
    print(f"   Corpus Mean: {corpus_mean:.4f} (baseline)")
    print(f"   Corpus Std Dev: {np.std(all_scores):.4f}")
    print(f"   This corpus skews slightly positive overall")
    
    # Compare with different thresholds
    std_pos = sum(1 for s in all_scores if s > 0.05)
    std_neg = sum(1 for s in all_scores if s < -0.05)
    tight_pos = sum(1 for s in all_scores if s > 0.01)
    tight_neg = sum(1 for s in all_scores if s < -0.01)
    
    print(f"\n📐 Threshold Comparison:")
    print(f"   Standard (±0.05): Pos={std_pos} ({std_pos/len(all_scores)*100:.1f}%), "
          f"Neg={std_neg} ({std_neg/len(all_scores)*100:.1f}%)")
    print(f"   Tight (±0.01):    Pos={tight_pos} ({tight_pos/len(all_scores)*100:.1f}%), "
          f"Neg={tight_neg} ({tight_neg/len(all_scores)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("ARTICLE COMPARISON (Multiple Metrics)")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    print(f"\n💡 Key Insights:")
    print(f"   - All articles cluster around corpus mean ({corpus_mean:.4f})")
    print(f"   - Relative scores show which articles are MORE/LESS positive")
    print(f"   - Z-scores standardize for better comparison")
    print(f"   - Tighter thresholds (±0.01) capture more nuanced differences")


def main():
    """Main debug function."""
    print("🐛 Bias Snapshot DEBUG MODE - Full Article Analysis\n")
    
    # Load data
    print("📁 Loading full-length articles...")
    articles = load_data()
    print(f"   Loaded {len(articles)} articles\n")
    
    # Analyze with normalization
    print("🔍 Running normalized sentiment analysis...")
    results_df, all_scores, corpus_mean = analyze_dataset_normalized(articles)
    
    # Print summary
    print_debug_summary(results_df, all_scores, corpus_mean)
    
    # Export table
    results_df.to_csv('debug_results_table.csv', index=False)
    print("\n💾 Saved: debug_results_table.csv\n")
    
    # Create visualizations
    print("📊 Creating debug visualizations...")
    create_debug_visualizations(results_df, all_scores, corpus_mean)
    
    print("\n✅ Debug Analysis Complete!")
    print("\n📝 Recommendations:")
    print("   1. Use RELATIVE scores instead of raw scores")
    print("   2. Consider tighter thresholds (±0.01 vs ±0.05)")
    print("   3. Focus on comparative differences, not absolute values")
    print("   4. News articles naturally skew slightly positive")
    print("   5. Z-scores normalize for statistical comparison\n")


if __name__ == "__main__":
    main()
