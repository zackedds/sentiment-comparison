"""
Bias Snapshot - Article Comparison Tool
Compare sentiment between two articles side-by-side
"""

import json
import matplotlib.pyplot as plt
from bias_snapshot import preprocess_text, analyze_sentiment, download_nltk_data


def load_articles(json_path="test_articles.json", topic=None):
    """Load articles from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Use first topic if none specified
    if topic is None:
        topic = list(data['topics'].keys())[0]
    
    if topic not in data['topics']:
        available = ', '.join(data['topics'].keys())
        raise ValueError(f"Topic '{topic}' not found. Available: {available}")
    
    return data['topics'][topic]


def compare_articles(text_a, text_b, title_a="Article A", title_b="Article B"):
    """Compare sentiment between two articles."""
    download_nltk_data()
    
    # Analyze both articles
    tokens_a = preprocess_text(text_a)
    tokens_b = preprocess_text(text_b)
    
    df_a = analyze_sentiment(tokens_a)
    df_b = analyze_sentiment(tokens_b)
    
    # Calculate metrics
    metrics_a = {
        'avg_score': df_a['score'].mean(),
        'positive': len(df_a[df_a['score'] > 0.05]),
        'negative': len(df_a[df_a['score'] < -0.05]),
        'total': len(df_a)
    }
    
    metrics_b = {
        'avg_score': df_b['score'].mean(),
        'positive': len(df_b[df_b['score'] > 0.05]),
        'negative': len(df_b[df_b['score'] < -0.05]),
        'total': len(df_b)
    }
    
    # Get top words
    top_a_pos = df_a[df_a['score'] > 0.05].nlargest(8, 'score')
    top_a_neg = df_a[df_a['score'] < -0.05].nsmallest(8, 'score')
    
    top_b_pos = df_b[df_b['score'] > 0.05].nlargest(8, 'score')
    top_b_neg = df_b[df_b['score'] < -0.05].nsmallest(8, 'score')
    
    # Create visualization
    create_comparison_chart(
        top_a_pos, top_a_neg, top_b_pos, top_b_neg,
        metrics_a, metrics_b, title_a, title_b
    )
    
    return metrics_a, metrics_b


def truncate_title(title, max_length=60):
    """Truncate title if too long."""
    if len(title) <= max_length:
        return title
    return title[:max_length] + "..."


def create_comparison_chart(top_a_pos, top_a_neg, top_b_pos, top_b_neg,
                           metrics_a, metrics_b, title_a, title_b):
    """Create side-by-side comparison visualization with dashboard."""
    
    # Truncate titles for display
    title_a_short = truncate_title(title_a, 50)
    title_b_short = truncate_title(title_b, 50)
    
    fig = plt.figure(figsize=(14, 8))
    
    # Create grid: 2 rows, 2 columns
    # Top row: side-by-side bar charts
    # Bottom row: dashboard metrics
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Article A chart (left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_article_bars(ax1, top_a_pos, top_a_neg, title_a_short, metrics_a)
    
    # Article B chart (right)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_article_bars(ax2, top_b_pos, top_b_neg, title_b_short, metrics_b)
    
    # Dashboard (bottom, spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])
    plot_dashboard(ax3, metrics_a, metrics_b, title_a_short, title_b_short)
    
    plt.show()


def plot_article_bars(ax, top_pos, top_neg, title, metrics):
    """Plot bar chart for a single article."""
    # Combine words
    words = list(top_neg['word']) + list(top_pos['word'])
    scores = list(top_neg['score']) + list(top_pos['score'])
    colors = ['#DC143C'] * len(top_neg) + ['#2E8B57'] * len(top_pos)
    
    if not words:
        ax.text(0.5, 0.5, 'No significant words', ha='center', va='center')
        ax.set_title(title, fontweight='bold', fontsize=12)
        return
    
    # Create bars
    ax.barh(range(len(words)), scores, color=colors, alpha=0.7)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9)
    ax.set_xlabel('Sentiment Score', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.grid(axis='x', alpha=0.2)


def plot_dashboard(ax, metrics_a, metrics_b, title_a, title_b):
    """Plot minimal dashboard with key metrics."""
    ax.axis('off')
    
    # Dashboard title
    ax.text(0.5, 0.9, 'Comparison Dashboard', ha='center', va='top',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    # Article A metrics (left side)
    y_pos = 0.6
    ax.text(0.25, y_pos, f"Avg Score: {metrics_a['avg_score']:.3f}",
            ha='center', fontsize=10, transform=ax.transAxes)
    
    ax.text(0.25, y_pos - 0.25, 
            f"Positive: {metrics_a['positive']}  |  Negative: {metrics_a['negative']}",
            ha='center', fontsize=9, transform=ax.transAxes)
    
    # Article B metrics (right side)
    ax.text(0.75, y_pos, f"Avg Score: {metrics_b['avg_score']:.3f}",
            ha='center', fontsize=10, transform=ax.transAxes)
    
    ax.text(0.75, y_pos - 0.25,
            f"Positive: {metrics_b['positive']}  |  Negative: {metrics_b['negative']}",
            ha='center', fontsize=9, transform=ax.transAxes)
    
    # Comparison indicator (center)
    diff = metrics_a['avg_score'] - metrics_b['avg_score']
    color = '#2E8B57' if diff > 0 else '#DC143C' if diff < 0 else 'gray'
    
    ax.text(0.5, y_pos, f"Difference: {diff:+.3f}",
            ha='center', fontsize=10, fontweight='bold',
            color=color, transform=ax.transAxes)
    
    # Divider line
    ax.plot([0.5, 0.5], [0, 0.8], 'k-', alpha=0.2, transform=ax.transAxes)


def list_topics(json_path="test_articles.json"):
    """List all available topics in the JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*60)
    print("AVAILABLE TOPICS")
    print("="*60)
    for key, topic in data['topics'].items():
        print(f"\n  {key}")
        print(f"    Name: {topic['name']}")
        print(f"    Description: {topic['description']}")
    print("\n" + "="*60 + "\n")


def main():
    """Main function to run comparison."""
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            json_file = sys.argv[2] if len(sys.argv) > 2 else "test_articles.json"
            list_topics(json_file)
            return
        elif sys.argv[1] == '--file':
            json_file = sys.argv[2] if len(sys.argv) > 2 else "test_articles.json"
            topic = sys.argv[3] if len(sys.argv) > 3 else None
        else:
            json_file = "test_articles.json"
            topic = sys.argv[1]
    else:
        json_file = "test_articles.json"
        topic = None
    
    # Load data
    data = load_articles(json_path=json_file, topic=topic)
    
    print(f"Topic: {data['name']}")
    print(f"Description: {data['description']}")
    print(f"\nArticle A: {data['article_a']['title']}")
    print(f"Article B: {data['article_b']['title']}")
    print("\nAnalyzing...\n")
    
    # Compare articles
    metrics_a, metrics_b = compare_articles(
        data['article_a']['content'],
        data['article_b']['content'],
        data['article_a']['title'],
        data['article_b']['title']
    )
    
    # Print summary
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{data['article_a']['title']}:")
    print(f"  Average sentiment: {metrics_a['avg_score']:.3f}")
    print(f"  Positive words: {metrics_a['positive']}, Negative words: {metrics_a['negative']}")
    
    print(f"\n{data['article_b']['title']}:")
    print(f"  Average sentiment: {metrics_b['avg_score']:.3f}")
    print(f"  Positive words: {metrics_b['positive']}, Negative words: {metrics_b['negative']}")
    
    diff = metrics_a['avg_score'] - metrics_b['avg_score']
    print(f"\nSentiment Difference: {diff:+.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
