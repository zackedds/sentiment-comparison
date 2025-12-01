import json
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')


def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if not all(c in '.,!?;:"()[]{}' for c in t)]
    return tokens


def analyze_sentiment(tokens):
    sia = SentimentIntensityAnalyzer()
    results = []
    for token in tokens:
        scores = sia.polarity_scores(token)
        results.append({
            'word': token,
            'score': scores['compound']
        })
    return pd.DataFrame(results)


def load_data():
    json_path = "data/real_articles_full.json"
    
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


def normalize_hybrid(raw_avg, article_scores_dict, topic, all_scores, pairwise_weight=0.4):
    corpus_mean = np.mean(all_scores)
    corpus_relative = raw_avg - corpus_mean
    
    topic_articles = [scores for (t, a), scores in article_scores_dict.items() if t == topic]
    if len(topic_articles) == 2:
        pair_mean = np.mean([np.mean(scores) for scores in topic_articles])
        pairwise = raw_avg - pair_mean
        baseline = pair_mean
    else:
        pairwise = corpus_relative
        baseline = corpus_mean
    
    hybrid_score = (pairwise_weight * pairwise) + ((1 - pairwise_weight) * corpus_relative)
    return hybrid_score, baseline


def analyze_dataset(articles):
    download_nltk_data()
    
    results = []
    all_scores = []
    article_dfs = []
    article_scores_dict = {}
    
    for article in articles:
        tokens = preprocess_text(article['content'])
        df = analyze_sentiment(tokens)
        article_dfs.append((article, df))
        non_neutral_scores = [s for s in df['score'].tolist() if s != 0.0]
        all_scores.extend(non_neutral_scores)
        
        topic = article['topic']
        article_key = 'a' if 'ARTICLE_A' in article['article_label'] else 'b'
        article_scores_dict[(topic, article_key)] = non_neutral_scores
    
    corpus_mean = np.mean(all_scores)
    corpus_median = np.median(all_scores)
    threshold = 0.01
    
    for article, df in article_dfs:
        non_neutral = df[df['score'] != 0.0]['score']
        raw_avg = non_neutral.mean() if len(non_neutral) > 0 else 0.0
        
        topic = article['topic']
        article_key = 'a' if 'ARTICLE_A' in article['article_label'] else 'b'
        
        display_score, baseline = normalize_hybrid(raw_avg, article_scores_dict, topic, all_scores, pairwise_weight=0.4)
        
        positive = len(df[df['score'] > threshold])
        negative = len(df[df['score'] < -threshold])
        
        results.append({
            'Topic': article['topic'],
            'Article': article['article_label'],
            'Stance': article['stance'],
            'Sentiment Score': f"{display_score:.4f}",
            'Raw Score': f"{raw_avg:.4f}",
            'Positive Words': positive,
            'Negative Words': negative,
            'Total Words': len(df)
        })
    
    return pd.DataFrame(results), all_scores, corpus_mean, True, corpus_median


def create_mvp_visualizations(results_df, all_scores, corpus_mean, use_normalization, corpus_median):
    results_df['Sentiment Numeric'] = results_df['Sentiment Score'].astype(float)
    
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    topics = results_df['Topic'].unique()
    x_pos = 0
    bar_width = 0.35
    
    for topic in topics:
        topic_df = results_df[results_df['Topic'] == topic].copy()
        article_a = topic_df[topic_df['Article'].str.contains('ARTICLE_A')]
        article_b = topic_df[topic_df['Article'].str.contains('ARTICLE_B')]
        
        if len(article_a) > 0 and len(article_b) > 0:
            score_a = article_a['Sentiment Numeric'].iloc[0]
            score_b = article_b['Sentiment Numeric'].iloc[0]
            stance_a = article_a['Stance'].iloc[0]
            stance_b = article_b['Stance'].iloc[0]
            
            ax1.barh(x_pos - bar_width/2, score_a, bar_width, 
                     color='#2E8B57' if score_a > 0 else '#DC143C' if score_a < 0 else '#808080',
                     alpha=0.8, edgecolor='black', linewidth=1.5,
                     label='Article A' if topic == topics[0] else '')
            ax1.barh(x_pos + bar_width/2, score_b, bar_width,
                     color='#2E8B57' if score_b > 0 else '#DC143C' if score_b < 0 else '#808080',
                     alpha=0.8, edgecolor='black', linewidth=1.5,
                     label='Article B' if topic == topics[0] else '')
            
            ax1.text(score_a, x_pos - bar_width/2, f'  {score_a:.3f} [{stance_a}]',
                    va='center', fontsize=8, fontweight='bold')
            ax1.text(score_b, x_pos + bar_width/2, f'  {score_b:.3f} [{stance_b}]',
                    va='center', fontsize=8, fontweight='bold')
        
        x_pos += 1
    
    ax1.set_yticks(range(len(topics)))
    ax1.set_yticklabels(topics, fontsize=10, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlabel('Relative Sentiment (Hybrid: 40% Pairwise, 60% Corpus)', fontweight='bold', fontsize=12)
    ax1.set_title('Sentiment Analysis - Pairwise Comparison\n(Hybrid Normalization)', fontweight='bold', fontsize=14, pad=15)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.6, zorder=0)
    ax1.grid(alpha=0.3, axis='x', linestyle=':', zorder=0)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    fig1.savefig('output/mvp_article_sentiment.png', dpi=300, bbox_inches='tight')
    
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))
    non_neutral_scores = [s for s in all_scores if s != 0.0]
    positive_scores = [s for s in non_neutral_scores if s > 0]
    negative_scores = [s for s in non_neutral_scores if s < 0]
    
    ax2.hist(negative_scores, bins=20, alpha=0.75, color='#DC143C', 
             edgecolor='black', label=f'Negative ({len(negative_scores)})')
    ax2.hist(positive_scores, bins=20, alpha=0.75, color='#2E8B57', 
             edgecolor='black', label=f'Positive ({len(positive_scores)})')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    
    if use_normalization:
        ax2.axvline(x=corpus_mean, color='orange', linestyle=':', linewidth=2, 
                   label=f'Corpus Mean ({corpus_mean:.3f})', alpha=0.7)
        ax2.axvline(x=corpus_median, color='purple', linestyle=':', linewidth=2,
                   label=f'Corpus Median ({corpus_median:.3f})', alpha=0.7)
    
    ax2.set_xlabel('Sentiment Score', fontweight='bold')
    ax2.set_ylabel('Word Count', fontweight='bold')
    ax2.set_title('Word Sentiment Distribution\n(Excluding Neutral Words)', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3, axis='y')
    
    positive = sum(1 for s in all_scores if s > 0.05)
    negative = sum(1 for s in all_scores if s < -0.05)
    neutral = len(all_scores) - positive - negative
    
    ax3.pie([positive, negative, neutral], 
            labels=[f'Positive\n{positive}', f'Negative\n{negative}', f'Neutral\n{neutral}'],
            colors=['#2E8B57', '#DC143C', '#808080'],
            autopct='%1.1f%%', startangle=90,
            textprops={'fontweight': 'bold'})
    ax3.set_title('Sentiment Category Distribution', fontweight='bold')
    
    plt.tight_layout()
    fig2.savefig('output/mvp_word_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(results_df, all_scores, corpus_mean, use_normalization, corpus_median):
    print("\n" + "="*70)
    print("BIAS SNAPSHOT - RESULTS")
    print("="*70)
    
    print(f"\nDataset Overview:")
    print(f"  Articles: {len(results_df)}")
    print(f"  Words: {len(all_scores)}")
    print(f"  Topics: {results_df['Topic'].nunique()}")
    
    threshold = 0.01
    positive = sum(1 for s in all_scores if s > threshold)
    negative = sum(1 for s in all_scores if s < -threshold)
    neutral = len(all_scores) - positive - negative
    
    print(f"\nSentiment Distribution:")
    print(f"  Positive: {positive} ({positive/len(all_scores)*100:.1f}%)")
    print(f"  Negative: {negative} ({negative/len(all_scores)*100:.1f}%)")
    print(f"  Neutral: {neutral} ({neutral/len(all_scores)*100:.1f}%)")
    print(f"  Mean: {corpus_mean:.4f}")
    print(f"  Median: {corpus_median:.4f}")
    print(f"  Std Dev: {np.std(all_scores):.4f}")
    
    print("\n" + "="*70)
    print("ARTICLE-LEVEL RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70 + "\n")


def main():
    articles = load_data()
    results_df, all_scores, corpus_mean, use_normalization, corpus_median = analyze_dataset(articles)
    
    print_summary(results_df, all_scores, corpus_mean, use_normalization, corpus_median)
    
    results_df.to_csv('output/mvp_results_table.csv', index=False)
    create_mvp_visualizations(results_df, all_scores, corpus_mean, use_normalization, corpus_median)


if __name__ == "__main__":
    main()