"""
Simplified Bias Snapshot - Core sentiment analysis functionality
"""

import re
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize


def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')


def preprocess_text(text):
    """Clean and tokenize text."""
    # Clean text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation-only tokens
    tokens = [t for t in tokens if not all(c in '.,!?;:"()[]{}' for c in t)]
    
    return tokens


def analyze_sentiment(tokens):
    """Analyze sentiment for tokens."""
    sia = SentimentIntensityAnalyzer()
    
    results = []
    for token in tokens:
        scores = sia.polarity_scores(token)
        results.append({
            'word': token,
            'score': scores['compound']
        })
    
    return pd.DataFrame(results)


def create_chart(df, title="Sentiment Analysis"):
    """Create a simple sentiment bar chart."""
    # Get top positive and negative words
    positive = df[df['score'] > 0.05].nlargest(10, 'score')
    negative = df[df['score'] < -0.05].nsmallest(10, 'score')
    
    # Combine for plotting
    words = list(negative['word']) + list(positive['word'])
    scores = list(negative['score']) + list(positive['score'])
    colors = ['red'] * len(negative) + ['green'] * len(positive)
    
    if not words:
        print("No significant sentiment words found")
        return
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('Sentiment Score')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_text(text):
    """Main analysis function."""
    print(f"Analyzing: {text[:50]}...")
    
    # Download NLTK data
    download_nltk_data()
    
    # Process text
    tokens = preprocess_text(text)
    print(f"Found {len(tokens)} words")
    
    # Analyze sentiment
    df = analyze_sentiment(tokens)
    
    # Print summary
    avg_score = df['score'].mean()
    pos_words = len(df[df['score'] > 0.05])
    neg_words = len(df[df['score'] < -0.05])
    
    print(f"Average sentiment: {avg_score:.3f}")
    print(f"Positive words: {pos_words}, Negative words: {neg_words}")
    
    # Show chart
    create_chart(df)
    
    return df


if __name__ == "__main__":
    # Test with different texts
    import sys
    
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        # Default sample text
        text = """
        This product is absolutely amazing! I love how well it works and the quality is outstanding. 
        The customer service was fantastic and they were so helpful. I would definitely recommend 
        this to anyone looking for a great solution. Excellent value for money!
        """
    
    analyze_text(text)