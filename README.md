# Bias Snapshot - Sentiment Analysis Tool

A clean, lightweight Python tool for sentiment analysis using NLTK VADER.

## Features

- **Single Article Analysis**: Analyze sentiment of individual texts
- **Article Comparison**: Side-by-side comparison of two articles
- **Minimal Dashboard**: Key metrics at a glance
- **Clean Visualizations**: Simple, effective charts

## Quick Start

```bash
# Install dependencies
pip install nltk matplotlib pandas

# Analyze single text
python bias_snapshot.py "Your text here"

# Compare two articles
python compare_articles.py
```

## Files

- `bias_snapshot.py` - Single article sentiment analysis
- `compare_articles.py` - Compare two articles side-by-side
- `test_articles.json` - Sample article data for testing
- `requirements.txt` - Python dependencies

## Single Article Analysis

```bash
# Default sample
python bias_snapshot.py

# Your own text
python bias_snapshot.py "This product is amazing!"
```

**Output:**
- Console summary (word counts, average sentiment)
- Horizontal bar chart (top positive/negative words)

## Article Comparison

```bash
# Compare articles using default topic (remote_work)
python compare_articles.py

# Compare a specific topic
python compare_articles.py electric_vehicles
```

**Output:**
- Side-by-side bar charts showing top words from each article
- Minimal dashboard with:
  - Average sentiment scores
  - Positive/negative word counts
  - Sentiment difference between articles

**Available Topics:**
- `remote_work` - Perspectives on remote vs. office work (default)
- `electric_vehicles` - Electric vs. gas-powered transportation debate

## Test Data Format

Edit `test_articles.json` to add your own topics. Structure:

```json
{
  "topics": {
    "your_topic_key": {
      "name": "Display Name",
      "description": "Brief description",
      "article_a": {
        "title": "First Article Title",
        "source": "Source Name",
        "stance": "pro",
        "content": "Article text..."
      },
      "article_b": {
        "title": "Second Article Title",
        "source": "Source Name", 
        "stance": "con",
        "content": "Article text..."
      }
    }
  },
  "metadata": {
    "default_topic": "your_topic_key"
  }
}
```

## Code Structure

**bias_snapshot.py** - Core functions:
- `preprocess_text()` - Clean and tokenize
- `analyze_sentiment()` - VADER sentiment scoring
- `create_chart()` - Single article visualization
- `analyze_text()` - Main analysis pipeline

**compare_articles.py** - Comparison functions:
- `load_articles()` - Load from JSON
- `compare_articles()` - Analyze and compare two texts
- `create_comparison_chart()` - Side-by-side visualization
- `plot_dashboard()` - Minimal metrics dashboard

## Ready for Extension

Clean, modular code ready for:
- Web interfaces (FastAPI/Flask/Streamlit)
- Batch processing
- Additional metrics
- Custom visualizations