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
- `dataset_stats.py` - Corpus-wide statistics and word distribution analysis
- `test_articles.json` - Sample article data for testing
- `real_articles copy.json` - Real-world articles for analysis
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
# Compare articles using default topic (first in JSON)
python compare_articles.py

# List all available topics in a file
python compare_articles.py --list
python compare_articles.py --list "real_articles copy.json"

# Compare a specific topic from test_articles.json
python compare_articles.py demo_remote_work

# Compare from a different JSON file
python compare_articles.py --file "real_articles copy.json" ai_labor_automation
python compare_articles.py --file "real_articles copy.json" trump_tariffs
python compare_articles.py --file "real_articles copy.json" ai_bubble_boom
```

**Output:**
- Side-by-side bar charts showing top words from each article (titles auto-truncated)
- Minimal dashboard with:
  - Average sentiment scores
  - Positive/negative word counts
  - Sentiment difference between articles (color-coded)

**Demo Topics (test_articles.json):**
- `demo_remote_work` - Perspectives on remote vs. office work
- `demo_electric_vehicles` - Electric vs. gas-powered transportation debate

**Real Topics (real_articles copy.json):**
- `ai_labor_automation` - AI's impact on jobs and employment
- `trump_tariffs` - Trump administration tariff policies
- `ai_bubble_boom` - AI investment: bubble or sustainable boom

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

## Dataset Statistics

Analyze word-level sentiment distribution across your entire corpus:

```bash
# Analyze all articles in a dataset
python dataset_stats.py "real_articles copy.json"
```

**Output:**
- Console statistics: corpus overview, word distribution, top frequent words
- Article comparison table (exported to CSV)
- Two visualizations:
  - `word_sentiment_distribution.png` - Distribution of non-neutral word scores (histogram + pos/neg split)
  - `sentiment_breakdown.png` - Overall categories (pie chart) + granular breakdown (bar chart)

**Key Insights:**
- Shows word sentiment follows a distribution (most words are neutral)
- Reveals corpus-level patterns across all articles
- Useful for research presentations and project documentation

## Ready for Extension

Clean, modular code ready for:
- Web interfaces (FastAPI/Flask/Streamlit)
- Batch processing
- Additional metrics
- Custom visualizations