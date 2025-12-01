# Bias Snapshot - Sentiment Analysis Tool

A Python tool for detecting and visualizing sentiment bias in news articles using VADER sentiment analysis with hybrid normalization.

## Features

- **Article Comparison**: Side-by-side comparison of articles on the same topic
- **Hybrid Normalization**: Balances within-topic differences with global corpus context
- **Word-Level Highlighting**: Interactive web dashboard with sentiment word highlighting
- **Visualizations**: Professional charts showing sentiment distributions and comparisons
- **Standalone Analysis**: Command-line tool for batch analysis

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Analysis

```bash
# Analyze all articles and generate visualizations
python bias_snapshot_mvp.py
```

### Run Web Dashboard

```bash
# Start the Flask web server
python app.py

# Open browser to:
# http://localhost:5000
```

## Project Structure

```
.
├── bias_snapshot_mvp.py    # Standalone analysis script
├── app.py                  # Flask web dashboard
├── templates/
│   └── index.html          # Web dashboard frontend
├── data/
│   ├── real_articles.json         # Summarized articles
│   └── real_articles_full.json    # Full-length articles
├── output/                 # Generated outputs
│   ├── mvp_article_sentiment.png
│   ├── mvp_word_distribution.png
│   └── mvp_results_table.csv
└── requirements.txt
```

## Analysis Script (`bias_snapshot_mvp.py`)

Standalone script for analyzing article sentiment with hybrid normalization.

### Configuration

Edit line 17 in `bias_snapshot_mvp.py`:

```python
USE_FULL_ARTICLES = True   # For full-length articles (hybrid normalization)
USE_FULL_ARTICLES = False  # For summarized articles (raw scoring)
```

### Outputs

1. **mvp_article_sentiment.png** - Pairwise comparison chart showing sentiment scores
2. **mvp_word_distribution.png** - Word-level sentiment distribution (histogram + pie chart)
3. **mvp_results_table.csv** - Summary table of all articles with scores

### Features

- **Hybrid Normalization**: 40% pairwise + 60% corpus-relative
  - Preserves within-topic differences
  - Maintains global corpus context
  - Prevents forcing articles into opposition
- **Adaptive Scoring**: Different thresholds for full vs summarized articles
- **Neutral Word Filtering**: Excludes neutral words (score = 0) from analysis

## Web Dashboard (`app.py`)

Interactive Flask web application for comparing articles with word-level highlighting.

### Features

- **Topic Selection**: Dropdown to select any topic from the dataset
- **Side-by-Side Comparison**: View both articles (A and B) simultaneously
- **Sentiment Scores**: 
  - Normalized score (hybrid normalization)
  - Raw score (absolute VADER sentiment)
  - Positive/negative word counts
- **Word Highlighting**: 
  - 🟢 Green = Positive sentiment words
  - 🔴 Pink = Negative sentiment words
  - Hover over words to see exact sentiment score

### Usage

1. Start the server: `python app.py`
2. Open `http://localhost:5000` in your browser
3. Select a topic from the dropdown
4. View side-by-side comparison with highlighted sentiment words

## Hybrid Normalization

The tool uses a hybrid normalization approach that combines:

- **40% Pairwise**: Normalizes relative to the topic pair mean (highlights within-topic differences)
- **60% Corpus-Relative**: Normalizes relative to the global corpus mean (maintains global context)

This approach:
- ✅ Shows relative differences between articles on the same topic
- ✅ Preserves whether articles are positive/negative relative to the corpus
- ✅ Prevents forcing articles into opposition when they're on the same side
- ✅ Allows neutral articles to remain neutral

### Example

If two articles both have positive sentiment:
- **Pure Pairwise**: Would force one positive, one negative
- **Hybrid**: Both can remain positive, but shows which is MORE positive

## Data Format

Articles are stored in JSON format:

```json
{
  "topics": {
    "topic_key": {
      "name": "Topic Display Name",
      "description": "Brief description",
      "article_a": {
        "title": "Article Title",
        "source": "Source Name",
        "stance": "pro|con|neutral",
        "content": "Article text..."
      },
      "article_b": {
        "title": "Article Title",
        "source": "Source Name",
        "stance": "pro|con|neutral",
        "content": "Article text..."
      }
    }
  }
}
```

## Technical Details

### Sentiment Analysis

- **Model**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Preprocessing**: 
  - Tokenization (NLTK Punkt)
  - Lowercase conversion
  - Punctuation removal
- **Scoring**: Compound score from VADER (-1 to +1)

### Normalization Methods

1. **Corpus-Relative**: `score - corpus_mean`
2. **Pairwise**: `score - pair_mean`
3. **Hybrid**: `0.4 × pairwise + 0.6 × corpus_relative`

### Output Statistics

- Article-level sentiment scores
- Word-level sentiment distribution
- Positive/negative/neutral word counts
- Corpus mean and median
- Standard deviation

## Requirements

- Python 3.7+
- nltk >= 3.8
- matplotlib >= 3.7.0
- pandas >= 2.0.0
- flask >= 2.3.0
- numpy >= 1.24.0

## Development

The codebase is designed to be:
- **Standalone**: `bias_snapshot_mvp.py` has no external dependencies
- **Modular**: Functions can be imported and reused
- **Minimal**: Ultra-MVP approach with essential features only

## License

This project is part of a university course assignment.
