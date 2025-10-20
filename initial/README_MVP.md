# Bias Snapshot MVP - Initial Evaluation

**Minimal Viable Product for Initial Assessment**

This folder contains a simplified version demonstrating the core functionality for initial evaluation with adaptive normalization for full-length articles.

## What's Included

### Scripts
- `bias_snapshot_mvp.py` - Main MVP with configurable article length support
- `bias_snapshot_debug.py` - Debug version with detailed normalization analysis

### Generated Outputs
1. **mvp_article_sentiment.png** - Bar chart showing sentiment scores (normalized for full articles)
2. **mvp_word_distribution.png** - Word-level distribution (histogram + pie chart, excludes neutral)
3. **mvp_results_table.csv** - Summary table of all articles

## Run the MVP

```bash
cd initial/
python bias_snapshot_mvp.py
```

### Configuration

Edit line 20 in `bias_snapshot_mvp.py` to toggle article types:

```python
USE_FULL_ARTICLES = True   # For full-length articles (normalized scoring)
USE_FULL_ARTICLES = False  # For summarized articles (raw scoring)
```

**Full articles:** Uses relative scoring (normalized against corpus mean)  
**Summarized:** Uses raw VADER scores (already more extreme)

## What It Demonstrates

### ✅ Required Elements

**1. Model Output (VADER Sentiment)**
- Article-level sentiment scores from VADER
- Visual comparison across all articles in dataset

**2. Preprocessing Evidence**
- Tokenization of text into individual words
- Cleaning (lowercase, punctuation removal)
- Sentiment scoring applied to each token

**3. Dataset Adequacy**
- 6 articles across 3 topics analyzed
- 373 words total with measurable sentiment distribution
- Balanced representation of positive/negative perspectives

**4. Exploratory Data Analysis**
- Word sentiment distribution histogram
- Sentiment category breakdown (pie chart)
- Article-level comparison table

### 📊 Key Findings

- **4.3%** positive words, **4.3%** negative words, **91.4%** neutral
- Mean sentiment nearly neutral (-0.001) across corpus
- Clear differences detectable between opposing article perspectives
- VADER successfully identifies sentiment-bearing language

### 🎯 Reflection

**Dataset Adequacy:** ✅ Confirmed
- Sufficient articles for comparison
- Measurable sentiment differences between perspectives
- Real-world topics with authentic bias patterns

**Preprocessing Quality:** ✅ Validated
- Clean tokenization and normalization working
- VADER lexicon applying correctly
- Distribution shows expected patterns (most words neutral)

**Normalization Insight:** ✅ Critical Discovery
- Full-length news articles naturally cluster near neutral
- Relative scoring (vs corpus mean) reveals true differences
- Pro/con stances align correctly with normalized scores
- Summarized articles show stronger absolute sentiment

**Final Goals Feasibility:** ✅ Realistic
- Side-by-side comparison: Already functional
- Bias visualization: Clear differences detected with normalization
- Web deployment: Backend logic is modular and ready

## Next Steps

The MVP validates that:
1. VADER can detect sentiment differences in news articles
2. The dataset provides adequate signal for bias analysis
3. The preprocessing pipeline works correctly
4. The project scope is achievable

Ready to proceed with full implementation in parent directory.


