from flask import Flask, render_template, jsonify
import numpy as np
from bias_snapshot import (
    load_data, normalize_hybrid, 
    preprocess_text, analyze_sentiment, download_nltk_data
)

app = Flask(__name__)

download_nltk_data()
articles_data = load_data()

articles_by_topic = {}
for article in articles_data:
    topic = article['topic']
    if topic not in articles_by_topic:
        articles_by_topic[topic] = {}
    article_key = 'a' if 'ARTICLE_A' in article['article_label'] else 'b'
    articles_by_topic[topic][article_key] = article


@app.route('/')
def index():
    topics = list(articles_by_topic.keys())
    return render_template('index.html', topics=topics)


@app.route('/api/articles/<topic>')
def get_articles(topic):
    if topic not in articles_by_topic:
        return jsonify({'error': 'Topic not found'}), 404
    
    topic_data = articles_by_topic[topic]
    result = {}
    
    for key in ['a', 'b']:
        if key in topic_data:
            article = topic_data[key]
            result[f'article_{key}'] = {
                'title': article['title'],
                'source': article['source'],
                'stance': article['stance'],
                'content': article['content']
            }
    
    return jsonify(result)


@app.route('/api/analyze/<topic>')
def analyze_topic(topic):
    if topic not in articles_by_topic:
        return jsonify({'error': 'Topic not found'}), 404
    
    topic_data = articles_by_topic[topic]
    results = {}
    all_scores = []
    article_scores_dict = {}
    
    for key in ['a', 'b']:
        if key not in topic_data:
            continue
            
        article = topic_data[key]
        tokens = preprocess_text(article['content'])
        df = analyze_sentiment(tokens)
        
        word_scores = {}
        non_neutral_scores = []
        for _, row in df.iterrows():
            score = row['score']
            word = row['word']
            if score != 0.0:
                word_scores[word] = float(score)
                non_neutral_scores.append(score)
        
        all_scores.extend(non_neutral_scores)
        article_scores_dict[(topic, key)] = non_neutral_scores
        
        raw_avg = np.mean(non_neutral_scores) if non_neutral_scores else 0.0
        
        results[f'article_{key}'] = {
            'title': article['title'],
            'source': article['source'],
            'stance': article['stance'],
            'content': article['content'],
            'raw_score': float(raw_avg),
            'word_scores': word_scores,
            'positive_count': len([s for s in non_neutral_scores if s > 0.01]),
            'negative_count': len([s for s in non_neutral_scores if s < -0.01])
        }
    
    if len(article_scores_dict) == 2:
        for key in ['a', 'b']:
            if f'article_{key}' in results:
                raw_avg = results[f'article_{key}']['raw_score']
                normalized, _ = normalize_hybrid(raw_avg, article_scores_dict, topic, all_scores, pairwise_weight=0.4)
                results[f'article_{key}']['normalized_score'] = float(normalized)
    
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

