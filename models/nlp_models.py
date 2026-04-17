"""NLP module: text classification and sentiment analysis for any dataset."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import json
from models.database import execute_db

# Sample text data for training — generic, domain-agnostic topics
SAMPLE_TEXTS = [
    ("Housing prices continue to rise in major urban areas across the country", "housing"),
    ("New railway stations along the metro line opened to passengers this week", "transport"),
    ("Schools adopt updated STEM curriculum for primary and secondary students", "education"),
    ("Solar panel installations increase across residential rooftops nationwide", "sustainability"),
    ("Population density in suburban districts exceeds city planning projections", "urban_planning"),
    ("Affordable housing developments feature smart home technology and green spaces", "housing"),
    ("Bus route optimization reduces average commute time for suburban commuters", "transport"),
    ("Government introduces coding education for secondary school students", "education"),
    ("Country targets 2 GW solar deployment within the next decade", "sustainability"),
    ("Urban master plan guides land use and zoning decisions for the next 15 years", "urban_planning"),
    ("Resale property prices in outer suburbs remain affordable for young families", "housing"),
    ("Metro line extension improves east-west connectivity across the city", "transport"),
    ("Polytechnic and vocational graduates see improved employment rates", "education"),
    ("Green building standards are now mandatory for new commercial developments", "sustainability"),
    ("Business district expansion designated as secondary central business area", "urban_planning"),
    ("Mid-range condominiums offer practical housing options for middle-income families", "housing"),
    ("Transit authority enhances first-and-last-mile public transport connectivity", "transport"),
    ("Universities rank among top programs in Asia for data science and AI", "education"),
    ("Electric vehicle adoption accelerates with expanded charging infrastructure", "sustainability"),
    ("New eco-district introduces pedestrian-friendly and car-free town centre design", "urban_planning"),
    ("Prices for large apartments in central areas reach record highs", "housing"),
    ("The cross-city rail line will be the longest fully underground transit route", "transport"),
    ("National library expands digital learning resources and databases", "education"),
    ("Government pledges net-zero emissions by 2050 under comprehensive green plan", "sustainability"),
    ("Waterfront area undergoes further development with new mixed-use projects", "urban_planning"),
    ("Property cooling measures affect resale transaction volumes significantly", "housing"),
    ("Ride-hailing and public transport integration improves commuter experience", "transport"),
    ("Skills development initiative supports mid-career transitions into data science", "education"),
    ("Waste-to-energy facilities help meet the nation's sustainability goals", "sustainability"),
    ("Former industrial zone relocation to free up land for residential development", "urban_planning"),
]

SENTIMENT_DATA = [
    ("The new metro station is really convenient and well-designed", "positive"),
    ("Very unhappy with the long construction delays at my apartment block", "negative"),
    ("Transport fares are reasonable compared to other major cities", "positive"),
    ("The bus service frequency needs significant improvement", "negative"),
    ("Love the new cycling paths and park connector network", "positive"),
    ("Housing prices are becoming unaffordable for young couples", "negative"),
    ("Our local schools provide excellent education quality overall", "positive"),
    ("Too much homework burden on primary school children every day", "negative"),
    ("Green spaces in new residential areas are beautiful and well-maintained", "positive"),
    ("Air quality during peak pollution season is terrible", "negative"),
    ("The housing grant scheme is very helpful for first-time buyers", "positive"),
    ("Crowded trains during peak hours are extremely uncomfortable", "negative"),
    ("Smart city initiatives make government services convenient to access", "positive"),
    ("Rising cost of living is a major concern for many residents", "negative"),
    ("Community events in residential neighborhoods build social bonds", "positive"),
    ("Noise pollution from nearby construction projects is very disruptive", "negative"),
    ("Clean streets and excellent public infrastructure throughout the city", "positive"),
    ("Rental prices in central areas are excessively high", "negative"),
    ("Congestion pricing system helps manage traffic flow effectively", "positive"),
    ("Limited parking spaces in older residential estates", "negative"),
]


def preprocess_text(text):
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_text_classifier():
    """Train a TF-IDF + Naive Bayes text classifier for topic categorization."""
    texts = [preprocess_text(t[0]) for t in SAMPLE_TEXTS]
    labels = [t[1] for t in SAMPLE_TEXTS]

    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Get top features per category
    feature_names = vectorizer.get_feature_names_out()
    top_features = {}
    for i, category in enumerate(model.classes_):
        top_indices = model.feature_log_prob_[i].argsort()[-5:][::-1]
        top_features[category] = [feature_names[j] for j in top_indices]

    metrics = {
        'model_name': 'TF-IDF + Naive Bayes',
        'task': 'Topic Classification',
        'accuracy': round(float(acc), 4),
        'classification_report': {k: v for k, v in report.items() if k not in ('accuracy',)},
        'categories': list(model.classes_),
        'top_features': top_features,
        'vocabulary_size': len(feature_names),
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'tfidf_sample': {
            'text': SAMPLE_TEXTS[0][0],
            'top_terms': sorted(
                zip(feature_names, X[0].toarray()[0]),
                key=lambda x: x[1], reverse=True
            )[:10]
        }
    }

    # Convert numpy types for JSON serialization
    metrics['tfidf_sample']['top_terms'] = [
        (str(term), round(float(score), 4))
        for term, score in metrics['tfidf_sample']['top_terms']
        if score > 0
    ]

    execute_db(
        "INSERT INTO model_metrics (model_name, model_type, metric_name, metric_value, parameters, dataset_size) VALUES (?,?,?,?,?,?)",
        ('NaiveBayes_TF-IDF', 'nlp', 'accuracy', acc,
         json.dumps({'alpha': 0.1}), len(texts))
    )

    return metrics


def train_sentiment_classifier():
    """Train a sentiment classifier on feedback text."""
    texts = [preprocess_text(t[0]) for t in SENTIMENT_DATA]
    labels = [t[1] for t in SENTIMENT_DATA]

    vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    metrics = {
        'model_name': 'TF-IDF + Logistic Regression',
        'task': 'Sentiment Analysis',
        'accuracy': round(float(acc), 4),
        'classes': list(model.classes_),
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'sample_predictions': [
            {'text': SENTIMENT_DATA[i][0], 'actual': labels[i], 'predicted': str(model.predict(X[i])[0])}
            for i in range(min(5, len(texts)))
        ]
    }

    execute_db(
        "INSERT INTO model_metrics (model_name, model_type, metric_name, metric_value, parameters, dataset_size) VALUES (?,?,?,?,?,?)",
        ('LogReg_Sentiment', 'nlp', 'accuracy', acc, None, len(texts))
    )

    return metrics


def classify_text(text):
    """Classify a single text into a topic category."""
    texts_train = [preprocess_text(t[0]) for t in SAMPLE_TEXTS]
    labels_train = [t[1] for t in SAMPLE_TEXTS]

    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_train = vectorizer.fit_transform(texts_train)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, labels_train)

    cleaned = preprocess_text(text)
    X_new = vectorizer.transform([cleaned])
    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0]

    return {
        'input_text': text,
        'predicted_category': prediction,
        'confidence': {cls: round(float(prob), 4) for cls, prob in zip(model.classes_, probabilities)}
    }
