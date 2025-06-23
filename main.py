import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import subprocess
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yfinance as yf


def get_twitter_sentiment(query, start_date, end_date, limit=100):
    """Collect tweets using snscrape and compute average sentiment."""
    cmd = [
        "snscrape",
        "--jsonl",
        f"--since {start_date}",
        f"twitter-search",
        f"{query} until:{end_date}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("snscrape failed")
    tweets = [
        pd.json_normalize(eval(line))
        for line in result.stdout.splitlines()[:limit]
    ]
    df = pd.concat(tweets, ignore_index=True)
    analyzer = SentimentIntensityAnalyzer()
    sentiments = df['content'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return sentiments.mean()


def get_investing_news_sentiment(url="https://www.investing.com/indices/us-spx-500-news"):
    """Scrape investing.com news headlines and compute average sentiment."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    headlines = [a.get_text(strip=True) for a in soup.select('.js-article-item .title')]  # may change
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    return np.mean(sentiments)


def get_sp500_data(start_date, end_date):
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    return sp500


def prepare_dataset(sp500, twitter_s, news_s):
    df = sp500.copy()
    df['twitter_sentiment'] = twitter_s
    df['news_sentiment'] = news_s
    df['tomorrow_close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    df['target'] = (df['tomorrow_close'] > df['Close']).astype(int)
    features = df[['twitter_sentiment', 'news_sentiment']]
    labels = df['target']
    return features, labels


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.2f}")
    return model


def predict_next_day(model, twitter_s, news_s):
    X = np.array([[twitter_s, news_s]])
    pred = model.predict(X)[0]
    return "up" if pred == 1 else "down"


def main():
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=30)

    twitter_s = get_twitter_sentiment("S&P500", start_date.isoformat(), end_date.isoformat())
    news_s = get_investing_news_sentiment()

    sp500 = get_sp500_data(start_date.isoformat(), end_date.isoformat())
    X, y = prepare_dataset(sp500, twitter_s, news_s)
    model = train_model(X, y)

    next_move = predict_next_day(model, twitter_s, news_s)
    print(f"Predicted direction for next day: {next_move}")


if __name__ == "__main__":
    main()
