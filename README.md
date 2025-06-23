# Sentiment-based US500 Direction Predictor

This project demonstrates a simple approach to predict the next-day direction of the S&P 500 using sentiment analysis of news from Investing.com and tweets mentioning the index.

## Requirements

```
pip install -r requirements.txt
```

The main dependencies are:

- `pandas`
- `numpy`
- `scikit-learn`
- `yfinance`
- `vaderSentiment`
- `beautifulsoup4`
- `requests`
- `snscrape`

`snscrape` is used to collect tweets without requiring Twitter API credentials.

## Usage

Run the script directly:

```
python main.py
```

The script fetches recent tweets and news headlines, computes their sentiment scores, downloads recent S&P 500 prices via `yfinance`, trains a logistic regression model, and prints the predicted direction ("up" or "down") for the next trading day.

The example uses the last 30 days of data. You can modify the code to change the training window or add more features as desired.
