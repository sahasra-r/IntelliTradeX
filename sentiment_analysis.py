# sentiment_analysis.py
from newsapi import NewsApiClient
from textblob import TextBlob

# Replace this with your NewsAPI key (free account)
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"

def get_sentiment(ticker):
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    articles = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=5)
    
    sentiments = []
    for article in articles['articles']:
        text = article['title'] + ". " + (article['description'] or "")
        polarity = TextBlob(text).sentiment.polarity
        sentiments.append(polarity)
    
    if not sentiments:
        return 0.0  # neutral if no news found
    
    avg_sentiment = sum(sentiments)/len(sentiments)
    return avg_sentiment
