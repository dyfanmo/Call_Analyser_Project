from transformers import pipeline
from models.sentiment.constants import neutral_thresehold


def return_call_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)
    return "NEUTRAL" if result[0]["score"] < neutral_thresehold else result[0]["label"]
