import os
from models.sentiment.model import return_call_sentiment


sentiment_data_dir = os.path.join("test_data", "sentiment")


def test_positive_conversation():
    with open(os.path.join(sentiment_data_dir, "positive_conversation.txt"), "r") as file:
        text_data = file.read().replace("\n", "")

    assert return_call_sentiment(text_data) == "POSITIVE"


def test_negative_conversation():
    with open(os.path.join(sentiment_data_dir, "negative_conversation.txt"), "r") as file:
        text_data = file.read().replace("\n", "")

    assert return_call_sentiment(text_data) == "NEGATIVE"


def test_neutral_conversation():
    with open(os.path.join(sentiment_data_dir, "neutral_conversation.txt"), "r") as file:
        text_data = file.read().replace("\n", "")

    assert return_call_sentiment(text_data) == "NEUTRAL"
