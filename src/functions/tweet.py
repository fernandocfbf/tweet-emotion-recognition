from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def get_tweet(data):
    tweets = [x["text"] for x in data]
    labels = [x["label"] for x in data]
    return tweets, labels

def get_sequences(tokenizer, tweets, maxlen=50):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences, truncating="post", padding="post", maxlen=maxlen)
    return padded

def label_to_number(labels, class_to_index):
    return np.array([class_to_index.get(x) for x in labels])
