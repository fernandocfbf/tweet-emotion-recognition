from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_tweet(data):
    tweets = [x["text"] for x in data]
    labels = [x["label"] for x in data]
    return tweets, labels

def get_sequences(tokenizer, tweets, maxlen=50):
    sequences = tokenizer.text_to_sequences(tweets)
    padded = pad_sequences(sequences, truncating="post", padding="post", maxlen=maxlen)
