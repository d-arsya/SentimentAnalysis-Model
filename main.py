from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model('model.h5')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 30  # same as used in training

# Define API
app = FastAPI()

class NewsHeadline(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: NewsHeadline):
    seq = tokenizer.texts_to_sequences([data.text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded)
    label_idx = pred.argmax(axis=1)[0]
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = label_map[label_idx]
    return {"sentiment": sentiment, "probabilities": pred[0].tolist()}
