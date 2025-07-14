import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import json

# this 3 are our vocab - token map - array of labels
words: list[str] = []
token_map: dict[str, int] = {}
labels: list[str] = []
# this is are our individual inputs ,
docs_query: list[list[str]] = []
docs_label: list[str] = []

# reading data from json file , tokenize it and create input and label for output
with open("./intents.json") as f:
    data = json.load(f)
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        pattern = (
            pattern.replace(",", "")
            .replace("!", "")
            .replace("?", "")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "")
        )
        stemmed_pattern = [stemmer.stem(w.lower()) for w in pattern.split(" ")]
        for word in stemmed_pattern:
            if word not in token_map.keys():
                token_map[word] = len(token_map) + 3
                words.append(word)
        docs_query.append(stemmed_pattern)
        docs_label.append(intent["tag"])

labels = sorted(list(set(docs_label)))
words = sorted(words)
token_map = {"<START>": 0, "<PAD>": 1, "<UNKNOWN>": 2, **token_map}
training_input = np.zeros((len(docs_query), len(words)))
training_output = np.zeros((len(docs_query), len(labels)))
for index, item in enumerate(docs_query):
    for w in item:
        # i know for fact that they are , but check if so if in future change previos code , this remain the same
        if w in words:
            training_input[index][words.index(w)] = 1
    training_output[index][labels.index(docs_label[index])] = 1
# bag of word is working
tf.keras.backend.clear_session()

model = Sequential(
    [
        Input(shape=(training_input.shape[1],)),
        Dense(8, activation="relu"),
        Dense(8, activation="relu"),
        Dense(training_output.shape[1], activation="softmax"),
    ]
)

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(training_input, training_output, epochs=1000, batch_size=8, verbose=1)

model.save("model.h5")
