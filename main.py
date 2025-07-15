from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import json
import os
import random

stemmer = LancasterStemmer()


class preprocessing:
    def __init__(self) -> None:
        # this 3 are our vocab - token map - array of labels
        self.words: list[str] = []
        self.token_map: dict[str, int] = {}
        self.labels: list[str] = []
        # this is are our individual inputs ,
        self.docs_query: list[list[str]] = []
        self.docs_label: list[str] = []

    # reading data from json file , tokenize it and create input and label for output
    #
    def preprocess_data(self):
        words: list[str] = []
        token_map: dict[str, int] = {}
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
                self.docs_query.append(stemmed_pattern)
                self.docs_label.append(intent["tag"])

        self.labels = sorted(list(set(self.docs_label)))
        self.words = sorted(words)
        self.token_map = {"<START>": 0, "<PAD>": 1, "<UNKNOWN>": 2, **token_map}
        training_input = np.zeros((len(self.docs_query), len(self.words)))
        training_output = np.zeros((len(self.docs_query), len(self.labels)))
        for index, item in enumerate(self.docs_query):
            for w in item:
                # i know for fact that they are , but check if so if in future change previos code , this remain the same
                if w in self.words:
                    training_input[index][self.words.index(w)] = 1
            training_output[index][self.labels.index(self.docs_label[index])] = 1
        # bag of word is working
        return training_input, training_output

    def tokenize_user_input(self, input):
        # there is not control over user input size
        output_bow = np.zeros((1, len(self.words)))
        input = (
            input.replace(",", "")
            .replace("!", "")
            .replace("?", "")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "")
        )
        for w in input.split(" "):
            if w in self.words:
                output_bow[0][self.words.index(w)] = 1
        return output_bow

    def get_lable(self, index):
        return self.labels[index]


preprocessor = preprocessing()
training_input, training_output = preprocessor.preprocess_data()


def train_model():
    tf.keras.backend.clear_session()
    model = Sequential(
        [
            Input(shape=(training_input.shape[1],)),
            Dense(8, activation="relu"),
            Dense(8, activation="relu"),
            Dense(training_output.shape[1], activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(training_input, training_output, epochs=300, batch_size=8, verbose=1)
    model.save("model.keras")
    return model


def load_model():
    return tf.keras.models.load_model("model.keras")


if os.path.exists("model.keras"):
    model = load_model()
else:
    model = train_model()


def interact_with_user():
    print("hi , how are you doing to day ? ask me , or quit with(exit) :")
    while True:
        user_input = input("you:")
        if user_input.lower() == "exit":
            break
        tokenize_user_input = preprocessor.tokenize_user_input(user_input)
        predict_result = model.predict(tokenize_user_input)
        predict_result = np.argmax(predict_result)
        predict_result = preprocessor.get_lable(predict_result)
        with open("./intents.json") as f:
            data = json.load(f)
        for item in data["intents"]:
            if item["tag"] == predict_result:
                answer = random.choice(item["responses"])
                print("assistanse:", answer)


interact_with_user()
