import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import numpy as np
import tensorflow
import json
import random

# this 3 are our vocab - token map - array of labels
words: list[str] = []
token_map: dict[str, int] = {}
labels: list[str] = []
# this is are our individual inputs ,
docs_query: list[list[str]] = []
docs_label: list[str] = []

# reading data from json file , tokenize it and create input and label for output
