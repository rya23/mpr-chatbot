import torch
from safetensors import safe_open
from model import *  # Import the model definition

# Load the saved safetensors file
import torch
import torch.nn as nn
import nltk
import numpy as np
import json
from nltk.stem import PorterStemmer

# nltk.download("punkt_tab")
stemmer = PorterStemmer()

with open("intents.json") as file:
    data = json.load(file)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


with safe_open("model_and_data.safetensors", framework="pt") as f:
    tensors = {key: f.get_tensor(key) for key in f.keys()}

# Retrieve all_words, tags, and model state_dict from the loaded tensors
all_words = tensors.pop("all_words")
tags = tensors.pop("tags")

# Recreate the model with the correct input and output sizes
input_size = int(all_words.item())
hidden_size = 8
output_size = int(tags.item())
model = NeuralNet(input_size, hidden_size, output_size)


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


all_words = []
tags = []
xy = []

for intent in data["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = sorted(set([stem(w) for w in all_words if w not in ["?", ".", "!"]]))
tags = sorted(set(tags))


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag


def predict_class(sentence):
    X = bag_of_words(tokenize(sentence), all_words)
    X = torch.from_numpy(X).float()
    output = model(X)
    _, predicted = torch.max(output, dim=0)
    return tags[predicted.item()]


def get_response(intents_json, predicted_tag):
    for intent in intents_json["intents"]:
        if intent["tag"] == predicted_tag:
            return np.random.choice(intent["responses"])


# Load the model's state_dict
model.load_state_dict(tensors)

# Now the model and data are ready to be used for inference

while True:
    sentence = input("You: ")
    tag = predict_class(sentence)
    response = get_response(data, tag)
    print("Chatbot:", response)
