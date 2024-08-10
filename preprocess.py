import nltk
import numpy as np
import json
from nltk.stem import PorterStemmer

# nltk.download("punkt_tab")
stemmer = PorterStemmer()

with open("intents.json") as file:
    data = json.load(file)


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag


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


import torch
import torch.nn as nn
import torch.optim as optim
from safetensors import safe_open
from safetensors.torch import save_file


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


input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
# model = NeuralNet(input_size, hidden_size, output_size)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(1000):
#     for pattern_sentence, tag in xy:
#         X = bag_of_words(pattern_sentence, all_words)
#         X = torch.from_numpy(X).float()
#         y = tags.index(tag)
#         y = torch.tensor(y)

#         output = model(X)
#         loss = criterion(output, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     if (epoch + 1) % 100 == 0:
#         print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}")
#         if loss.item() == 0 :
#             break


# model_state = model.state_dict()
# save_file(model_state, "model.safetensors")
tensors = {}

model = NeuralNet(input_size, hidden_size, output_size)

with safe_open("model.safetensors", framework="pt") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)
model.load_state_dict(tensors)


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


sentence = input("Print your Question :")


tag = predict_class(sentence)
response = get_response(data, tag)
print(response)
