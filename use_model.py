





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



sentence = "Can you track my order?"
tag = predict_class(sentence)
response = get_response(data, tag)
print(response)
