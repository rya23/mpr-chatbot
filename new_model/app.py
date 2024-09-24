import torch
from fastapi import FastAPI
from pydantic import BaseModel
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# Define the model architecture again (must match the one used during training)
class ChatbotModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ChatbotModel, self).__init__()
        self.fc = nn.Linear(input_size, 128)  # Hidden layer with 128 units
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# Load your trained model
input_size = 1539  # This should be set to the size of your input features
num_classes = 11  # This should be set to the number of classes in your label encoder

model = ChatbotModel(input_size, num_classes)
model.load_state_dict(torch.load("chatbot_model.pth"))
model.eval()  # Set model to eval mode

# Load your CSV with Intent and Bot Response
df = pd.read_csv("intent.csv")  # Replace with actual CSV path

# Simulating vectorizer and label encoder (you should load them if you have them saved)
# vectorizer = CountVectorizer()
# label_encoder = LabelEncoder()

vectorizer = (
    CountVectorizer()
)  # You can also use TF-IDF by replacing this with TfidfVectorizer
X = vectorizer.fit_transform(
    df["User Query"]
).toarray()  # Tokenize and convert to BoW vectors

# 3. Label Encoding for Intent column
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Intent"])


# Preprocess the user input (tokenization)
def preprocess_input(user_input):
    input_vector = vectorizer.transform([user_input]).toarray()
    return torch.tensor(input_vector, dtype=torch.float32)


# Predict intent
def predict_intent(user_input):
    with torch.no_grad():
        input_tensor = preprocess_input(user_input)
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        intent = label_encoder.inverse_transform(predicted.numpy())[0]
        return intent


# Get bot response for the predicted intent
def get_bot_response(intent):
    response = df[df["Intent"] == intent]["Bot Response"].values[0]
    return response


# print("Chatbot is ready to chat! Type 'exit' to end.")
# while True:
#     user_input = input("You: ")
#     print(f"You : {user_input}")
#     if user_input.lower() == "exit":
#         break
#     user_input = repr(user_input)
#     predicted_intent = predict_intent(user_input)
#     bot_response = get_bot_response(predicted_intent)
#     print("Bot:", bot_response)


app = FastAPI()


class Query(BaseModel):
    message: str

@app.post("/chat/")
async def chat(query: Query):
    user_message = query.message
    predicted_intent = predict_intent(user_message)
    bot_response = get_bot_response(predicted_intent)
    return {"intent": predicted_intent, "response": bot_response}
