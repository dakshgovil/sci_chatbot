import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from fuzzywuzzy import fuzz
nltk.download('punkt_tab')   
nltk.download('wordnet')  

# Class Definition
class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# NLP system for chatbot 
class ChatbotAssistant:
    def __init__(self, intent_path, function_mappings=None):
        self.model = None
        self.intents_path = intent_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.intent_patterns = {}
        self.function_mappings = function_mappings
        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, "r") as f:
                intents_data = json.load(f)
            for intent in intents_data["intents"]:
                tag = intent["tag"]
                if tag not in self.intents:
                    self.intents.append(tag)
                    self.intents_responses[tag] = intent["responses"]
                    self.intent_patterns[tag] = intent["patterns"]

                for pattern in intent["patterns"]:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, tag))

            self.vocabulary = sorted(list(set(self.vocabulary)))
        else:
            raise FileNotFoundError(f"Intent file not found at: {self.intents_path}")

    def prepare_data(self):
        if not self.documents:
            raise ValueError("No documents parsed. Check intent file and parsing.")

        bags = []
        indices = []
        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        if self.X is None or self.X.size == 0:
            raise ValueError("Training data is empty. Check data preparation.")

        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}: Loss: {running_loss/len(loader):.4f}")

    def save_model(self, model_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'vocabulary': self.vocabulary,
            'intents': self.intents,
            'intents_responses': self.intents_responses,
            'intent_patterns': self.intent_patterns,
            'input_size': self.X.shape[1],
            'output_size': len(self.intents)
        }
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model = ChatbotModel(checkpoint['input_size'], checkpoint['output_size'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.vocabulary = checkpoint['vocabulary']
        self.intents = checkpoint['intents']
        self.intents_responses = checkpoint['intents_responses']
        self.intent_patterns = checkpoint['intent_patterns']

    def process_message(self, input_message, threshold=0.85, fuzzy_threshold=85):
        if not self.model:
            raise RuntimeError("Model not loaded. Load model before processing.")

        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(bag_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()

        predicted_intent = None

        if confidence >= threshold:
            predicted_intent = self.intents[predicted_idx]
        else:
            best_match = None
            best_score = 0
            for tag, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    score = fuzz.ratio(input_message.lower(), pattern.lower())
                    if score > best_score:
                        best_score = score
                        best_match = tag

            if best_score >= fuzzy_threshold:
                predicted_intent = best_match
            else:
                predicted_intent = "Fallback"

        if self.function_mappings and predicted_intent in self.function_mappings:
            self.function_mappings[predicted_intent]()

        return random.choice(
            self.intents_responses.get(
                predicted_intent, 
                ["I'm sorry, I couldn't find a suitable response."]
            )
        )

if __name__ == "__main__":
    INTENT_FILE = "intentg.json"

    if not os.path.exists(INTENT_FILE):
        print(f"Error: Intent file '{INTENT_FILE}' not found in directory:")
        print(os.listdir())
        exit(1)

    assistant = ChatbotAssistant(INTENT_FILE)

    try:
        print("Parsing intents...")
        assistant.parse_intents()

        print("Preparing data...")
        assistant.prepare_data()

        print(f"Loaded dataset: {len(assistant.documents)} examples, {len(assistant.intents)} intents")
        print(f"Vocabulary size: {len(assistant.vocabulary)} words")

        batch_size = 8
        lr = 0.005
        epochs = 140

        print("\nTraining model...")
        assistant.train_model(batch_size, lr, epochs)

        print("\nSaving model...")
        assistant.save_model("chatbot_model.pth")

        print("\nReloading model...")
        assistant.load_model("chatbot_model.pth")

        print("""
        \nChatbot Ready!
        Hello! I'm SCI's Virtual Assistant. 
        I can help you explore services, track shipments, check schedules, and more.
        How may I assist you today?
        (Type 'quit' to exit)
        """)

        while True:
            message = input("You: ")
            if message.lower() in ["quit", "exit", "bye"]:
                break
            response = assistant.process_message(message)
            print("Bot:", response)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Check your intent file and data paths")
