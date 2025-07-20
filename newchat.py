import os     #for operating system-related functionalities
import json     #handling JSON data
import random     #for generating random numbers
import nltk     #for natural language processing tasks
import numpy as np     #for numerical computations
import torch     #framework for deep learning
import torch.nn as nn     #neural network module
import torch.nn.functional as F     #for operations like, activation functions
import torch.optim as optim     #for training models
from torch.utils.data import DataLoader, TensorDataset     #for loading and managing data in batches during training
from fuzzywuzzy import fuzz
nltk.download('punkt_tab')
nltk.download('wordnet')
#Class Definition
class ChatbotModel(nn.Module):     #inherts from nn.Module, base class for all neural network modules in Pytorch

    #Initialization
    def __init__(self, input_size, output_size):     #constructor with input_size as dimension of input data and output_size as dimension of output data 
        super(ChatbotModel, self).__init__()     #calls the constructor of the parent class
        self.fc1 = nn.Linear(input_size,256)     #maps input to 128-dimensional hidden layer
        self.fc2 = nn.Linear(256,128)     #maps 128-dimensionas to 64-dimensional hidden layer 
        self.fc3 = nn.Linear(128,output_size)     #maps 64-dimensions to output size
        self.relu = nn.ReLU()     #applies ReLU activation function for non-linearity
        self.dropout = nn.Dropout(0.5)     #applies dropout with 50% probability to prevent overfitting

    def forward(self,x):
        x= self.relu(self.fc1(x))     #applies first linear layer followed by ReLU
        x=self.dropout(x)     #applies dropout to result
        x=self.relu(self.fc2(x))     #applies second linear layer followed by ReLU
        x=self.dropout(x)     #applies dropout again
        x=self.fc3(x)     #applies the final linear layer 
        return x     #returns the output of the network 

#NLP system for chatbot 
class ChatbotAssistant:
    def __init__(self,intent_path,function_mappings=None):
        self.model=None
        self.intents_path=intent_path
        self.documents=[]
        self.vocabulary=[]
        self.intents=[]
        self.intents_responses={}

        self.function_mappings=function_mappings
        
        self.X=None
        self.y=None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words=nltk.word_tokenize(text)
        words=[lemmatizer.lemmatize(word.lower())for word in words]
        return words

    def bag_of_words(self,words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer=nltk.WordNetLemmatizer()
      
        if os.path.exists(self.intents_path):
            with open(self.intents_path,"r") as f:
                intents_data=json.load(f)
            for intent in intents_data["intents"]:
                if intent["tag"] not in self.intents:
                    self.intents.append(intent["tag"])
                    self.intents_responses[intent["tag"]]=intent["responses"]

                for pattern in intent["patterns"]:
                    pattern_words=self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words,intent["tag"]))

            self.vocabulary=sorted(list(set(self.vocabulary)))

    def prepare_data(self):
        bags=[]
        indices=[]
        for document in self.documents:
            words=document[0]
            bag=self.bag_of_words(words)
              
            intent_index=self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X=np.array(bags)
        self.y=np.array(indices)

    def train_model(self,batch_size,lr,epochs):
      X_tensor = torch.tensor(self.X, dtype=torch.float32)
      y_tensor = torch.tensor(self.y, dtype=torch.long)

      dataset=TensorDataset(X_tensor,y_tensor)
      loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

      self.model=ChatbotModel(self.X.shape[1],len(self.intents))

      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(self.model.parameters(), lr=lr)

      for epoch in range(epochs):
          running_loss=0.0
          for batch_X,batch_y in loader:
              optimizer.zero_grad()
              outputs=self.model(batch_X)
              loss=criterion(outputs,batch_y)
              loss.backward()
              optimizer.step()
              running_loss+=loss.item()
          
          print(f"Epoch {epoch+1}: Loss: {running_loss/len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(),model_path)

        with open(dimensions_path, "w") as f:
            json.dump({"input_size":self.X.shape[1], "output_size":len(self.intents)},f)
    
    def load_model(self,model_path,dimensions_path):
        with open(dimensions_path,"r") as f:
            dimensions = json.load(f)
        
        self.model=ChatbotModel(dimensions["input_size"],dimensions["output_size"])
        self.model.load_state_dict(torch.load(model_path,weights_only=True))
    

    def process_message(self, input_message, threshold=0.65, fuzzy_threshold=80):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            predicted_class_index = predicted_class_index.item()

        if confidence < threshold:
            # Try fuzzy matching as backup
            best_match = None
            best_score = 0
            for tag, patterns in self._get_intent_patterns().items():
                for pattern in patterns:
                    score = fuzz.ratio(input_message.lower(), pattern.lower())
                    if score > best_score:
                        best_score = score
                        best_match = tag

            if best_score >= fuzzy_threshold:
                predicted_intent = best_match
            else:
                predicted_intent = "Fallback"
        else:
            predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings and predicted_intent in self.function_mappings:
            self.function_mappings[predicted_intent]()

        return random.choice(self.intents_responses.get(predicted_intent, ["I'm sorry, I couldn't find a suitable response."]))

    def _get_intent_patterns(self):
        with open(self.intents_path, "r") as f:
            intents_data = json.load(f)
        return {intent["tag"]: intent["patterns"] for intent in intents_data["intents"]}

        
if __name__ == "__main__":
    # Create a dummy intents.json for testing if it doesn't exist
    if not os.path.exists("intentg.json"):
        dummy_intents = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["Hi", "Hello", "Hey"],
                    "responses": ["Hello!", "Hi there!", "Hey!"]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["Bye", "Goodbye", "See you later"],
                    "responses": ["Goodbye!", "See you later!", "Bye!"]
                }
            ]
        }
        with open("intentg.json", "w") as f:
            json.dump(dummy_intents, f, indent=4)

    assistant = ChatbotAssistant("intentg.json")
    assistant.parse_intents()
    assistant.prepare_data()

    if assistant.X is None or assistant.X.size == 0:
        print("Error: Dataset is empty. Please check your intentg.json file and data preparation steps.")
    else:
        batch_size = 8
        lr = 0.005
        epochs = 140
        assistant.train_model(batch_size, lr, epochs)
        assistant.save_model("chatbot_model.pth", "dimensions.json")
        assistant.load_model("chatbot_model.pth", "dimensions.json")

        print("""Hello! I'm SCI's Virtual Assistant. 
I can help you explore services, track shipments, check schedules, and more.
How may I assist you today?""")
        while True:
            message = input("Enter your message: ")
            if message.lower() == "quit":
                break
            print(assistant.process_message(message))