from flask import Flask, request, jsonify
from flask_cors import CORS
from newchat import ChatbotAssistant

app = Flask(__name__)
CORS(app)

# --- FIX 1: UNCOMMENT THESE LINES ---
# These lines create and load your chatbot
assistant = ChatbotAssistant("intentg.json")
assistant.parse_intents()
assistant.prepare_data()
# You might need to train the model first if you don't have the .pth file
# assistant.train_model(batch_size=8, lr=0.005, epochs=140) 
# assistant.save_model("chatbot_model.pth", "dimensions.json")
assistant.load_model("chatbot_model.pth", "dimensions.json")

@app.route("/chat", methods=["POST"])
def chat():
    # --- FIX 2: ADD THIS LINE ---
    global assistant
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"response": "Invalid request"}), 400

    user_message = data["message"]
    
    response = assistant.process_message(user_message)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)