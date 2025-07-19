from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from newchat import ChatbotAssistant
import os

app = Flask(
    __name__,
    template_folder="templates",  # Path to index.html
    static_folder="static"        # Path to images and icons
)
CORS(app)

# Initialize the assistant
assistant = ChatbotAssistant("intentg.json")
assistant.parse_intents()
assistant.prepare_data()
assistant.load_model("chatbot_model.pth", "dimensions.json")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global assistant
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"response": "Invalid request"}), 400

    user_message = data["message"]
    response = assistant.process_message(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
