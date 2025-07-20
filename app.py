import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from newchat import ChatbotAssistant

def create_app():
    # Initialize Flask app with proper paths
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static")
    )
    CORS(app)

    # Configure file paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INTENT_FILE = os.path.join(BASE_DIR, "intentg.json")
    MODEL_FILE = os.path.join(BASE_DIR, "chatbot_model.pth")
    DIMENSIONS_FILE = os.path.join(BASE_DIR, "dimensions.json")

    # Initialize chatbot assistant with error handling
    try:
        assistant = ChatbotAssistant(INTENT_FILE)
        assistant.parse_intents()
        assistant.prepare_data()
        
        if not all(os.path.exists(f) for f in [MODEL_FILE, DIMENSIONS_FILE]):
            raise FileNotFoundError("Model files missing. Please train the model first.")
            
        assistant.load_model(MODEL_FILE, DIMENSIONS_FILE)
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        assistant = None

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/chat", methods=["POST"])
    def chat():
        if assistant is None:
            return jsonify({"response": "Chatbot initialization failed. Please try again later."}), 500

        try:
            data = request.get_json()
            if not data or "message" not in data:
                return jsonify({"response": "Invalid request format"}), 400

            user_message = data["message"]
            response = assistant.process_message(user_message)
            return jsonify({"response": response})

        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return jsonify({"response": "Sorry, I encountered an error processing your request."}), 500

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))