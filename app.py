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
    MODEL_FILE = os.path.join(BASE_DIR, "chatbot_model.pth")  # Only model file now

    # Initialize chatbot assistant with error handling
    try:
        assistant = ChatbotAssistant(INTENT_FILE)
        
        # Check if model file exists
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError("Model file missing. Please train the model first.")
            
        # Load model with metadata (single file now)
        assistant.load_model(MODEL_FILE)
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        assistant = None

    @app.route("/")
    def home():
        return render_template("index.html")
    
    # Health check endpoint for Render monitoring
    @app.route("/health")
    def health_check():
        return jsonify({"status": "healthy", "message": "SCI Chatbot is operational"}), 200

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
            app.logger.error(f"Chat error: {str(e)}")  # Add this line
            return jsonify({"response": "Sorry, I encountered an error..."}), 500

    return app

# Application entry point
application = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    application.run(host='0.0.0.0', port=port)