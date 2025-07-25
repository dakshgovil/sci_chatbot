# SCI Chatbot – Local Setup Guide

This utility allows you to run the SCI Virtual Assistant chatbot locally on Windows desktop and provides step-by-step instructions to help the IT department run the SCI Chatbot locally on any machine.

---

## 📁 Project Structure

```
sci_chatbot/
├── app.py                  # Flask backend entry point
├── chatbot_model.pth       # Trained PyTorch model
├── intentg.json            # Intent definitions
├── newchat.py              # Chatbot logic (NLP + model)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Optional config (used in deployment)
├── wsgi.py                 # WSGI entry point for servers like Gunicorn
├── static/                 # Static assets (images/icons)
│   ├── sci_logo.png
│   └── chat_icon.png
├── templates/
│   └── index.html          # Frontend UI
```

---

## 💻 Prerequisites

Make sure the following are installed on the system:

- Python 3.10+
- pip (Python package manager)

---

## 🔧 Setup Instructions

### 1. Clone or Copy the Project Folder

Ensure all files listed above are present.

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# OR
source venv/bin/activate   # On Linux/Mac
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Local Server

```bash
python app.py
```

### 5. Open Chatbot in Browser

Visit:

```
http://localhost:5000
```

You’ll see the SCI chatbot interface.

---

## 🧠 Model Notes

- The `chatbot_model.pth` file is pre-trained.
- No training is needed unless `intentg.json` is modified.
- If intents change, run `newchat.py` to retrain:

```bash
python newchat.py
```

It will parse intents, train the model, and overwrite `chatbot_model.pth`.

---

## 🚨 Common Issues

|                                   |                                                                             |
| --------------------------------- | --------------------------------------------------------------------------- |
| Server runs but page doesn't load | Check that you are visiting http\://localhost:5000                          |
|                                   |                                                                             |
| Flask errors                      | Ensure all dependencies are installed via `pip install -r requirements.txt` |

---

## 📞 Support

For further help, contact the developer DAKSH GOVIL ([dakshgovil@gmail.com](mailto\:dakshgovil@gmail.com)) or refer to the README instructions.

---

**SCI Chatbot – Built for assisting with Website Navigation and Shipping Corporation of India services**

