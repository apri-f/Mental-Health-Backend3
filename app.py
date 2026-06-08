from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# ======================
# LOAD MODEL ML
# ======================
model = joblib.load("depression_model.pkl")
encoder = joblib.load("severity_encoder.pkl")


# ======================
# PREDICT PHQ-9
# ======================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    answers = [[
        data["Q1"],
        data["Q2"],
        data["Q3"],
        data["Q4"],
        data["Q5"],
        data["Q6"],
        data["Q7"],
        data["Q8"],
        data["Q9"]
    ]]

    pred = model.predict(answers)[0]

    severity = encoder.inverse_transform([pred])[0]

    return jsonify({
        "severity": severity
    })

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

chat_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ======================
# AI CHAT ENDPOINT
# ======================
@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.json

        user_message = data["message"]
        severity = data["severity"]

        prompt = f"""
Kamu adalah asisten kesehatan mental yang suportif.

Hasil asesmen pengguna:
{severity}

Pesan pengguna:
{user_message}

Jawab dalam Bahasa Indonesia.
"""

        response = chat_model.generate_content(
            prompt
        )

        return jsonify({
            "reply": response.text
        })

    except Exception as e:

        print("CHAT ERROR:", e)

        return jsonify({
            "reply":
            "Maaf, terjadi kesalahan saat menghubungi AI."
        })

@app.route("/")
def home():
    return "Mental Health API Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)