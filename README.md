# Fake News Detection Using NLP and Transformer Models

## 📊 Project Poster

# 📰 Fake News Detection AI

> AI-powered fake news detection system using Machine Learning, Transformer Models, and Large Language Models.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green)
![DistilBERT](https://img.shields.io/badge/DistilBERT-Transformer-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Overview

Fake News Detection AI is a hybrid news classification system that combines traditional Machine Learning, Transformer-based Deep Learning, and Large Language Models (LLMs) to identify whether a news article is **Real** or **Fake**.

Unlike traditional classifiers, this project not only predicts the label but also provides confidence scores and AI-generated explanations.

---

## ✨ Features

- 🤖 DistilBERT Fake News Classifier
- 📊 TF-IDF + Logistic Regression Baseline
- 🧠 Groq LLaMA 3.3 70B Reasoning
- 🔀 Ensemble Prediction System
- ⚡ FastAPI REST API
- 🌐 Web-Based Interface
- 📖 Swagger API Documentation
- 📈 Confidence Score Calculation
- 💬 AI-Powered Prediction Explanations
- 🧪 Automated Testing Structure

---

## 🏗️ System Architecture

```text
News Article
      │
      ▼
┌────────────────────┐
│ Logistic Regression│
└────────────────────┘
      │
      ▼
┌────────────────────┐
│     DistilBERT     │
└────────────────────┘
      │
      ▼
┌────────────────────┐
│  Groq LLaMA 3.3    │
└────────────────────┘
      │
      ▼
┌────────────────────┐
│ Ensemble Decision  │
└────────────────────┘
      │
      ▼
Prediction + Confidence + Explanation
```

---

## 📂 Dataset

The models were trained using the Fake and Real News Dataset from Kaggle.

| Dataset | Records |
|----------|----------|
| Fake News | ~23,000 |
| Real News | ~21,000 |
| Total | ~44,000 |

### Preprocessing

- Text Cleaning
- Lowercasing
- Tokenization
- TF-IDF Vectorization
- Train/Test Split

---

## 📊 Model Performance

| Model | Accuracy |
|---------|---------|
| DistilBERT | 99.4% |
| Logistic Regression | 97.4% |

### Why Ensemble?

Combining multiple models improves:

- Reliability
- Context Understanding
- Prediction Stability
- Explainability

---

## 🛠️ Technology Stack

### Backend

- FastAPI
- Python
- Pydantic

### Machine Learning

- Scikit-Learn
- Logistic Regression
- TF-IDF

### Deep Learning

- PyTorch
- Hugging Face Transformers
- DistilBERT

### LLM Integration

- Groq API
- LLaMA 3.3 70B

---

## 📡 API Endpoints

### Health Check

```http
GET /health
```

### Prediction

```http
POST /predict
```

Example Request:

```json
{
  "text": "Scientists have discovered a new planet capable of supporting life."
}
```

Example Response:

```json
{
  "prediction": "REAL",
  "confidence": 98.6,
  "explanation": "The article contains credible and fact-based language."
}
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Mmbugraaslan/fake-news-detection-ai.git
cd fake-news-detection-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run API

```bash
uvicorn app.main:app --reload
```

---

## 🌐 API Documentation

After starting the server, open:

```text
http://localhost:8000/docs
```

Swagger UI will automatically display all available endpoints.

---

## 📁 Project Structure

```text
fake-news-detection-ai/
│
├── app/
│   ├── api/
│   ├── services/
│   └── main.py
│
├── models/
│
├── scripts/
│
├── tests/
│
├── data/
│
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- Multi-language support
- Social media misinformation detection
- Cloud deployment
- Real-time fact checking
- Advanced ensemble optimization

---

## 👨‍💻 Developer

### Muhammet Mustafa Buğra Aslan

Software Engineering Student  
Machine Learning & Artificial Intelligence Developer

GitHub: https://github.com/Mmbugraaslan

---

## 📜 License

This project is licensed under the MIT License.

- Model training was performed locally
- Results folder may remain empty (training was done in-memory)
- This project demonstrates both traditional and deep learning approaches
