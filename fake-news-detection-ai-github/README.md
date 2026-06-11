# Fake News Detection AI

A FastAPI application for detecting fake news using a local DistilBERT model and Groq LLaMA 3.3 70B ensemble method.

## Features

- **Local AI Model**: Fine-tuned DistilBERT (trained on Kaggle 44k dataset)
- **LLM Analysis**: Deep news analysis with Groq LLaMA 3.3 70B
- **Ensemble System**: Combines results from two models for more reliable predictions
- **REST API**: Fast, scalable service with FastAPI
- **Swagger Docs**: Automatic API documentation
- **Web Interface**: Built-in UI for testing predictions

## Project Structure

```
fake-news-detection-ai/
├── app/
│   ├── api.py                 # FastAPI endpoints + web UI
│   ├── schemas.py             # Pydantic request/response models
│   ├── config.py              # Project constants and configurations
│   └── services/
│       ├── predictor.py       # Ensemble prediction service (DistilBERT + Groq)
│       └── model_registry.py  # Model registration and management
├── models/
│   ├── bert.py                # DistilBERT model class
│   ├── classical.py           # TF-IDF + LogisticRegression model
│   └── preprocessing.py       # Text preprocessing utilities
├── scripts/
│   ├── prepare_data.py        # Hugging Face data preparation
│   ├── prepare_kaggle_data.py # Kaggle 44k data preparation
│   ├── train_bert.py          # DistilBERT fine-tuning
│   └── train_classical.py     # Classical model training
├── tests/
│   ├── test_api.py            # API endpoint tests
│   └── test_predictor.py      # Predictor service tests
├── main.py                    # Uvicorn entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mmbugraaslan/fake-news-detection-ai.git
cd fake-news-detection-ai
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Groq API Key

Create a `groq_config.txt` file and add your API key:

```
GROQ_API_KEY=your_api_key_here
```

> Get a Groq API key at: [console.groq.com](https://console.groq.com)

### 5. Download Dataset (Optional - For Model Training)

Using Kaggle API:
```bash
python -m kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p archive
```

Or manually download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place in `archive/` folder.

### 6. Train Models (If Pre-trained Models Not Available)

```bash
# Prepare data
python scripts/prepare_kaggle_data.py

# Train classical model
python scripts/train_classical.py

# Train DistilBERT (~2-3 hours on CPU)
python scripts/train_bert.py
```

> Note: If pre-trained model files exist in `data/artifacts/`, you can skip this step.

## Usage

### Start the API

```bash
python main.py
```

The API will run at `http://localhost:8000`.

### Swagger Documentation

Open in browser: `http://localhost:8000/docs`

### API Endpoints

#### POST /predict

Analyze a news article.

**Request:**
```json
{
  "text": "The Federal Reserve announced interest rate changes today after their monthly meeting in Washington DC."
}
```

**Response:**
```json
{
  "label": "real",
  "label_tr": "gercek",
  "confidence": 0.9,
  "local_model": {
    "label": "real",
    "confidence": 0.6247,
    "model": "distilbert"
  },
  "llm_analysis": {
    "label": "real",
    "confidence": 0.8,
    "explanation": "official announcements from the Federal Reserve are common and publicly available"
  },
  "agreement": "both_agree",
  "model": "ensemble",
  "fallback_used": false
}
```

#### GET /health

Check system status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "distilbert"
}
```

### Test with cURL

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here..."}'
```

## Model Performance

| Model | Dataset | Accuracy | F1 Score |
|---|---|---|---|
| DistilBERT | Kaggle 44k | 99.4% | 0.994 |
| TF-IDF + LogisticRegression | Kaggle 44k | 97.4% | 0.974 |

## Technologies

- **Python 3.10+**
- **FastAPI** — Web framework
- **Transformers (Hugging Face)** — DistilBERT model
- **PyTorch** — Deep learning engine
- **Scikit-learn** — Classical ML pipeline
- **Groq API** — LLaMA 3.3 70B LLM access
- **Pandas** — Data processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please open an issue.
