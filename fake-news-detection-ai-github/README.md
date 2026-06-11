# Fake News Detection AI

Sahte haber tespiti icin gelistirilmis, yerel DistilBERT modeli ve Groq LLaMA 3.3 70B ensemble yontemiyle calisan bir FastAPI uygulamasi.

## Ozellikler

- **Yerel AI Modeli**: Fine-tuned DistilBERT (Kaggle 44k veri setiyle egitilmis)
- **LLM Analizi**: Groq LLaMA 3.3 70B ile derinlemesine haber analizi
- **Ensemble Sistem**: Iki modelin sonuclarini birlestirerek daha guvenilir tahminler
- **REST API**: FastAPI ile hizli ve olceklenebilir servis
- **Swagger Docs**: Otomatik API dokumantasyonu

## Proje Yapisi

```
fake-news-detection-ai/
├── app/
│   ├── api.py                 # FastAPI endpoint'leri
│   ├── schemas.py             # Pydantic request/response modelleri
│   ├── config.py              # Proje sabitleri ve yapilandirmalari
│   └── services/
│       ├── predictor.py       # Ensemble tahmin servisi (DistilBERT + Groq)
│       └── model_registry.py  # Model kayit ve yonetim
├── models/
│   ├── bert.py                # DistilBERT model sinifi
│   ├── classical.py           # TF-IDF + LogisticRegression modeli
│   └── preprocessing.py       # Metin on isleme araclari
├── scripts/
│   ├── prepare_data.py        # Hugging Face veri hazirlama
│   ├── prepare_kaggle_data.py # Kaggle 44k veri hazirlama
│   ├── train_bert.py          # DistilBERT fine-tuning
│   └── train_classical.py     # Klasik model egitimi
├── tests/
│   ├── test_api.py            # API endpoint testleri
│   └── test_predictor.py      # Predictor servis testleri
├── main.py                    # Uvicorn giris noktasi
├── requirements.txt           # Python bagimliliklari
└── README.md                  # Bu dosya
```

## Kurulum

### 1. Depoyu Klonla

```bash
git clone https://github.com/Mmbugraaslan/fake-news-detection-ai.git
cd fake-news-detection-ai
```

### 2. Sanal Ortam Olustur (Tavsiye Edilir)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Bagimliliklari Yukle

```bash
pip install -r requirements.txt
```

### 4. Groq API Anahtari Ekle

`groq_config.txt` dosyasi olustur ve icine API anahtarini yaz:

```
GROQ_API_KEY=senin_api_anahtarin_buraya
```

> Groq API anahtari almak icin: [console.groq.com](https://console.groq.com)

### 5. Veri Setini Indir (Opsiyonel - Model Egitimi Icin)

Kaggle API ile:
```bash
python -m kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p archive
```

Veya manuel olarak [Kaggle'dan](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) indirip `archive/` klasorune yerlestir.

### 6. Modeli Egit (Eger Onceden Egitilmis Model Yoksa)

```bash
# Veriyi hazirla
python scripts/prepare_kaggle_data.py

# Klasik modeli egit
python scripts/train_classical.py

# DistilBERT'i egit (CPU'da ~2-3 saat)
python scripts/train_bert.py
```

> Not: Onceden egitilmis model dosyalari (`data/artifacts/`) repoda bulunuyorsa bu adimi atlayabilirsin.

## Kullanim

### API'yi Baslat

```bash
python main.py
```

API `http://localhost:8000` adresinde calisacak.

### Swagger Dokumantasyonu

Tarayicinda ac: `http://localhost:8000/docs`

### API Endpoint'leri

#### POST /predict

Haber metnini analiz et.

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

Sistem durumunu kontrol et.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "distilbert"
}
```

### cURL ile Test

```bash
# Health check
curl http://localhost:8000/health

# Tahmin
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here..."}'
```

## Model Performansi

| Model | Veri Seti | Dogruluk | F1 Skor |
|---|---|---|---|
| DistilBERT | Kaggle 44k | %99.4 | 0.994 |
| TF-IDF + LogisticRegression | Kaggle 44k | %97.4 | 0.974 |

## Teknolojiler

- **Python 3.10+**
- **FastAPI** — Web framework
- **Transformers (Hugging Face)** — DistilBERT modeli
- **PyTorch** — Derin ogrenme motoru
- **Scikit-learn** — Klasik ML pipeline
- **Groq API** — LLaMA 3.3 70B LLM erisimi
- **Pandas** — Veri isleme

## Katkida Bulunma

1. Fork yap
2. Feature branch olustur (`git checkout -b feature/yeni-ozellik`)
3. Degisikliklerini commit et (`git commit -am 'Yeni ozellik eklendi'`)
4. Branch'i push et (`git push origin feature/yeni-ozellik`)
5. Pull Request ac

## Lisans

Bu proje MIT Lisansi altinda lisanslanmistir.

## Iletisim

Sorulariniz veya onerileriniz icin issue acabilirsiniz.
