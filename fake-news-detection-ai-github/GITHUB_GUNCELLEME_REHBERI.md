# GitHub Guncelleme Rehberi

Bu dosya, https://github.com/Mmbugraaslan/fake-news-detection-ai reposunu nasil guncelleyeceginizi adim adim anlatir.

---

## 1. SILINECEK DOSYALAR

Asagidaki eski dosyalari sil:

```
bert_model.py          -> Eski demo dosyasi (yerine models/bert.py var)
bert_predict.py        -> Eski demo dosyasi (yerine app/services/predictor.py var)
bert_train.py          -> Eski egitim dosyasi (yerine scripts/train_bert.py var)
main.py                -> Eski main.py (yerine yeni main.py var)
README.md              -> Eski README (yerine yeni README.md var)
requirements.txt       -> Eski requirements (yerine yeni requirements.txt var)
```

---

## 2. EKLENECEK YENI DOSYALAR

Asagidaki dosyalari yeni ekle (hepsi C:\Users\aslan\projects\fake-news-detection-ai-github\ klasorunde hazir):

### Kok Dizin
```
.gitignore              -> Git icin gereksiz dosyalari hariç tutar
main.py                 -> Yeni FastAPI baslatma dosyasi
README.md               -> Yeni tam kurulum kılavuzu
requirements.txt        -> Yeni bagimliliklar (groq dahil)
```

### app/ Klasoru (YENI)
```
app/__init__.py
app/api.py              -> FastAPI endpoint'leri + web arayuzu
app/config.py           -> Proje sabitleri
app/schemas.py          -> Request/Response modelleri
app/services/__init__.py
app/services/predictor.py      -> Ensemble tahmin (DistilBERT + Groq LLaMA)
app/services/model_registry.py -> Model yonetimi
```

### models/ Klasoru (YENI)
```
models/__init__.py
models/bert.py          -> DistilBERT model sinifi
models/classical.py     -> TF-IDF + LogisticRegression
models/preprocessing.py -> Metin on isleme
```

### scripts/ Klasoru (YENI)
```
scripts/prepare_data.py        -> Hugging Face veri hazirlama
scripts/prepare_kaggle_data.py -> Kaggle 44k veri hazirlama
scripts/train_bert.py          -> DistilBERT fine-tuning
scripts/train_classical.py     -> Klasik model egitimi
```

### tests/ Klasoru (YENI)
```
tests/test_api.py       -> API endpoint testleri
tests/test_predictor.py -> Predictor servis testleri
```

### Kalacak Eski Dosyalar
```
LICENSE                 -> MIT Lisansi (korunacak)
archive/                -> Veri seti klasoru (korunacak)
```

---

## 3. ADIM ADIM GITHUB'A YUKLEME

### Adim 1: Eski dosyalari sil
```bash
cd fake-news-detection-ai
git rm bert_model.py bert_predict.py bert_train.py main.py README.md requirements.txt
git commit -m "chore: remove old demo files"
```

### Adim 2: Yeni dosyalari kopyala
Yukaridaki tum yeni dosyalari proje klasorune kopyala.

### Adim 3: Git'e ekle
```bash
git add .
git commit -m "feat: complete rewrite with FastAPI, DistilBERT + Groq LLaMA ensemble"
```

### Adim 4: Push et
```bash
git push origin main
```

---

## 4. NOTLAR

- **Model dosyalari** (data/artifacts/) cok buyuk oldugu icin .gitignore'da hariç tutuldu.
- Kullanicilar kendi modelini `scripts/train_bert.py` ile egitmeli.
- **groq_config.txt** dosyasini olusturmayi unutma (API key icin).
- **archive/** klasorundeki veri seti korunacak.

---

## 5. HAZIR DOSYALARIN YERI

Tum yeni dosyalar burada hazir:
```
C:\Users\aslan\projects\fake-news-detection-ai-github\
```

Bu klasordeki her seyi dogrudan GitHub'a yukleyebilirsin.
