# GitHub Guncelleme Rehberi - Fake News Detection AI

Repo: https://github.com/Mmbugraaslan/fake-news-detection-ai

---

## 1. SILINECEK DOSYALAR (6 dosya)

Asagidaki eski dosyalari GitHub'dan sil:

```
bert_model.py
bert_predict.py
bert_train.py
main.py
README.md
requirements.txt
```

---

## 2. EKLENECEK YENI DOSYALAR (26 dosya)

Tum bu dosyalar su klasorde hazir:
C:\Users\aslan\projects\fake-news-detection-ai-github\

### Kok Dizin (4 dosya)
```
.gitignore
main.py
README.md
requirements.txt
```

### app/ Klasoru (7 dosya)
```
app/__init__.py
app/api.py
app/config.py
app/schemas.py
app/services/__init__.py
app/services/predictor.py
app/services/model_registry.py
```

### models/ Klasoru (4 dosya)
```
models/__init__.py
models/bert.py
models/classical.py
models/preprocessing.py
```

### scripts/ Klasoru (4 dosya)
```
scripts/prepare_data.py
scripts/prepare_kaggle_data.py
scripts/train_bert.py
scripts/train_classical.py
```

### tests/ Klasoru (2 dosya)
```
tests/test_api.py
tests/test_predictor.py
```

---

## 3. KALACAK ESKI DOSYALAR (2 dosya)

Bunlar degismeyecek:
```
LICENSE
archive/  (veri seti klasoru)
```

---

## 4. ADIM ADIM YUKLEME

```bash
# 1. Repoya git
cd fake-news-detection-ai

# 2. Eski dosyalari sil
git rm bert_model.py bert_predict.py bert_train.py main.py README.md requirements.txt

# 3. Yeni dosyalari kopyala (C:\Users\aslan\projects\fake-news-detection-ai-github\ icerigini buraya)
#    - .gitignore
#    - main.py
#    - README.md
#    - requirements.txt
#    - app/
#    - models/
#    - scripts/
#    - tests/

# 4. Git'e ekle
git add .

# 5. Commit et
git commit -m "feat: Fake News Detection API - DistilBERT + Groq LLaMA ensemble"

# 6. Push et
git push origin main
```

---

## 5. ONEMLI NOTLAR

- API anahtari (groq_config.txt) GIT'E GITMEZ (.gitignore'da var)
- Model dosyalari (data/artifacts/) GIT'E GITMEZ (.gitignore'da var)
- Kullanici kendi groq_config.txt dosyasini olusturacak
- Kullanici kendi modelini egitmeli (veya sen model dosyalarini ayrica ver)
