import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset yükleme
fake = pd.read_csv("archive/Fake.csv")
real = pd.read_csv("archive/True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# TEXT + LABEL
X = data["text"]
y = data["label"]

# Train / Test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text → sayı (vectorization)
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Tahmin
y_pred = model.predict(X_test_vec)

# Sonuç
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
