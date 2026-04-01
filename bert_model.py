from transformers import pipeline

# Basit Fake News test modeli (pretrained)
classifier = pipeline("text-classification")

# Test metni
text = "Breaking news: Aliens landed in New York!"

# Tahmin
result = classifier(text)

print("Sonuç:", result)
