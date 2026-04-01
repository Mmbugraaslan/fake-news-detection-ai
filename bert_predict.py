from transformers import pipeline

# Fake news demo classifier (pretrained)
classifier = pipeline("text-classification")

texts = [
    "Breaking news: Aliens landed in New York!",
    "The government passed a new law today regarding education."
]

for text in texts:
    result = classifier(text)[0]
    print("\nText:", text)
    print("Prediction:", result["label"], "| Score:", round(result["score"], 4))
