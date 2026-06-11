import pandas as pd
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

# Verileri yükle
fake = pd.read_csv(r'C:\Users\aslan\projects\fake-news-detection-ai\archive\Fake.csv')
true = pd.read_csv(r'C:\Users\aslan\projects\fake-news-detection-ai\archive\True.csv')

# Etiket ekle
fake['label'] = 0
true['label'] = 1

# Birleştir
df = pd.concat([fake, true], ignore_index=True)

# text kolonu: title + text
df['text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).str.strip()
df = df[['text', 'label']]
df = df[df['text'].str.len() > 20].drop_duplicates(subset=['text']).reset_index(drop=True)

print(f'Toplam: {len(df)} haber')
print(df['label'].value_counts())

# Split
train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

# Kaydet
base = Path(r'C:\Users\aslan\projects\fake-news-detection-ai\data\processed')
train.to_csv(base / 'train.csv', index=False)
val.to_csv(base / 'val.csv', index=False)
test.to_csv(base / 'test.csv', index=False)

summary = {
    'dataset': 'Kaggle fake-and-real-news-dataset',
    'total_rows': int(len(df)),
    'train_rows': int(len(train)),
    'val_rows': int(len(val)),
    'test_rows': int(len(test)),
    'label_distribution': {str(k): int(v) for k, v in df['label'].value_counts().to_dict().items()}
}
with open(base / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
