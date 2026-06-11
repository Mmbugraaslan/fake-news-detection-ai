from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = BASE_DIR / "archive"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
CLASSICAL_ARTIFACTS_DIR = ARTIFACTS_DIR / "classical"
DISTILBERT_ARTIFACTS_DIR = ARTIFACTS_DIR / "distilbert"

FAKE_DATASET_PATH = ARCHIVE_DIR / "Fake.csv"
TRUE_DATASET_PATH = ARCHIVE_DIR / "True.csv"
COMBINED_DATASET_PATH = PROCESSED_DATA_DIR / "news_dataset.csv"
TRAIN_DATASET_PATH = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_DATASET_PATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_DATASET_PATH = PROCESSED_DATA_DIR / "test.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
LABEL_FAKE = 0
LABEL_TRUE = 1
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
CLASSICAL_MODEL_FILENAME = "model.joblib"
CLASSICAL_METRICS_FILENAME = "metrics.json"
DISTILBERT_METRICS_FILENAME = "metrics.json"
DISTILBERT_MAX_SAMPLES = 5000
DISTILBERT_MAX_LENGTH = 256
DISTILBERT_BATCH_SIZE = 8
DISTILBERT_EPOCHS = 2

KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"
HF_DATASET_FALLBACK = "mrm8488/fake-news"


def ensure_directories() -> None:
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CLASSICAL_ARTIFACTS_DIR,
        DISTILBERT_ARTIFACTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)