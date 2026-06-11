from pathlib import Path
import json

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"
MAX_ROWS = 5000
RANDOM_STATE = 42


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_fake_news_dataframe() -> pd.DataFrame:
    dataset = load_dataset("GonzaloA/fake_news", split="train")
    dataframe = dataset.to_pandas()

    text_column = "text" if "text" in dataframe.columns else dataframe.columns[0]
    label_column = "label" if "label" in dataframe.columns else dataframe.columns[-1]

    dataframe = dataframe[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "label"}
    )
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe["label"] = dataframe["label"].astype(int)
    dataframe = dataframe[dataframe["text"].str.len() > 20].drop_duplicates(subset=["text"])

    if len(dataframe) > MAX_ROWS:
        samples = []
        for _label, group in dataframe.groupby("label"):
            n = min(len(group), MAX_ROWS // 2)
            samples.append(group.sample(n, random_state=RANDOM_STATE))
        dataframe = pd.concat(samples).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    return dataframe.reset_index(drop=True)


def split_and_save(dataframe: pd.DataFrame) -> dict:
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataframe["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "dataset": "GonzaloA/fake_news",
        "total_rows": int(len(dataframe)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "label_distribution": {
            str(key): int(value) for key, value in dataframe["label"].value_counts().to_dict().items()
        },
    }

    (PROCESSED_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ensure_directories()
    dataframe = load_fake_news_dataframe()
    summary = split_and_save(dataframe)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()