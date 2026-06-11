import re

import pandas as pd

from app.config import LABEL_COLUMN, TEXT_COLUMN


def normalize_text(value: str) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_text_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    working_frame = dataframe.copy()

    if "text" in working_frame.columns:
        working_frame[TEXT_COLUMN] = working_frame["text"].fillna("").map(normalize_text)
    else:
        title_series = working_frame.get("title", pd.Series("", index=working_frame.index)).fillna("")
        subject_series = working_frame.get("subject", pd.Series("", index=working_frame.index)).fillna("")
        combined = title_series.astype(str) + " " + subject_series.astype(str)
        working_frame[TEXT_COLUMN] = combined.map(normalize_text)

    working_frame = working_frame[working_frame[TEXT_COLUMN].str.len() > 0].copy()
    if LABEL_COLUMN in working_frame.columns:
        working_frame[LABEL_COLUMN] = working_frame[LABEL_COLUMN].astype(int)
    return working_frame