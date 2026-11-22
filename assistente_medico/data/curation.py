from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict

from assistente_medico.config import DATA_DIR, SEED, SYSTEM_PROMPT
from assistente_medico.data.preprocessing import anonymize, clean_text


def load_raw_dataset(csv_path: str = "medquad.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)


def clean_and_anonymize(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.dropna(subset=["question", "answer"]).copy()
    df["question"] = df["question"].map(clean_text)
    df["answer"] = df["answer"].map(clean_text)
    df["focus_area"] = df["focus_area"].fillna("Não especificado").map(clean_text)
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)

    for col in ["question", "answer"]:
        df[col] = df[col].map(anonymize)

    return df


def curate(
    df: pd.DataFrame,
    min_chars: int = 40,
    max_chars: int = 2500,
    sample_size: Optional[int] = 6000,
) -> pd.DataFrame:
    mask = df["answer"].str.len().between(min_chars, max_chars) & df["question"].str.len().ge(10)
    df_cur = df[mask].reset_index(drop=True)

    if sample_size and len(df_cur) > sample_size:
        df_cur = (
            df_cur.groupby("source", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, int(sample_size * len(g) / len(df_cur)))),
                random_state=SEED))
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True)
        )
    return df_cur


def build_example(row: pd.Series) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"] +
                f"\n\n[Fonte: {row['source']} | Tema: {row['focus_area']}]"},
        ]
    }


def build_dataset(df_cur: pd.DataFrame) -> DatasetDict:
    records = df_cur.apply(build_example, axis=1).tolist()
    ds_full = Dataset.from_list(records).shuffle(seed=SEED)
    n = len(ds_full)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    return DatasetDict({
        "train": ds_full.select(range(0, n_train)),
        "validation": ds_full.select(range(n_train, n_train + n_val)),
        "test": ds_full.select(range(n_train + n_val, n)),
    })


def save_splits(dataset: DatasetDict, out_dir: Path = DATA_DIR) -> None:
    out_dir.mkdir(exist_ok=True)
    for split, d in dataset.items():
        d.to_json(str(out_dir / f"medquad_{split}.jsonl"), orient="records", lines=True, force_ascii=False)
