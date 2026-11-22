from datasets import DatasetDict

from assistente_medico.config import MAX_LEN


def tokenize_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    def tokenize_fn(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(text, truncation=True, max_length=MAX_LEN, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize_fn, remove_columns=["messages"])
    tokenized.set_format(type="torch")
    return tokenized
