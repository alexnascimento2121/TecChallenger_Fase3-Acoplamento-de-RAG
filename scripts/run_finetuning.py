#!/usr/bin/env python
"""Fase 1: fine-tuning LoRA do assistente médico a partir do MedQuAD."""
import argparse

from assistente_medico.config import ADAPTER_DIR, FINETUNING_LOG
from assistente_medico.data.curation import (
    build_dataset,
    clean_and_anonymize,
    curate,
    load_raw_dataset,
    save_splits,
)
from assistente_medico.finetuning.dataset import tokenize_dataset
from assistente_medico.finetuning.evaluate import ask, compute_perplexity
from assistente_medico.finetuning.model import apply_lora, load_base_model_and_tokenizer
from assistente_medico.finetuning.train import save_adapter, train_model
from assistente_medico.logging_utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA do assistente médico (Fase 1)")
    parser.add_argument("--csv", default="medquad.csv", help="Caminho do dataset MedQuAD")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output-dir", default=str(ADAPTER_DIR))
    args = parser.parse_args()

    log = get_logger("tech_challenge_fase3", FINETUNING_LOG)
    log.info("Início do pipeline de fine-tuning")

    df_raw = load_raw_dataset(args.csv)
    df = clean_and_anonymize(df_raw)
    df_cur = curate(df)
    dataset = build_dataset(df_cur)
    save_splits(dataset)
    print("Shape final curado:", df_cur.shape)

    model, tokenizer = load_base_model_and_tokenizer()
    model = apply_lora(model)
    tokenized = tokenize_dataset(dataset, tokenizer)

    trainer, train_result = train_model(model, tokenizer, tokenized, max_steps=args.max_steps)
    save_adapter(trainer, tokenizer, args.output_dir)
    print("Métricas de treino:", train_result.metrics)

    ppl, _ = compute_perplexity(trainer, tokenized["test"])
    print(f"Perplexidade (modelo fine-tuned) no teste: {ppl:.2f}")

    for ex in dataset["test"].select(range(min(3, len(dataset["test"])))):
        pergunta = ex["messages"][1]["content"]
        print("\nPERGUNTA:", pergunta)
        print("RESPOSTA:", ask(model, tokenizer, pergunta))


if __name__ == "__main__":
    main()
