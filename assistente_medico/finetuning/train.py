import logging

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from assistente_medico.config import DEVICE, SEED

log = logging.getLogger("tech_challenge_fase3")


def train_model(model, tokenizer, tokenized, output_dir: str = "checkpoints_medquad_lora", max_steps: int = 10):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        max_steps=max_steps,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=(DEVICE == "cuda"),
        report_to="none",
        seed=SEED,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    log.info("Início do treinamento LoRA")
    train_result = trainer.train()
    log.info(f"Treinamento concluído. Métricas: {train_result.metrics}")
    return trainer, train_result


def save_adapter(trainer, tokenizer, output_dir) -> None:
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info(f"Adapter salvo em {output_dir}")
