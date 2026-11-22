import math

import torch

from assistente_medico.config import SYSTEM_PROMPT


def compute_perplexity(trainer, dataset):
    metrics = trainer.evaluate(eval_dataset=dataset)
    return math.exp(metrics["eval_loss"]), metrics


def ask(model, tokenizer, question: str, max_new_tokens: int = 200) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
