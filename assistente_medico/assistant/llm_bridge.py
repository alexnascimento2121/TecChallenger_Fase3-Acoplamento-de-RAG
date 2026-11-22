import logging

import torch
from langchain_core.runnables import RunnableLambda
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from assistente_medico.config import ADAPTER_DIR, BASE_MODEL, DEVICE, SYSTEM_PROMPT

alog = logging.getLogger("assistente_medico")


def carregar_llm_customizada():
    alog.info(f"Carregando tokenizer/modelo: {BASE_MODEL} | device={DEVICE}")

    tok = AutoTokenizer.from_pretrained(str(ADAPTER_DIR) if ADAPTER_DIR.exists() else BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = None
    if DEVICE == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto" if DEVICE == "cuda" else None,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    )

    if ADAPTER_DIR.exists():
        mdl = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        alog.info(f"Adapter LoRA carregado de {ADAPTER_DIR}")
        modo = "fine-tuned (LoRA)"
    else:
        mdl = base
        alog.warning("Adapter LoRA NÃO encontrado — usando modelo base (demo do fluxo)")
        modo = "base (sem fine-tuning)"

    mdl.eval()
    return tok, mdl, modo


class LLMBridge:
    """Ponte entre o modelo Hugging Face (base ou fine-tuned) e o LangChain."""

    def __init__(self):
        self.tokenizer, self.model, self.modo = carregar_llm_customizada()
        self.runnable = RunnableLambda(
            lambda x: self.gerar_resposta(
                pergunta=x["pergunta"] if isinstance(x, dict) else str(x),
                contexto_extra=x.get("contexto", "") if isinstance(x, dict) else "",
            )
        )

    def gerar_resposta(
        self,
        pergunta: str,
        contexto_extra: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.4,
    ) -> str:
        system = SYSTEM_PROMPT
        if contexto_extra.strip():
            system = (
                SYSTEM_PROMPT
                + "\n\n### Contexto atualizado do paciente / protocolos internos\n"
                + contexto_extra.strip()
                + "\n\nUse o contexto acima para personalizar a resposta. "
                  "Se houver conflito, priorize o prontuário e indique a fonte."
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": pergunta},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        resposta = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        alog.info(
            "LLM_GENERATE | tokens_in=%s | preview=%s",
            inputs["input_ids"].shape[1],
            resposta[:180].replace("\n", " "),
        )
        return resposta
