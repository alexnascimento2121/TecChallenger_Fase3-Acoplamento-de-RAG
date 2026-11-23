import random
from pathlib import Path

import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN = 1024

ADAPTER_DIR = Path("modelo_medquad_lora_final")
DATA_DIR = Path("data_curada")
DB_PATH = Path("hospital_prontuarios.db")
AUDIT_JSONL = Path("auditoria_respostas.jsonl")
ETAPA3_AUDIT_JSONL = Path("auditoria_etapa3.jsonl")

FINETUNING_LOG = "pipeline_finetuning.log"
ASSISTENTE_LOG = "assistente_medico.log"

SYSTEM_PROMPT = (
    "Você é um assistente médico virtual do hospital, treinado para apoiar médicos com "
    "informações baseadas em protocolos e literatura clínica. "
    "Responda de forma objetiva e cite que a informação é de caráter educativo. "
    "NUNCA prescreva medicação ou defina conduta final sem indicar que é necessária "
    "validação humana por um profissional habilitado."
)
