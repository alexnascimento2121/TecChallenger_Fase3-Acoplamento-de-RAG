import datetime as dt
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from assistente_medico.config import AUDIT_JSONL

alog = logging.getLogger("assistente_medico")

PADROES_RISCO = [
    r"\bprescrev[oa]\b",
    r"\badministre\b",
    r"\binicie\s+\w+\s+\d+\s*mg\b",
    r"\btome\s+\d+\s*mg\b",
    r"\bconduta\s+definitiva\b",
    r"\blibero\s+o\s+paciente\b",
    r"\balta\s+hospitalar\s+autorizada\b",
]

DISCLAIMER_OBRIGATORIO = (
    "\n\n⚠️ **AVISO DE SEGURANÇA:** Esta resposta é de apoio clínico e caráter educativo. "
    "NÃO constitui prescrição nem conduta final. "
    "Toda decisão terapêutica deve ser validada por médico habilitado responsável pelo paciente."
)


def detectar_risco(texto: str) -> List[str]:
    return [pat for pat in PADROES_RISCO if re.search(pat, texto, flags=re.IGNORECASE)]


def aplicar_guardrails(resposta: str) -> Dict[str, Any]:
    hits = detectar_risco(resposta)
    texto = resposta.rstrip()

    if DISCLAIMER_OBRIGATORIO.strip() not in texto:
        texto += DISCLAIMER_OBRIGATORIO

    if hits:
        texto += (
            "\n\n🛑 **BLOQUEIO PARCIAL:** Linguagem compatível com prescrição/conduta definitiva "
            "foi detectada. Encaminhar para **validação humana obrigatória** antes de qualquer ação."
        )
        alog.warning(f"GUARDRAIL_HIT | padroes={hits}")
    else:
        alog.info("GUARDRAIL_OK | nenhum padrão de risco crítico")

    return {
        "resposta_segura": texto,
        # Política institucional: SEMPRE exige validação humana
        "precisa_revisao_humana": True,
        "padroes_risco": hits,
    }


def extrair_fontes(textos: List[str]) -> List[str]:
    fontes: List[str] = []
    for t in textos:
        for m in re.findall(r"\[Fonte:[^\]]+\]", t or ""):
            if m not in fontes:
                fontes.append(m)
    return fontes


def registrar_auditoria(evento: Dict[str, Any], audit_path: Path = AUDIT_JSONL) -> None:
    evento = {
        **evento,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "event_id": str(uuid.uuid4()),
    }
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")
    alog.info(f"AUDIT_SAVE | event_id={evento['event_id']} | tipo={evento.get('tipo')}")
