import logging
from typing import Any, Dict, Optional

from assistente_medico.assistant.guardrails import (
    aplicar_guardrails,
    extrair_fontes,
    registrar_auditoria,
)
from assistente_medico.assistant.tools import buscar_protocolo

alog = logging.getLogger("assistente_medico")


def perguntar_assistente(
    assistente_app,
    llm_bridge,
    paciente_id: str,
    pergunta_medico: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Executa o fluxo completo LangGraph para um paciente + pergunta clínica."""
    alog.info(f"ASK | paciente={paciente_id} | pergunta={pergunta_medico[:120]}")
    estado_inicial = {
        "paciente_id": paciente_id,
        "pergunta_medico": pergunta_medico,
        "alertas": [],
        "logs_fluxo": [],
    }
    resultado = assistente_app.invoke(estado_inicial)

    if verbose:
        print("=" * 100)
        print(f"Paciente: {paciente_id} | request_id: {resultado.get('request_id')}")
        print(f"Pergunta: {pergunta_medico}")
        print("-" * 100)
        print("RESPOSTA FINAL:\n")
        print(resultado.get("resposta_final", ""))
        print("-" * 100)
        print("ALERTAS:")
        for a in resultado.get("alertas") or []:
            print(" •", a)
        print("-" * 100)
        print("FONTES:", resultado.get("fontes"))
        print("Revisão humana obrigatória?", resultado.get("precisa_revisao_humana"))
        print("Trilha do fluxo:")
        for step in resultado.get("logs_fluxo") or []:
            print("  →", step)
        print("=" * 100)

    return resultado


def pipeline_consulta_geral(
    llm_bridge,
    pergunta: str,
    termo_protocolo: Optional[str] = None,
) -> str:
    """Pipeline LangChain curto: protocolo (opcional) → LLM → guardrail → auditoria."""
    contexto = ""
    fontes = []
    if termo_protocolo:
        proto = buscar_protocolo.invoke(termo_protocolo)
        contexto = proto
        fontes = extrair_fontes([proto])

    bruta = llm_bridge.runnable.invoke({"pergunta": pergunta, "contexto": contexto})
    segura = aplicar_guardrails(bruta)
    final = segura["resposta_segura"]
    if fontes:
        final += "\n\n### Fontes utilizadas (explainability)\n" + "\n".join(
            f"- {f}" for f in fontes
        )

    registrar_auditoria({
        "tipo": "consulta_geral",
        "pergunta": pergunta,
        "termo_protocolo": termo_protocolo,
        "fontes": fontes,
        "precisa_revisao_humana": segura["precisa_revisao_humana"],
        "resposta_preview": final[:500],
        "modo_llm": llm_bridge.modo,
    })
    return final
