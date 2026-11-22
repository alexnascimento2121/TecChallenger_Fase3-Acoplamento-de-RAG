import logging
import re
import textwrap
import uuid
from operator import add
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from assistente_medico.assistant.guardrails import (
    aplicar_guardrails,
    extrair_fontes,
    registrar_auditoria,
)
from assistente_medico.assistant.tools import buscar_protocolo, consultar_paciente, listar_exames

alog = logging.getLogger("assistente_medico")


class AssistenteState(TypedDict, total=False):
    """Estado compartilhado entre os nós do StateGraph."""

    request_id: str
    paciente_id: str
    pergunta_medico: str
    prontuario: str
    exames: str
    protocolos: str
    contexto: str
    resposta_bruta: str
    resposta_final: str
    fontes: List[str]
    alertas: Annotated[List[str], add]
    exames_urgentes_pendentes: bool
    precisa_revisao_humana: bool
    logs_fluxo: Annotated[List[str], add]


def _log_node(state: AssistenteState, nome: str, detalhe: str = "") -> List[str]:
    msg = f"[{nome}] {detalhe}".strip()
    alog.info(f"LANGGRAPH | {msg}")
    return [msg]


def no_receber_solicitacao(state: AssistenteState) -> Dict[str, Any]:
    rid = state.get("request_id") or str(uuid.uuid4())[:8]
    return {
        "request_id": rid,
        "alertas": [],
        "logs_fluxo": _log_node(
            state, "receber_solicitacao", f"rid={rid} paciente={state.get('paciente_id')}"
        ),
    }


def no_consultar_prontuario(state: AssistenteState) -> Dict[str, Any]:
    texto = consultar_paciente.invoke(state["paciente_id"])
    return {"prontuario": texto, "logs_fluxo": _log_node(state, "consultar_prontuario", "EHR ok")}


def no_verificar_exames(state: AssistenteState) -> Dict[str, Any]:
    texto = listar_exames.invoke({"paciente_id": state["paciente_id"], "apenas_pendentes": True})
    urgente = ("prioridade=urgente" in texto) and (
        "status=pendente" in texto or "status=agendado" in texto
    )
    alertas = []
    if urgente:
        alertas.append(
            f"ALERTA: paciente {state['paciente_id']} possui exame(s) URGENTE(S) "
            "pendente(s)/agendado(s)."
        )
    if "Nenhum exame" not in texto and "pendente" in texto:
        alertas.append(
            f"INFO: há exames pendentes para {state['paciente_id']} — "
            "conduzir só após revisão dos resultados quando aplicável."
        )
    return {
        "exames": texto,
        "exames_urgentes_pendentes": urgente,
        "alertas": alertas,
        "logs_fluxo": _log_node(state, "verificar_exames", f"urgente={urgente}"),
    }


def no_buscar_protocolos(state: AssistenteState) -> Dict[str, Any]:
    base_termo = state.get("pergunta_medico", "")
    pront = state.get("prontuario", "")
    candidatos = [
        termo
        for termo in [
            "hipertensão", "glaucoma", "DPOC", "respirat", "torácic", "troponina",
            "pneumonia", "angina", "SCA", "HAS",
        ]
        if re.search(termo, base_termo + " " + pront, flags=re.IGNORECASE)
    ]
    if not candidatos:
        candidatos = [base_termo.split()[0] if base_termo.split() else "protocolo"]

    blocos = [buscar_protocolo.invoke(termo) for termo in candidatos[:2]]
    return {
        "protocolos": "\n\n".join(blocos),
        "logs_fluxo": _log_node(state, "buscar_protocolos", f"termos={candidatos[:2]}"),
    }


def no_montar_contexto(state: AssistenteState) -> Dict[str, Any]:
    contexto = textwrap.dedent(f"""
    === PRONTUÁRIO ===
    {state.get('prontuario', '')}

    === EXAMES ===
    {state.get('exames', '')}

    === PROTOCOLOS INTERNOS ===
    {state.get('protocolos', '')}
    """).strip()
    fontes = extrair_fontes(
        [state.get("prontuario", ""), state.get("exames", ""), state.get("protocolos", "")]
    )
    return {
        "contexto": contexto,
        "fontes": fontes,
        "logs_fluxo": _log_node(state, "montar_contexto", f"fontes={len(fontes)}"),
    }


def build_graph(llm_bridge):
    """Monta e compila o StateGraph, injetando a LLM customizada nos nós que a utilizam."""

    def no_gerar_sugestao(state: AssistenteState) -> Dict[str, Any]:
        pergunta = state["pergunta_medico"]
        pergunta_enriquecida = (
            f"{pergunta}\n\n"
            "Com base no contexto do paciente e nos protocolos, sugira próximos passos "
            "de apoio à decisão (NÃO prescreva doses nem feche conduta). "
            "Ao final, liste as fontes utilizadas."
        )
        resposta = llm_bridge.gerar_resposta(
            pergunta=pergunta_enriquecida,
            contexto_extra=state.get("contexto", ""),
            max_new_tokens=280,
            temperature=0.35,
        )
        return {"resposta_bruta": resposta, "logs_fluxo": _log_node(state, "gerar_sugestao", "LLM ok")}

    def no_validar_seguranca(state: AssistenteState) -> Dict[str, Any]:
        gr = aplicar_guardrails(state.get("resposta_bruta", ""))
        fontes = state.get("fontes") or []
        bloco_fontes = (
            "\n".join(f"- {f}" for f in fontes)
            if fontes else "- (sem marcador de fonte estruturado)"
        )
        resposta_final = (
            gr["resposta_segura"] + "\n\n### Fontes utilizadas (explainability)\n" + bloco_fontes
        )
        return {
            "resposta_final": resposta_final,
            "precisa_revisao_humana": gr["precisa_revisao_humana"],
            "alertas": (
                ["GUARDRAIL: padrões de risco detectados — revisão humana obrigatória"]
                if gr["padroes_risco"] else []
            ),
            "logs_fluxo": _log_node(
                state, "validar_seguranca", f"risco={bool(gr['padroes_risco'])}"
            ),
        }

    def no_emitir_alerta(state: AssistenteState) -> Dict[str, Any]:
        alertas = list(state.get("alertas") or [])
        if state.get("exames_urgentes_pendentes"):
            alertas.append(
                "📣 Equipe médica notificada (simulado): há exames urgentes pendentes — "
                "priorizar coleta/resultado antes de conduta definitiva."
            )
        if state.get("precisa_revisao_humana"):
            alertas.append(
                "📣 Fila de validação humana: resposta do assistente aguardando "
                "revisão do médico responsável."
            )

        registrar_auditoria({
            "tipo": "fluxo_assistente",
            "request_id": state.get("request_id"),
            "paciente_id": state.get("paciente_id"),
            "pergunta": state.get("pergunta_medico"),
            "alertas": alertas,
            "fontes": state.get("fontes"),
            "precisa_revisao_humana": state.get("precisa_revisao_humana"),
            "resposta_preview": (state.get("resposta_final") or "")[:500],
            "modo_llm": llm_bridge.modo,
        })

        return {
            "alertas": alertas,
            "logs_fluxo": _log_node(state, "emitir_alerta", f"n_alertas={len(alertas)}"),
        }

    graph = StateGraph(AssistenteState)
    graph.add_node("receber_solicitacao", no_receber_solicitacao)
    graph.add_node("consultar_prontuario", no_consultar_prontuario)
    graph.add_node("verificar_exames", no_verificar_exames)
    graph.add_node("buscar_protocolos", no_buscar_protocolos)
    graph.add_node("montar_contexto", no_montar_contexto)
    graph.add_node("gerar_sugestao", no_gerar_sugestao)
    graph.add_node("validar_seguranca", no_validar_seguranca)
    graph.add_node("emitir_alerta", no_emitir_alerta)

    graph.add_edge(START, "receber_solicitacao")
    graph.add_edge("receber_solicitacao", "consultar_prontuario")
    graph.add_edge("consultar_prontuario", "verificar_exames")
    graph.add_edge("verificar_exames", "buscar_protocolos")
    graph.add_edge("buscar_protocolos", "montar_contexto")
    graph.add_edge("montar_contexto", "gerar_sugestao")
    graph.add_edge("gerar_sugestao", "validar_seguranca")
    graph.add_edge("validar_seguranca", "emitir_alerta")
    graph.add_edge("emitir_alerta", END)

    return graph.compile()
