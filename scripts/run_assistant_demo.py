#!/usr/bin/env python
"""Fase 2: demonstração ponta a ponta do assistente médico (LangChain + LangGraph)."""
from assistente_medico.assistant.database import criar_banco_hospital
from assistente_medico.assistant.graph import build_graph
from assistente_medico.assistant.llm_bridge import LLMBridge
from assistente_medico.assistant.pipeline import perguntar_assistente, pipeline_consulta_geral
from assistente_medico.config import ASSISTENTE_LOG
from assistente_medico.logging_utils import get_logger

CENARIOS = [
    (
        "PAC-001",
        "Paciente com HAS e queixa de visão turva. Quais próximos passos segundo "
        "nossos protocolos, considerando exames pendentes?",
    ),
    (
        "PAC-003",
        "Tenho um paciente com dor torácica e troponina ainda pendente. O que o "
        "protocolo interno recomenda enquanto aguardamos o resultado?",
    ),
    (
        "PAC-004",
        "Paciente com suspeita de pneumonia e CURB-65 = 2. Há alergia a sulfa. "
        "Quais cuidados e exames ainda precisamos antes de definir antibioticoterapia?",
    ),
]


def main() -> None:
    get_logger("assistente_medico", ASSISTENTE_LOG, also_console=True)

    criar_banco_hospital()
    llm_bridge = LLMBridge()
    assistente_app = build_graph(llm_bridge)

    for paciente_id, pergunta in CENARIOS:
        perguntar_assistente(assistente_app, llm_bridge, paciente_id, pergunta)

    print(pipeline_consulta_geral(
        llm_bridge,
        "Quais são os sintomas clássicos de glaucoma?",
        termo_protocolo="glaucoma",
    ))


if __name__ == "__main__":
    main()
