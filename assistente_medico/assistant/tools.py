import logging
import textwrap

import pandas as pd
from langchain_core.tools import tool

from assistente_medico.assistant.database import query_df

alog = logging.getLogger("assistente_medico")


@tool
def consultar_paciente(paciente_id: str) -> str:
    """Retorna dados demográficos e prontuário mais recente do paciente (ID no formato PAC-XXX)."""
    alog.info(f"TOOL consultar_paciente | id={paciente_id}")
    pac = query_df("SELECT * FROM pacientes WHERE id = ?", (paciente_id,))
    if pac.empty:
        return f"[Fonte: EHR] Paciente {paciente_id} não encontrado."

    pront = query_df(
        """SELECT * FROM prontuarios WHERE paciente_id = ?
           ORDER BY data_registro DESC LIMIT 1""",
        (paciente_id,),
    )
    p = pac.iloc[0].to_dict()
    pr = pront.iloc[0].to_dict() if not pront.empty else {}
    texto = textwrap.dedent(f"""
    [Fonte: EHR/prontuário | paciente_id={paciente_id}]
    Código interno: {p.get('nome_codigo')} | Idade: {p.get('idade')} | Sexo: {p.get('sexo')}
    Leito: {p.get('leito')} | Setor: {p.get('setor')}
    Data registro: {pr.get('data_registro', 'N/A')}
    Queixa: {pr.get('queixa_principal', 'N/A')}
    Diagnósticos: {pr.get('diagnosticos', 'N/A')}
    Alergias: {pr.get('alergias', 'N/A')}
    Medicações em uso: {pr.get('medicacoes_em_uso', 'N/A')}
    Observações: {pr.get('observacoes', 'N/A')}
    """).strip()
    return texto


@tool
def listar_exames(paciente_id: str, apenas_pendentes: bool = True) -> str:
    """Lista exames do paciente. Se apenas_pendentes=True, retorna pendentes e agendados."""
    alog.info(f"TOOL listar_exames | id={paciente_id} | pendentes={apenas_pendentes}")
    if apenas_pendentes:
        df = query_df(
            """SELECT nome_exame, status, prioridade, data_solicitacao, resultado
               FROM exames
               WHERE paciente_id = ? AND status IN ('pendente', 'agendado')
               ORDER BY CASE prioridade WHEN 'urgente' THEN 0 ELSE 1 END""",
            (paciente_id,),
        )
    else:
        df = query_df(
            """SELECT nome_exame, status, prioridade, data_solicitacao, resultado
               FROM exames WHERE paciente_id = ?""",
            (paciente_id,),
        )

    if df.empty:
        return (
            f"[Fonte: EHR/exames] Nenhum exame encontrado para {paciente_id} "
            f"(filtro pendentes={apenas_pendentes})."
        )

    linhas = [f"[Fonte: EHR/exames | paciente_id={paciente_id}]"]
    for _, row in df.iterrows():
        res = row["resultado"] if pd.notna(row["resultado"]) else "—"
        linhas.append(
            f"- {row['nome_exame']} | status={row['status']} | "
            f"prioridade={row['prioridade']} | solicitado={row['data_solicitacao']} | resultado={res}"
        )
    return "\n".join(linhas)


@tool
def buscar_protocolo(termo: str) -> str:
    """Busca protocolos internos do hospital por palavra-chave (ex.: hipertensão, DPOC, pneumonia)."""
    alog.info(f"TOOL buscar_protocolo | termo={termo}")
    like = f"%{termo}%"
    df = query_df(
        """SELECT codigo, titulo, especialidade, conteudo, versao, fonte
           FROM protocolos
           WHERE titulo LIKE ? OR conteudo LIKE ? OR especialidade LIKE ? OR codigo LIKE ?
           LIMIT 3""",
        (like, like, like, like),
    )
    if df.empty:
        return f"[Fonte: Protocolos internos] Nenhum protocolo encontrado para '{termo}'."

    blocos = []
    for _, row in df.iterrows():
        blocos.append(textwrap.dedent(f"""
        [Fonte: {row['fonte']} | código={row['codigo']} | versão={row['versao']}]
        Título: {row['titulo']} ({row['especialidade']})
        Conteúdo: {row['conteudo']}
        """).strip())
    return "\n\n".join(blocos)
