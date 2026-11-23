import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from assistente_medico.config import ETAPA3_AUDIT_JSONL

PADROES_RISCO_ETAPA3 = {
    "prescricao_ou_conduta": r"\b(prescrev\w*|prescri\w*|receit\w*|dosagem|dose|posologia|administr\w*|suspend\w*|suspens\w*|inici(?:e|ar)|tome|tomar|aplique|liber(?:ar|e) o paciente|autorizar alta)\b",
    "quantidade_de_medicamento": r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ml|ui|comprimidos?|gotas?)\b",
}


def registrar_etapa3(caminho, pedido: str, etapa: str, **dados: Any) -> None:
    evento = dict(id=pedido, utc=datetime.now(timezone.utc).isoformat(), etapa=etapa, **dados)
    with Path(caminho).open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")


def normalizar(texto: str) -> str:
    minusculo = texto.casefold()
    separado = unicodedata.normalize("NFKD", minusculo)
    return separado.encode("ascii", "ignore").decode()


def motivo_risco(texto: str) -> Optional[str]:
    texto = normalizar(texto)
    for nome, padrao in PADROES_RISCO_ETAPA3.items():
        if re.search(padrao, texto):
            return nome
    return None


def exames_confirmados(estado: Dict[str, Any], paciente: str) -> List[Tuple[str, str, str]]:
    itens = []
    exames = estado.get("exames", "").strip().splitlines()
    cabecalho = f"[Fonte: EHR/exames | paciente_id={paciente}]"
    if not exames or exames[0].strip() != cabecalho:
        return itens
    for linha in exames[1:]:
        campos = [c.strip() for c in linha.split("|")]
        if len(campos) < 3 or not campos[0].startswith("- "):
            continue
        nome = campos[0][2:].strip()
        if not nome or campos[1] not in {"status=pendente", "status=agendado"}:
            continue
        if campos[2] not in {
            "prioridade=urgente", "prioridade=alta", "prioridade=normal",
            "prioridade=baixa", "prioridade=rotina", "prioridade=média",
        }:
            continue
        status = campos[1].split("=")[1]
        prioridade = campos[2].split("=")[1]
        itens.append((nome, f"{nome} — {status}, prioridade {prioridade}.", cabecalho))
    return itens


def protocolos_confirmados(estado: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    itens = []
    protocolo = re.compile(
        r"^\[Fonte: ([^\]\n]+?)\s*\|\s*código=([^|\]\n]+?)\s*\|\s*versão=([^\]\n]+?)\]\s*\n\s*Título:\s*([^\n]+)",
        re.M,
    )
    for match in protocolo.finditer(estado.get("protocolos", "")):
        fonte, codigo, versao, titulo = (v.strip() for v in match.groups())
        nome = titulo.split(" (")[0]
        ref = f"{fonte} | código={codigo} | versão={versao}"
        itens.append((nome, f"Protocolo localizado: {titulo}.", ref))
    return itens


def preparar_pergunta_etapa3(pergunta: str) -> str:
    instrucao = (
        "Use os registros recuperados para esta consulta. "
        "Cite pelo nome os exames pendentes ou agendados e os protocolos pertinentes. "
        "Se faltar informação, indique que o registro é insuficiente."
    )
    return pergunta.strip() + "\n\n" + instrucao


def encerrar_etapa3(
    caminho, pedido: str, status: str, motivo: str, itens: Tuple = (),
) -> Dict[str, Any]:
    fontes: List[str] = []
    frases: List[str] = []
    for nome, frase, referencia in itens:
        if referencia not in fontes:
            fontes.append(referencia)
        numero = fontes.index(referencia) + 1
        frases.append(f"{frase} [{numero}]")

    registrar_etapa3(
        caminho, pedido, "decisao", status=status, motivo=motivo, fontes=fontes,
        itens_publicados=len(itens), revisao_humana=True,
    )
    if frases:
        mensagem = "\n".join(frases)
    else:
        mensagem = "Consulta encaminhada para revisão. Nenhuma orientação clínica foi liberada."
    return dict(
        id=pedido, status=status, motivo=motivo, resposta=mensagem,
        fontes=fontes, revisao_humana=True,
    )


def verificar_entrada_etapa3(paciente: str, pergunta: str) -> Optional[str]:
    if not isinstance(paciente, str) or not re.fullmatch(r"PAC-\d{3}", paciente):
        return "identificador_invalido"
    if not isinstance(pergunta, str) or not pergunta.strip():
        return "pergunta_vazia"
    risco = motivo_risco(pergunta)
    if risco:
        return "entrada_" + risco
    return None


def verificar_retorno_etapa3(estado: Dict[str, Any], paciente: str) -> Optional[str]:
    prontuario = estado.get("prontuario", "").strip().splitlines()
    cabecalho = f"[Fonte: EHR/prontuário | paciente_id={paciente}]"
    if estado.get("paciente_id") != paciente or not prontuario or prontuario[0].strip() != cabecalho:
        return "prontuario_nao_confirmado"
    rascunho = estado.get("resposta_bruta", "")
    if not isinstance(rascunho, str) or not rascunho.strip():
        return "rascunho_vazio"
    risco = motivo_risco(rascunho)
    if risco:
        return "saida_" + risco
    return None


def conferir_itens_etapa3(estado: Dict[str, Any], paciente: str):
    disponiveis = exames_confirmados(estado, paciente) + protocolos_confirmados(estado)
    citados = []
    texto = normalizar(estado["resposta_bruta"])
    for item in disponiveis:
        nome = normalizar(item[0])
        padrao = r"(?<!\w)" + re.escape(nome) + r"(?!\w)"
        if re.search(padrao, texto):
            citados.append(item)
    return disponiveis, citados


def verificar_com_grafo(
    paciente: str, pergunta: str, executar, caminho, pedido: str,
) -> Dict[str, Any]:
    estado = executar(dict(
        request_id=pedido, paciente_id=paciente,
        pergunta_medico=preparar_pergunta_etapa3(pergunta), alertas=[], logs_fluxo=[],
    ))
    registrar_etapa3(
        caminho, pedido, "retorno_grafo", passos=len(estado.get("logs_fluxo", [])),
        alerta_urgente=bool(estado.get("exames_urgentes_pendentes")),
    )
    motivo = verificar_retorno_etapa3(estado, paciente)
    if motivo:
        return encerrar_etapa3(caminho, pedido, "retida", motivo)
    disponiveis, citados = conferir_itens_etapa3(estado, paciente)
    registrar_etapa3(
        caminho, pedido, "conferencia", itens_disponiveis=len(disponiveis),
        itens_reconhecidos=len(citados),
    )
    if not citados:
        return encerrar_etapa3(caminho, pedido, "retida", "sem_evidencia_correspondente")
    unicos = list(dict.fromkeys(citados))
    return encerrar_etapa3(caminho, pedido, "resumo_documental", "dados_conferidos", unicos)


def consultar_com_validacao(
    paciente: str,
    pergunta: str,
    executar_fluxo=None,
    caminho=ETAPA3_AUDIT_JSONL,
    *,
    assistente_app=None,
) -> Dict[str, Any]:
    pedido = str(uuid4())
    registrar_etapa3(caminho, pedido, "entrada", politica="etapa3-v2")
    motivo = verificar_entrada_etapa3(paciente, pergunta)
    if motivo:
        return encerrar_etapa3(caminho, pedido, "retida", motivo)
    try:
        if executar_fluxo is None:
            if assistente_app is None:
                raise ValueError("Forneça executar_fluxo ou assistente_app.")
            executar_fluxo = assistente_app.invoke
        return verificar_com_grafo(paciente, pergunta, executar_fluxo, caminho, pedido)
    except (OSError, UnicodeError):
        raise
    except Exception as erro:
        registrar_etapa3(caminho, pedido, "erro_fluxo", classe=type(erro).__name__)
        return encerrar_etapa3(caminho, pedido, "retida", "falha_no_fluxo")


def ler_eventos_da_consulta(caminho, id_consulta: str) -> List[Dict[str, Any]]:
    eventos = []
    with Path(caminho).open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            evento = json.loads(linha)
            if evento["id"] == id_consulta:
                eventos.append(evento)
    return eventos
