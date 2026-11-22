import logging
import sqlite3
from pathlib import Path

import pandas as pd

from assistente_medico.config import DB_PATH

alog = logging.getLogger("assistente_medico")


def criar_banco_hospital(db_path: Path = DB_PATH, overwrite: bool = True) -> Path:
    if overwrite and db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE pacientes (
        id TEXT PRIMARY KEY,
        nome_codigo TEXT NOT NULL,
        idade INTEGER,
        sexo TEXT,
        leito TEXT,
        setor TEXT
    );

    CREATE TABLE prontuarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        paciente_id TEXT,
        data_registro TEXT,
        queixa_principal TEXT,
        diagnosticos TEXT,
        alergias TEXT,
        medicacoes_em_uso TEXT,
        observacoes TEXT,
        FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
    );

    CREATE TABLE exames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        paciente_id TEXT,
        nome_exame TEXT,
        status TEXT,
        resultado TEXT,
        data_solicitacao TEXT,
        prioridade TEXT,
        FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
    );

    CREATE TABLE protocolos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        codigo TEXT UNIQUE,
        titulo TEXT,
        especialidade TEXT,
        conteudo TEXT,
        versao TEXT,
        fonte TEXT
    );
    """)

    pacientes = [
        ("PAC-001", "PX-A17", 58, "F", "3B-12", "Clínica Médica"),
        ("PAC-002", "PX-B22", 67, "M", "UTI-04", "UTI"),
        ("PAC-003", "PX-C09", 41, "F", "2A-07", "Cardiologia"),
        ("PAC-004", "PX-D33", 72, "M", "PS-02", "Pronto-Socorro"),
    ]
    cur.executemany("INSERT INTO pacientes VALUES (?,?,?,?,?,?)", pacientes)

    prontuarios = [
        ("PAC-001", "2026-03-10", "Cefaleia e visão turva",
         "Hipertensão arterial sistêmica; Suspeita de glaucoma",
         "Dipirona", "Losartana 50mg 1x/dia",
         "PA na admissão 168/98 mmHg. Aguardando avaliação oftalmológica."),
        ("PAC-002", "2026-03-11", "Dispneia e saturação baixa",
         "DPOC exacerbada; Insuficiência respiratória tipo 1",
         "Penicilina", "Formoterol + budesonida; Oxigenoterapia 2L",
         "SpO2 88% em ar ambiente na admissão. Em O2 suplementar."),
        ("PAC-003", "2026-03-09", "Dor torácica atípica",
         "Angina estável sob investigação",
         "Nenhuma conhecida", "AAS 100mg; Atenolol 25mg",
         "ECG sem supradesnível. Troponina pendente."),
        ("PAC-004", "2026-03-12", "Febre e tosse produtiva",
         "Pneumonia comunitária provável",
         "Sulfa", "Nenhuma em uso contínuo",
         "CURB-65 = 2. Radiografia de tórax pendente."),
    ]
    cur.executemany(
        """INSERT INTO prontuarios
           (paciente_id, data_registro, queixa_principal, diagnosticos,
            alergias, medicacoes_em_uso, observacoes)
           VALUES (?,?,?,?,?,?,?)""",
        prontuarios,
    )

    exames = [
        ("PAC-001", "Campimetria visual", "pendente", None, "2026-03-10", "urgente"),
        ("PAC-001", "Creatinina sérica", "realizado", "1.1 mg/dL (ref 0.6-1.2)", "2026-03-10", "rotina"),
        ("PAC-001", "Fundo de olho", "agendado", None, "2026-03-11", "urgente"),
        ("PAC-002", "Gasometria arterial", "realizado", "pH 7.36; PaO2 62; PaCO2 48", "2026-03-11", "urgente"),
        ("PAC-002", "RX tórax", "pendente", None, "2026-03-11", "urgente"),
        ("PAC-003", "Troponina I", "pendente", None, "2026-03-09", "urgente"),
        ("PAC-003", "ECG 12 derivações", "realizado", "Ritmo sinusal; sem supra ST", "2026-03-09", "urgente"),
        ("PAC-004", "Hemograma completo", "realizado", "Leucócitos 14.200; Neutrófilos 82%", "2026-03-12", "urgente"),
        ("PAC-004", "RX tórax PA", "pendente", None, "2026-03-12", "urgente"),
        ("PAC-004", "PCR", "agendado", None, "2026-03-12", "rotina"),
    ]
    cur.executemany(
        """INSERT INTO exames
           (paciente_id, nome_exame, status, resultado, data_solicitacao, prioridade)
           VALUES (?,?,?,?,?,?)""",
        exames,
    )

    protocolos = [
        ("PROT-HAS-01", "Protocolo de Hipertensão Arterial — Conduta inicial",
         "Clínica Médica",
         "1) Confirmar PA em 2 aferições. 2) Avaliar lesão de órgão-alvo. "
         "3) Solicitar creatinina, eletrólitos, ECG. 4) Iniciar ou ajustar IECA/BRA "
         "conforme protocolo institucional. 5) Encaminhar oftalmo se sintomas visuais. "
         "IMPORTANTE: qualquer ajuste medicamentoso exige validação do médico assistente.",
         "v2.1", "Comissão de Protocolos Clínicos — Hospital (sintético)"),
        ("PROT-DPOC-02", "Protocolo de Exacerbação de DPOC",
         "Pneumologia / UTI",
         "1) Avaliar SpO2 e gasometria. 2) O2 visando SpO2 88–92%. "
         "3) Broncodilatadores e corticosteroide sistêmico conforme gravidade. "
         "4) Antibiótico se critérios de Anthonisen. 5) Considerar VNI se acidose respiratória. "
         "Toda prescrição deve ser validada pelo plantonista.",
         "v1.4", "Comissão de Protocolos Clínicos — Hospital (sintético)"),
        ("PROT-SCA-03", "Protocolo de Dor Torácica / SCA",
         "Cardiologia",
         "1) ECG em até 10 minutos. 2) Troponina seriada. 3) AAS se sem contraindicação. "
         "4) Classificar risco (HEART/TIMI). 5) Não liberar paciente com troponina pendente "
         "e dor em investigação. Conduta final: cardiologista de plantão.",
         "v3.0", "Comissão de Protocolos Clínicos — Hospital (sintético)"),
        ("PROT-PNM-04", "Protocolo de Pneumonia Adquirida na Comunidade",
         "Pronto-Socorro",
         "1) Calcular CURB-65. 2) RX tórax. 3) Hemograma + PCR. "
         "4) Antibioticoterapia empírica conforme gravidade e alergias. "
         "5) Internação se CURB-65 >= 2. Prescrição somente após validação médica.",
         "v2.0", "Comissão de Protocolos Clínicos — Hospital (sintético)"),
    ]
    cur.executemany(
        """INSERT INTO protocolos
           (codigo, titulo, especialidade, conteudo, versao, fonte)
           VALUES (?,?,?,?,?,?)""",
        protocolos,
    )

    conn.commit()
    conn.close()
    alog.info(f"Banco hospitalar sintético criado em {db_path}")
    return db_path


def query_df(sql: str, params: tuple = (), db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
