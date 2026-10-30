"""
==============================================================================
ENTOMOTRACK AI - MÓDULO DE BANCO DE DADOS E INTEGRIDADE CIENTÍFICA
SQLite local para rastreabilidade, espécies, correções e histórico de modelos.
==============================================================================
"""

import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple


class BancoEntomoTrack:
    """
    Gerenciador de integridade científica e persistência local SQLite.
    Garante que nenhum dado experimental seja excluído silenciosamente,
    armazenando histórico de alterações e versionamento de modelos.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.db_path = base_dir / "banco_dados" / "entomotrack.db"
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.inicializar_tabelas()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def inicializar_tabelas(self):
        """Cria as tabelas necessárias com controle relacional e auditoria."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Tabela de Espécies e Classes Taxonômicas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS especies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    codigo TEXT UNIQUE NOT NULL,
                    nome_cientifico TEXT NOT NULL,
                    nome_popular TEXT,
                    ordem TEXT,
                    familia TEXT,
                    genero TEXT,
                    observacoes TEXT,
                    imagem_referencia TEXT,
                    data_cadastro TEXT NOT NULL,
                    ativo INTEGER DEFAULT 1
                )
            """)

            # 2. Tabela de Sessões / Vídeos / Imagens de Experimentos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessoes_experimentais (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tipo TEXT NOT NULL,             -- 'video' ou 'imagem'
                    caminho_arquivo TEXT NOT NULL,
                    data_registro TEXT NOT NULL,
                    resolucao TEXT,
                    fps REAL,
                    duracao_segundos REAL,
                    observacoes TEXT
                )
            """)

            # 3. Tabela de Detecções, Identificações e Revisões Human-in-the-Loop
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS identificacoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sessao_id INTEGER,
                    origem_arquivo TEXT NOT NULL,
                    frame_idx INTEGER DEFAULT 0,
                    timestamp_video REAL DEFAULT 0.0,
                    bbox_x1 REAL NOT NULL,
                    bbox_y1 REAL NOT NULL,
                    bbox_x2 REAL NOT NULL,
                    bbox_y2 REAL NOT NULL,
                    confianca_detector REAL,
                    classe_detector TEXT,
                    especie_id INTEGER,
                    status_revisao TEXT DEFAULT 'PENDENTE', -- 'PENDENTE', 'CONFIRMADO', 'CORRIGIDO', 'FALSO_POSITIVO', 'DESCARTADO'
                    eh_falso_positivo INTEGER DEFAULT 0,
                    caminho_recorte TEXT,
                    caminho_imagem_original TEXT,
                    revisado_por_usuario INTEGER DEFAULT 0,
                    enviado_para_dataset INTEGER DEFAULT 0,
                    split_dataset TEXT,                     -- 'train', 'val', 'test'
                    observacoes TEXT,
                    criado_em TEXT NOT NULL,
                    atualizado_em TEXT NOT NULL,
                    excluido_logico INTEGER DEFAULT 0,
                    FOREIGN KEY (especie_id) REFERENCES especies(id),
                    FOREIGN KEY (sessao_id) REFERENCES sessoes_experimentais(id)
                )
            """)

            # 4. Tabela de Auditoria e Histórico de Alterações (Integridade Científica)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historico_alteracoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identificacao_id INTEGER NOT NULL,
                    campo_alterado TEXT NOT NULL,
                    valor_anterior TEXT,
                    valor_novo TEXT,
                    data_alteracao TEXT NOT NULL,
                    motivo TEXT,
                    FOREIGN KEY (identificacao_id) REFERENCES identificacoes(id)
                )
            """)

            # 5. Tabela de Histórico e Versões de Modelos Treinados
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS modelos_historico (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome_versao TEXT UNIQUE NOT NULL,
                    caminho_pesos TEXT NOT NULL,
                    modelo_base TEXT,
                    data_treino TEXT NOT NULL,
                    classes_json TEXT NOT NULL,
                    qtd_imagens INTEGER,
                    qtd_anotacoes INTEGER,
                    epocas INTEGER,
                    img_size INTEGER,
                    batch_size INTEGER,
                    map50 REAL,
                    map50_95 REAL,
                    em_uso INTEGER DEFAULT 0,
                    observacoes TEXT
                )
            """)

            # 6. Tabela de Calibrações de Escala
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calibracoes_escala (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_calibracao TEXT NOT NULL,
                    distancia_conhecida_mm REAL NOT NULL,
                    distancia_pixels REAL NOT NULL,
                    pixels_por_mm REAL NOT NULL,
                    observacoes TEXT
                )
            """)

            # Inserir classe genérica padrão 'inseto' se tabela vazia
            cursor.execute("SELECT COUNT(*) FROM especies")
            if cursor.fetchone()[0] == 0:
                agora = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO especies (codigo, nome_cientifico, nome_popular, ordem, familia, genero, observacoes, data_cadastro, ativo)
                    VALUES ('INSETO_GENERICO', 'Insecta indet.', 'Inseto Genérico', 'Insecta', '', '', 'Classe taxonômica genérica de base', ?, 1)
                """, (agora,))

            conn.commit()

    # =========================================================================
    # GERENCIAMENTO DE ESPÉCIES
    # =========================================================================

    def cadastrar_especie(self, codigo: str, nome_cientifico: str, nome_popular: str = "",
                          ordem: str = "", familia: str = "", genero: str = "",
                          observacoes: str = "", imagem_referencia: str = "") -> int:
        """Cadastra uma nova espécie ou grupo taxonômico."""
        agora = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO especies (codigo, nome_cientifico, nome_popular, ordem, familia, genero, observacoes, imagem_referencia, data_cadastro, ativo)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (codigo.strip().upper(), nome_cientifico.strip(), nome_popular.strip(),
                  ordem.strip(), familia.strip(), genero.strip(), observacoes.strip(),
                  imagem_referencia, agora))
            conn.commit()
            return cursor.lastrowid

    def atualizar_especie(self, especie_id: int, dados: Dict[str, Any]) -> bool:
        """Atualiza dados cadastrais de uma espécie."""
        campos = []
        valores = []
        for k, v in dados.items():
            if k in ['nome_cientifico', 'nome_popular', 'ordem', 'familia', 'genero', 'observacoes', 'imagem_referencia', 'ativo']:
                campos.append(f"{k} = ?")
                valores.append(v)
        if not campos:
            return False
        valores.append(especie_id)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE especies SET {', '.join(campos)} WHERE id = ?", valores)
            conn.commit()
            return cursor.rowcount > 0

    def listar_especies(self, apenas_ativas: bool = True) -> List[Dict[str, Any]]:
        """Lista todas as espécies cadastradas."""
        query = "SELECT * FROM especies"
        if apenas_ativas:
            query += " WHERE ativo = 1"
        query += " ORDER BY nome_cientifico ASC"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def obter_especie_por_id(self, especie_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM especies WHERE id = ?", (especie_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # DETECÇÕES, REVISÕES HUMAN-IN-THE-LOOP E AUDITORIA
    # =========================================================================

    def registrar_deteccao(self, origem_arquivo: str, bbox: Tuple[float, float, float, float],
                           confianca: float, classe_detector: str,
                           frame_idx: int = 0, timestamp_video: float = 0.0,
                           caminho_recorte: str = "", caminho_imagem_original: str = "",
                           status_revisao: str = "PENDENTE", especie_id: Optional[int] = None) -> int:
        """Registra uma detecção automática ou incerta para revisão."""
        agora = datetime.now().isoformat()
        x1, y1, x2, y2 = bbox
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO identificacoes (
                    origem_arquivo, frame_idx, timestamp_video,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    confianca_detector, classe_detector, especie_id,
                    status_revisao, caminho_recorte, caminho_imagem_original,
                    criado_em, atualizado_em, excluido_logico
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (origem_arquivo, frame_idx, timestamp_video,
                  float(x1), float(y1), float(x2), float(y2),
                  float(confianca), str(classe_detector), especie_id,
                  status_revisao, caminho_recorte, caminho_imagem_original,
                  agora, agora))
            conn.commit()
            return cursor.lastrowid

    def aplicar_correcao_humana(self, identificacao_id: int, nova_especie_id: Optional[int],
                                status: str, eh_falso_positivo: bool = False,
                                novo_bbox: Optional[Tuple[float, float, float, float]] = None,
                                observacao: str = "", motivo: str = "Revisão pelo usuário") -> bool:
        """
        Registra a correção do usuário com auditoria científica estrita.
        Nunca sobrescreve sem deixar rastro de alteração.
        """
        agora = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM identificacoes WHERE id = ?", (identificacao_id,))
            antigo = cursor.fetchone()
            if not antigo:
                return False

            antigo_dict = dict(antigo)

            # Gravar no histórico de auditoria
            if antigo_dict['especie_id'] != nova_especie_id:
                cursor.execute("""
                    INSERT INTO historico_alteracoes (identificacao_id, campo_alterado, valor_anterior, valor_novo, data_alteracao, motivo)
                    VALUES (?, 'especie_id', ?, ?, ?, ?)
                """, (identificacao_id, str(antigo_dict['especie_id']), str(nova_especie_id), agora, motivo))

            if antigo_dict['status_revisao'] != status:
                cursor.execute("""
                    INSERT INTO historico_alteracoes (identificacao_id, campo_alterado, valor_anterior, valor_novo, data_alteracao, motivo)
                    VALUES (?, 'status_revisao', ?, ?, ?, ?)
                """, (identificacao_id, antigo_dict['status_revisao'], status, agora, motivo))

            # Atualizar registro
            if novo_bbox:
                bx1, by1, bx2, by2 = novo_bbox
                cursor.execute("""
                    UPDATE identificacoes SET
                        especie_id = ?,
                        status_revisao = ?,
                        eh_falso_positivo = ?,
                        bbox_x1 = ?, bbox_y1 = ?, bbox_x2 = ?, bbox_y2 = ?,
                        revisado_por_usuario = 1,
                        observacoes = ?,
                        atualizado_em = ?
                    WHERE id = ?
                """, (nova_especie_id, status, 1 if eh_falso_positivo else 0,
                      float(bx1), float(by1), float(bx2), float(by2),
                      observacao, agora, identificacao_id))
            else:
                cursor.execute("""
                    UPDATE identificacoes SET
                        especie_id = ?,
                        status_revisao = ?,
                        eh_falso_positivo = ?,
                        revisado_por_usuario = 1,
                        observacoes = ?,
                        atualizado_em = ?
                    WHERE id = ?
                """, (nova_especie_id, status, 1 if eh_falso_positivo else 0,
                      observacao, agora, identificacao_id))

            conn.commit()
            return True

    def marcar_exclusao_logica(self, identificacao_id: int, motivo: str = "Exclusão de anotação incorreta", definitiva: bool = False) -> bool:
        """
        Exclui logicamente a anotação para preservar os dados brutos e arquivos originais.
        Se definitiva=True, remove o registro do banco.
        """
        agora = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if definitiva:
                cursor.execute("DELETE FROM historico_alteracoes WHERE identificacao_id = ?", (identificacao_id,))
                cursor.execute("DELETE FROM identificacoes WHERE id = ?", (identificacao_id,))
            else:
                cursor.execute("""
                    INSERT INTO historico_alteracoes (identificacao_id, campo_alterado, valor_anterior, valor_novo, data_alteracao, motivo)
                    VALUES (?, 'excluido_logico', '0', '1', ?, ?)
                """, (identificacao_id, agora, motivo))
                cursor.execute("""
                    UPDATE identificacoes SET
                        excluido_logico = 1,
                        status_revisao = 'DESCARTADO',
                        atualizado_em = ?
                    WHERE id = ?
                """, (agora, identificacao_id))
            conn.commit()
            return True

    def listar_pendentes_revisao(self, limite: int = 100) -> List[Dict[str, Any]]:
        """Retorna registros de insetos não identificados ou com baixa confiança."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT i.*, e.nome_cientifico, e.nome_popular, e.codigo as especie_codigo
                FROM identificacoes i
                LEFT JOIN especies e ON i.especie_id = e.id
                WHERE i.status_revisao IN ('PENDENTE', 'INCERTO')
                  AND i.excluido_logico = 0
                ORDER BY i.id DESC
                LIMIT ?
            """, (limite,))
            return [dict(row) for row in cursor.fetchall()]

    def listar_exemplos_para_dataset(self) -> List[Dict[str, Any]]:
        """Retorna anotações confirmadas ou corrigidas aptas para compor o dataset de treino."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT i.*, e.codigo as especie_codigo, e.nome_cientifico
                FROM identificacoes i
                JOIN especies e ON i.especie_id = e.id
                WHERE i.status_revisao IN ('CONFIRMADO', 'CORRIGIDO')
                  AND i.eh_falso_positivo = 0
                  AND i.excluido_logico = 0
                ORDER BY i.origem_arquivo, i.frame_idx ASC
            """)
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # HISTÓRICO E CONTROLE DE MODELOS
    # =========================================================================

    def registrar_modelo_treinado(self, nome_versao: str, caminho_pesos: str,
                                  classes: List[str], modelo_base: str,
                                  epocas: int, img_size: int, batch_size: int,
                                  qtd_imagens: int, qtd_anotacoes: int,
                                  map50: float = 0.0, map50_95: float = 0.0,
                                  observacoes: str = "") -> int:
        """Registra a geração de um novo modelo treinado."""
        agora = datetime.now().isoformat()
        classes_json = json.dumps(classes)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO modelos_historico (
                    nome_versao, caminho_pesos, modelo_base, data_treino,
                    classes_json, qtd_imagens, qtd_anotacoes,
                    epocas, img_size, batch_size, map50, map50_95,
                    em_uso, observacoes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (nome_versao, caminho_pesos, modelo_base, agora,
                  classes_json, qtd_imagens, qtd_anotacoes,
                  epocas, img_size, batch_size, map50, map50_95,
                  observacoes))
            conn.commit()
            return cursor.lastrowid

    def definir_modelo_em_uso(self, modelo_id: int) -> bool:
        """Define explicitamente qual versão de modelo está ativa."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE modelos_historico SET em_uso = 0")
            cursor.execute("UPDATE modelos_historico SET em_uso = 1 WHERE id = ?", (modelo_id,))
            conn.commit()
            return cursor.rowcount > 0

    def listar_versoes_modelos(self) -> List[Dict[str, Any]]:
        """Lista histórico de todas as versões de modelos."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM modelos_historico ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def salvar_calibracao(self, distancia_mm: float, distancia_px: float, observacoes: str = "") -> float:
        """Salva a calibração de escala em pixels por milímetro."""
        px_por_mm = distancia_px / distancia_mm if distancia_mm > 0 else 1.0
        agora = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO calibracoes_escala (data_calibracao, distancia_conhecida_mm, distancia_pixels, pixels_por_mm, observacoes)
                VALUES (?, ?, ?, ?, ?)
            """, (agora, float(distancia_mm), float(distancia_px), float(px_por_mm), observacoes))
            conn.commit()
        return px_por_mm

    def obter_ultima_calibracao(self) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM calibracoes_escala ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None


if __name__ == "__main__":
    banco = BancoEntomoTrack()
    especies = banco.listar_especies()
    print(f"[OK] Banco SQLite inicializado com sucesso em: {banco.db_path}")
    print(f"[INFO] Espécies cadastradas inicialmente: {len(especies)}")
    for esp in especies:
        print(f" - ID: {esp['id']} | Código: {esp['codigo']} | Científico: {esp['nome_cientifico']}")
