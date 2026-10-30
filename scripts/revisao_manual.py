"""
==============================================================================
ENTOMOTRACK AI - MÓDULO DE REVISÃO E HUMAN-IN-THE-LOOP LEARNING
Interface lógica para revisão de detecções incertas, correções e falso positivo.
==============================================================================
"""

import cv2
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Garantir import do diretório raiz
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.banco import BancoEntomoTrack


class GerenciadorRevisaoManual:
    """
    Controlador do ciclo de aprendizagem assistida pelo pesquisador (Human-in-the-Loop).
    Facilita a inspeção de recortes, atribuição de espécies e envio seguro ao dataset.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.banco = BancoEntomoTrack(db_path)
        self.pasta_recortes = self.base_dir / "imagens" / "recortes"
        self.pasta_pendentes = self.base_dir / "imagens" / "pendentes"
        self.pasta_recortes.mkdir(parents=True, exist_ok=True)
        self.pasta_pendentes.mkdir(parents=True, exist_ok=True)

    def extrair_e_registrar_recorte(self, frame_bgr, bbox: Tuple[float, float, float, float],
                                    confianca: float, classe_detector: str,
                                    origem_arquivo: str, frame_idx: int = 0,
                                    tempo_s: float = 0.0) -> int:
        """
        Extrai o recorte do inseto no quadro, salva no disco e registra no banco SQLite como PENDENTE.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return -1

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nome_recorte = f"crop_{Path(origem_arquivo).stem}_f{frame_idx}_{timestamp_str}.png"
        caminho_recorte = self.pasta_pendentes / nome_recorte
        cv2.imwrite(str(caminho_recorte), crop)

        # Salvar imagem do frame original para uso no dataset
        pasta_frames = self.base_dir / "imagens" / "frames_originais"
        pasta_frames.mkdir(parents=True, exist_ok=True)
        nome_frame = f"frame_{Path(origem_arquivo).stem}_f{frame_idx}_{timestamp_str}.png"
        caminho_frame = pasta_frames / nome_frame
        cv2.imwrite(str(caminho_frame), frame_bgr)

        # Registrar no SQLite
        identificacao_id = self.banco.registrar_deteccao(
            origem_arquivo=origem_arquivo,
            bbox=(x1, y1, x2, y2),
            confianca=confianca,
            classe_detector=classe_detector,
            frame_idx=frame_idx,
            timestamp_video=tempo_s,
            caminho_recorte=str(caminho_recorte),
            caminho_imagem_original=str(caminho_frame),
            status_revisao="PENDENTE",
            especie_id=1 # Classe genérica inicial
        )

        return identificacao_id

    def confirmar_especie(self, identificacao_id: int, especie_id: int, observacoes: str = "") -> bool:
        """Confirma a espécie de uma identificação pendente e move para aprovados."""
        sucesso = self.banco.aplicar_correcao_humana(
            identificacao_id=identificacao_id,
            nova_especie_id=especie_id,
            status="CONFIRMADO",
            eh_falso_positivo=False,
            observacao=observacoes,
            motivo="Confirmado pelo usuário via Human-in-the-Loop"
        )
        return sucesso

    def corrigir_especie(self, identificacao_id: int, nova_especie_id: int,
                         novo_bbox: Optional[Tuple[float, float, float, float]] = None,
                         observacoes: str = "") -> bool:
        """Registra a correção de espécie ou ajuste de caixa delimitadora."""
        sucesso = self.banco.aplicar_correcao_humana(
            identificacao_id=identificacao_id,
            nova_especie_id=nova_especie_id,
            status="CORRIGIDO",
            eh_falso_positivo=False,
            novo_bbox=novo_bbox,
            observacao=observacoes,
            motivo="Correção de espécie / caixa delimitadora pelo usuário"
        )
        return sucesso

    def marcar_falso_positivo(self, identificacao_id: int, observacoes: str = "") -> bool:
        """Marca a detecção como falso positivo (ex: sujeira, reflexo ou sombra)."""
        sucesso = self.banco.aplicar_correcao_humana(
            identificacao_id=identificacao_id,
            nova_especie_id=None,
            status="FALSO_POSITIVO",
            eh_falso_positivo=True,
            observacao=observacoes,
            motivo="Marcado como falso positivo"
        )
        return sucesso

    def excluir_anotacao(self, identificacao_id: int, definitiva: bool = False) -> bool:
        """Exclui a anotação sem apagar arquivos de vídeo originais."""
        return self.banco.marcar_exclusao_logica(
            identificacao_id=identificacao_id,
            motivo="Exclusão de anotação a pedido do usuário",
            definitiva=definitiva
        )

    def obter_estatisticas_revisao(self) -> Dict[str, int]:
        """Retorna resumo quantitativo de itens pendentes, confirmados e falsos positivos."""
        with self.banco.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status_revisao, COUNT(*) FROM identificacoes WHERE excluido_logico = 0 GROUP BY status_revisao")
            contagens = dict(cursor.fetchall())

            cursor.execute("SELECT COUNT(*) FROM especies WHERE ativo = 1")
            total_especies = cursor.fetchone()[0]

        return {
            "pendentes": contagens.get("PENDENTE", 0) + contagens.get("INCERTO", 0),
            "confirmados": contagens.get("CONFIRMADO", 0),
            "corrigidos": contagens.get("CORRIGIDO", 0),
            "falsos_positivos": contagens.get("FALSO_POSITIVO", 0),
            "total_especies_cadastradas": total_especies
        }


if __name__ == "__main__":
    rev = GerenciadorRevisaoManual()
    stats = rev.obter_estatisticas_revisao()
    print("[OK] Módulo de Revisão Manual inicializado.")
    print(f" -> Estatísticas atuais: {stats}")
