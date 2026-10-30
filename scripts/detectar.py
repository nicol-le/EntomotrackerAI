"""
==============================================================================
ENTOMOTRACK AI - MÓDULO DE DETECÇÃO DE INSETOS VIA YOLO (OTIMIZADO CPU)
Execução leve, sem GPU, com suporte a múltiplas espécies e situações A, B e C.
==============================================================================
"""

import os
import gc
import cv2
import sys
from pathlib import Path

# Garantir diretórios de cache locais graváveis
_BASE = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_BASE / ".matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_BASE / ".yolo"))
(_BASE / ".matplotlib").mkdir(parents=True, exist_ok=True)
(_BASE / ".yolo").mkdir(parents=True, exist_ok=True)

import yaml
import torch
from typing import List, Dict, Tuple, Any, Optional

# Garantir import do diretório raiz
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.banco import BancoEntomoTrack

# Otimização estrita de CPU para não congelar o Windows com 8 GB de RAM
torch.set_num_threads(4)


class DetectorInsetosYOLO:
    """
    Detector de insetos baseado em Ultralytics YOLOv8n (nano),
    altamente otimizado para execução exclusiva em CPU.
    """

    def __init__(self, caminho_modelo: Optional[str] = None,
                 conf_min: float = 0.35, conf_incerta_max: float = 0.55,
                 iou_thresh: float = 0.45, img_size: int = 416):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.conf_min = conf_min
        self.conf_incerta_max = conf_incerta_max
        self.iou_thresh = iou_thresh
        self.img_size = img_size

        self.caminho_modelo = caminho_modelo
        self.modelo = None
        self.nomes_classes = {}
        self.modelo_carregado = False

        self._carregar_configuracao()
        self.carregar_pesos(caminho_modelo)

    def _carregar_configuracao(self):
        config_path = self.base_dir / "configuracoes" / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                inf = cfg.get("inferencia", {})
                self.conf_min = inf.get("confianca_minima", self.conf_min)
                self.conf_incerta_max = inf.get("confianca_incerta_max", self.conf_incerta_max)
                self.iou_thresh = inf.get("iou_threshold", self.iou_thresh)
                self.img_size = inf.get("tamanho_imagem", self.img_size)
            except Exception as e:
                print(f"[AVISO] Não foi possível ler config.yaml: {e}")

    def carregar_pesos(self, caminho_pesos: Optional[str] = None):
        """Carrega modelo YOLOv8n em CPU com tratamento de erros."""
        from ultralytics import YOLO

        if caminho_pesos is None:
            # Buscar modelo treinado mais recente na pasta modelos ou modelo padrão
            modelos_dir = self.base_dir / "modelos"
            modelos_dir.mkdir(parents=True, exist_ok=True)
            pesos_disponiveis = list(modelos_dir.glob("*.pt"))
            if pesos_disponiveis:
                caminho_pesos = str(pesos_disponiveis[-1])
            else:
                caminho_pesos = "yolov8n.pt"  # Pesos base nano leves

        self.caminho_modelo = str(caminho_pesos)
        print(f"[INFO] Carregando detector YOLO em CPU: {self.caminho_modelo}...")

        try:
            self.modelo = YOLO(self.caminho_modelo)
            # Garantir dispositivo CPU
            self.modelo.to('cpu')
            self.nomes_classes = self.modelo.names if hasattr(self.modelo, 'names') else {0: 'inseto'}
            self.modelo_carregado = True
            print(f"[OK] Detector carregado com sucesso. Classes conhecidas: {self.nomes_classes}")
        except Exception as e:
            print(f"[ERRO CRÍTICO] Falha ao carregar modelo YOLO: {e}")
            self.modelo_carregado = False

    def detectar_frame(self, frame_bgr, salvar_crops_pendentes: bool = False,
                       origem_arquivo: str = "", frame_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Executa a inferência YOLO em um único quadro de imagem.
        Gera classificação em Situação A (Confiável), B (Incerta) e C (Desconhecido).
        """
        if not self.modelo_carregado or self.modelo is None:
            return []

        # Inferência em CPU com tamanho de imagem leve (416x416)
        resultados = self.modelo.predict(
            source=frame_bgr,
            conf=self.conf_min * 0.7, # captura limiar levemente inferior para identificar Situação C
            iou=self.iou_thresh,
            imgsz=self.img_size,
            device='cpu',
            verbose=False
        )

        deteccoes = []
        h_orig, w_orig = frame_bgr.shape[:2]
        base_recortes = self.base_dir / "imagens" / "recortes"
        base_pendentes = self.base_dir / "imagens" / "pendentes"

        for r in resultados:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                nome_classe = self.nomes_classes.get(cls_id, f"classe_{cls_id}")

                x1, y1, x2, y2 = [int(v) for v in xyxy]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_orig, x2)
                y2 = min(h_orig, y2)

                # Classificação em Situação A, B ou C
                if conf >= self.conf_incerta_max:
                    situacao = "SITUACAO_A_CONFIAVEL"
                elif conf >= self.conf_min:
                    situacao = "SITUACAO_B_INCERTA"
                else:
                    situacao = "SITUACAO_C_DESCONHECIDO"

                caminho_recorte = ""
                # Salvar recorte para revisão se for incerto ou não identificado
                if salvar_crops_pendentes and situacao in ["SITUACAO_B_INCERTA", "SITUACAO_C_DESCONHECIDO"]:
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        timestamp_str = f"f{frame_idx}_{x1}_{y1}"
                        nome_arquivo_recorte = base_pendentes / f"pendente_{timestamp_str}.jpg"
                        cv2.imwrite(str(nome_arquivo_recorte), crop)
                        caminho_recorte = str(nome_arquivo_recorte)

                deteccoes.append({
                    "bbox": (x1, y1, x2, y2),
                    "confianca": conf,
                    "classe_id": cls_id,
                    "classe_nome": nome_classe,
                    "situacao": situacao,
                    "centro": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "caminho_recorte": caminho_recorte
                })

        return deteccoes

    def desenhar_deteccoes(self, frame_bgr, deteccoes: List[Dict[str, Any]]) -> Any:
        """Desenha caixas e informações com cores diferenciadas por situação."""
        frame_out = frame_bgr.copy()
        for det in deteccoes:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confianca"]
            classe = det["classe_nome"]
            sit = det["situacao"]

            # Cor baseada no status de certeza
            if sit == "SITUACAO_A_CONFIAVEL":
                cor = (0, 255, 0)       # Verde: confiável
                label = f"{classe} {conf:.2f}"
            elif sit == "SITUACAO_B_INCERTA":
                cor = (0, 200, 255)     # Amarelo/Laranja: incerto
                label = f"? {classe} {conf:.2f}"
            else:
                cor = (0, 0, 255)       # Vermelho: não reconhecido
                label = f"Nao Identif. {conf:.2f}"

            cv2.rectangle(frame_out, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(frame_out, label, (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        return frame_out


if __name__ == "__main__":
    detector = DetectorInsetosYOLO()
    print("[TESTE] Verificando integridade do detector...")
    # Testar com imagem sintética
    import numpy as np
    img_teste = np.zeros((416, 416, 3), dtype=np.uint8)
    cv2.circle(img_teste, (208, 208), 30, (255, 255, 255), -1)
    dets = detector.detectar_frame(img_teste)
    print(f"[OK] Detector respondeu. Detecções no quadro de teste: {len(dets)}")
