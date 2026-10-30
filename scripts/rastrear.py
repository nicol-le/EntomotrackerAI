"""
==============================================================================
ENTOMOTRACK AI - MÓDULO DE RASTREAMENTO MULTIOBJETO (BYTETRACK CPU)
Associação temporal entre quadros, persistência de ID, desenho de trajetórias e exportação.
==============================================================================
"""

import gc
import cv2
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Garantir import do diretório raiz
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.detectar import DetectorInsetosYOLO
from scripts.banco import BancoEntomoTrack


def calcular_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Calcula a Interseção sobre União (IoU) entre duas caixas delimitadoras."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    area_inter = inter_w * inter_h

    area_b1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area_b2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    area_uniao = area_b1 + area_b2 - area_inter

    if area_uniao <= 0.0:
        return 0.0
    return float(area_inter / area_uniao)


class TrilhaObjeto:
    """Representa a trilha histórica de um único inseto com persistência de ID."""
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], confianca: float, classe: str, frame_idx: int):
        self.track_id = track_id
        self.bbox = bbox
        self.confianca = confianca
        self.classe = classe
        self.ultimo_frame = frame_idx
        self.perdido_por_frames = 0
        self.ativo = True

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.historico_centros = [(cx, cy, frame_idx)]

    def atualizar(self, bbox: Tuple[float, float, float, float], confianca: float, frame_idx: int):
        self.bbox = bbox
        self.confianca = confianca
        self.ultimo_frame = frame_idx
        self.perdido_por_frames = 0
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.historico_centros.append((cx, cy, frame_idx))
        if len(self.historico_centros) > 100:
            self.historico_centros.pop(0)

    def marcar_perdido(self):
        self.perdido_por_frames += 1


class ByteTrackNativo:
    """
    Implementação otimizada do algoritmo ByteTrack em puro NumPy.
    Garante máxima velocidade em CPU sem dependências pesadas externas.
    Associa em duas etapas: alta confiança e baixa confiança (ByteTrack original).
    """
    def __init__(self, track_thresh: float = 0.45, track_buffer: int = 30, match_thresh: float = 0.4):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.proximo_id = 1
        self.trilhas: List[TrilhaObjeto] = []

    def atualizar(self, deteccoes: List[Dict[str, Any]], frame_idx: int) -> List[TrilhaObjeto]:
        # Separar detecções de alta confiança e baixa confiança
        dets_altas = [d for d in deteccoes if d["confianca"] >= self.track_thresh]
        dets_baixas = [d for d in deteccoes if d["confianca"] < self.track_thresh]

        trilhas_ativas = [t for t in self.trilhas if t.perdido_por_frames <= self.track_buffer]

        # ETAPA 1: Associar detecções de alta confiança com trilhas ativas via IoU
        matriz_iou = np.zeros((len(trilhas_ativas), len(dets_altas)), dtype=np.float32)
        for i, t in enumerate(trilhas_ativas):
            for j, d in enumerate(dets_altas):
                matriz_iou[i, j] = calcular_iou(t.bbox, d["bbox"])

        trilhas_associadas = set()
        dets_associadas = set()

        # Associação gulosa ordenada por maior IoU
        if matriz_iou.size > 0:
            indices_ordenados = np.dstack(np.unravel_index(np.argsort(-matriz_iou, axis=None), matriz_iou.shape))[0]
            for i, j in indices_ordenados:
                if matriz_iou[i, j] < self.match_thresh:
                    break
                if i not in trilhas_associadas and j not in dets_associadas:
                    trilhas_ativas[i].atualizar(dets_altas[j]["bbox"], dets_altas[j]["confianca"], frame_idx)
                    trilhas_associadas.add(i)
                    dets_associadas.add(j)

        # ETAPA 2: Associar trilhas restantes não associadas com detecções de baixa confiança
        trilhas_restantes = [t for idx, t in enumerate(trilhas_ativas) if idx not in trilhas_associadas]
        matriz_iou2 = np.zeros((len(trilhas_restantes), len(dets_baixas)), dtype=np.float32)
        for i, t in enumerate(trilhas_restantes):
            for j, d in enumerate(dets_baixas):
                matriz_iou2[i, j] = calcular_iou(t.bbox, d["bbox"])

        trilhas_recuperadas = set()
        if matriz_iou2.size > 0:
            indices_ord2 = np.dstack(np.unravel_index(np.argsort(-matriz_iou2, axis=None), matriz_iou2.shape))[0]
            for i, j in indices_ord2:
                if matriz_iou2[i, j] < self.match_thresh:
                    break
                if i not in trilhas_recuperadas:
                    trilhas_restantes[i].atualizar(dets_baixas[j]["bbox"], dets_baixas[j]["confianca"], frame_idx)
                    trilhas_recuperadas.add(i)

        # Atualizar trilhas perdidas
        for idx, t in enumerate(trilhas_ativas):
            if idx not in trilhas_associadas and t not in [trilhas_restantes[k] for k in trilhas_recuperadas]:
                t.marcar_perdido()

        # Iniciar novas trilhas para detecções de alta confiança não associadas
        for j, d in enumerate(dets_altas):
            if j not in dets_associadas:
                nova_trilha = TrilhaObjeto(self.proximo_id, d["bbox"], d["confianca"], d["classe_nome"], frame_idx)
                self.proximo_id += 1
                self.trilhas.append(nova_trilha)

        # Manter apenas trilhas dentro do buffer de perda
        self.trilhas = [t for t in self.trilhas if t.perdido_por_frames <= self.track_buffer]
        return [t for t in self.trilhas if t.perdido_por_frames == 0]


# Paleta de 30 cores visualmente distintas para trajetórias individuais
CORES_PALETA = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 65), (35, 181, 211), (57, 220, 205), (59, 235, 255),
    (0, 152, 225), (34, 87, 255), (72, 85, 121), (158, 158, 158),
    (139, 125, 96), (180, 151, 88), (215, 131, 255), (170, 0, 255),
    (255, 87, 184), (255, 0, 128), (0, 204, 255), (255, 214, 0),
    (0, 230, 118), (0, 229, 255), (41, 121, 255), (101, 31, 255)
]


def rastrear_video(caminho_video: str,
                   detector: Optional[DetectorInsetosYOLO] = None,
                   pular_quadros: int = 1,
                   pasta_saida: Optional[str] = None,
                   callback_progresso = None,
                   flag_cancelar = None) -> Dict[str, Any]:
    """
    Processa um vídeo com ByteTrack, salvando vídeo anotado e CSV de trajetórias.
    Otimizado para streaming contínuo sem estouro de memória (8 GB RAM).
    """
    base_dir = Path(__file__).resolve().parent.parent
    if pasta_saida is None:
        data_str = time.strftime("%Y%m%d_%H%M%S")
        pasta_destino = base_dir / "resultados" / f"analise_{data_str}"
    else:
        pasta_destino = Path(pasta_saida)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    if detector is None:
        detector = DetectorInsetosYOLO()

    tracker = ByteTrackNativo(track_thresh=0.35, track_buffer=30, match_thresh=0.35)

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    total_quadros = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    caminho_video_saida = pasta_destino / "video_rastreado.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(caminho_video_saida), fourcc, fps / max(1, pular_quadros), (w, h))

    registros_csv = []
    frame_idx = 0
    quadros_processados = 0

    print(f"\n[INFO] Iniciando rastreamento de vídeo ({w}x{h} @ {fps:.1f} FPS, {total_quadros} frames)")
    print(f"[STREAMING] Processamento de quadros alternados: a cada {pular_quadros} quadro(s)")

    try:
        while True:
            if flag_cancelar and flag_cancelar():
                print("[INFO] Rastreamento interrompido pelo usuário.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if (frame_idx % pular_quadros) != 0:
                continue

            quadros_processados += 1
            tempo_segundos = frame_idx / fps

            # 1. Detecção YOLO
            deteccoes = detector.detectar_frame(frame, salvar_crops_pendentes=True,
                                                origem_arquivo=Path(caminho_video).name,
                                                frame_idx=frame_idx)

            # 2. Rastreamento ByteTrack
            trilhas_ativas = tracker.atualizar(deteccoes, frame_idx)

            # 3. Desenho no quadro
            frame_vis = frame.copy()
            for trilha in trilhas_ativas:
                tid = trilha.track_id
                x1, y1, x2, y2 = [int(v) for v in trilha.bbox]
                cor = CORES_PALETA[tid % len(CORES_PALETA)]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Desenhar caixa delimitadora e ID
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(frame_vis, f"ID #{tid:02d} ({trilha.classe}) {trilha.confianca:.2f}",
                            (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

                # Desenhar rastro de movimento (cauda temporal)
                pontos = [(int(p[0]), int(p[1])) for p in trilha.historico_centros]
                for i in range(1, len(pontos)):
                    cv2.line(frame_vis, pontos[i - 1], pontos[i], cor, 2)

                # Coletar dados para CSV
                registros_csv.append({
                    "frame": frame_idx,
                    "tempo_s": round(tempo_segundos, 3),
                    "track_id": tid,
                    "classe": trilha.classe,
                    "confianca": round(trilha.confianca, 3),
                    "centro_x": cx,
                    "centro_y": cy,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2
                })

            cv2.putText(frame_vis, f"EntomoTrack AI | Frame: {frame_idx}/{total_quadros} | Insetos Ativos: {len(trilhas_ativas)}",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            out.write(frame_vis)

            if callback_progresso and total_quadros > 0:
                callback_progresso(frame_idx / total_quadros)

            # Limpeza periódica de memória para respeitar limite de 8 GB RAM
            if quadros_processados % 100 == 0:
                gc.collect()

    finally:
        cap.release()
        out.release()
        gc.collect()

    # Exportar CSV de Coordenadas por Quadro
    caminho_csv = pasta_destino / "coordenadas_por_quadro.csv"
    df_coords = pd.DataFrame(registros_csv)
    df_coords.to_csv(str(caminho_csv), index=False, encoding="utf-8")
    print(f"[OK] Coordenadas salvas em CSV: {caminho_csv}")

    return {
        "pasta_resultados": str(pasta_destino),
        "video_anotado": str(caminho_video_saida),
        "csv_coordenadas": str(caminho_csv),
        "total_quadros": frame_idx,
        "total_insetos_rastreados": len(df_coords["track_id"].unique()) if not df_coords.empty else 0,
        "dataframe": df_coords,
        "largura": w,
        "altura": h,
        "fps": fps
    }


if __name__ == "__main__":
    import numpy as np
    # Teste unitário de criação e rastreamento com vídeo sintético
    print("[TESTE] Criando vídeo sintético de 30 quadros para validar ByteTrack...")
    temp_dir = Path(__file__).resolve().parent.parent / "videos"
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_teste = temp_dir / "teste_sintetico.mp4"

    writer = cv2.VideoWriter(str(video_teste), cv2.VideoWriter_fourcc(*'mp4v'), 10, (416, 416))
    for f in range(30):
        img = np.zeros((416, 416, 3), dtype=np.uint8)
        # Inseto 1 movendo-se diagonalmente
        cv2.circle(img, (50 + f * 5, 50 + f * 5), 15, (200, 200, 200), -1)
        # Inseto 2 movendo-se horizontalmente
        cv2.circle(img, (350 - f * 5, 200), 12, (180, 180, 180), -1)
        writer.write(img)
    writer.release()

    res = rastrear_video(str(video_teste))
    print(f"[SUCESSO] Rastreamento concluído. Insetos encontrados: {res['total_insetos_rastreados']}")
