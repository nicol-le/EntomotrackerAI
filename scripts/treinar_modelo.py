"""
==============================================================================
ENTOMOTRACK AI - MÓDULO DE TREINAMENTO DE MODELOS YOLO EM CPU
Preparação de dataset anti-vazamento, fine-tuning e versionamento científico.
==============================================================================
"""

import os
import gc
import sys
import yaml
import shutil
import cv2
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Garantir import do diretório raiz
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.banco import BancoEntomoTrack


class TreinadorYOLO:
    """
    Controlador de treinamento e fine-tuning de modelos YOLO para detecção de insetos.
    Otimizado para execução em CPU (8 GB RAM) sem GPU.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.banco = BancoEntomoTrack(db_path)
        self.pasta_datasets = self.base_dir / "datasets"
        self.pasta_modelos = self.base_dir / "modelos"
        self.pasta_datasets.mkdir(parents=True, exist_ok=True)
        self.pasta_modelos.mkdir(parents=True, exist_ok=True)

    def preparar_dataset_yolo(self, split_val_ratio: float = 0.2) -> Tuple[str, Dict[int, str]]:
        """
        Exporta exemplos aprovados e corrigidos do SQLite para o formato oficial YOLO.
        Agrupa por arquivo de origem para evitar vazamento de dados (data leakage) entre treino e validação.
        """
        exemplos = self.banco.listar_exemplos_para_dataset()
        if not exemplos:
            raise ValueError("Nenhum exemplo aprovado/confirmado no banco para compor o dataset.")

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        ds_dir = self.pasta_datasets / f"dataset_insetos_{timestamp_str}"

        img_train_dir = ds_dir / "images" / "train"
        img_val_dir = ds_dir / "images" / "val"
        lbl_train_dir = ds_dir / "labels" / "train"
        lbl_val_dir = ds_dir / "labels" / "val"

        for p in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # Mapeamento dinâmico de classes a partir das espécies aprovadas
        especies = self.banco.listar_especies()
        id_para_idx = {}
        idx_para_nome = {}
        for idx, esp in enumerate(especies):
            id_para_idx[esp["id"]] = idx
            idx_para_nome[idx] = esp["nome_cientifico"] if esp["nome_cientifico"] else esp["codigo"]

        # Agrupar anotações por imagem/frame original
        grupos_imagem: Dict[str, List[Dict[str, Any]]] = {}
        for ex in exemplos:
            caminho_img = ex.get("caminho_imagem_original")
            if not caminho_img or not Path(caminho_img).exists():
                continue
            if caminho_img not in grupos_imagem:
                grupos_imagem[caminho_img] = []
            grupos_imagem[caminho_img].append(ex)

        if not grupos_imagem:
            raise ValueError("As imagens originais vinculadas aos exemplos aprovados não foram encontradas no disco.")

        # Divisão agrupada por vídeo/origem para prevenir vazamento de dados
        chaves_imagens = list(grupos_imagem.keys())
        import random
        random.seed(42)
        random.shuffle(chaves_imagens)

        num_val = max(1, int(len(chaves_imagens) * split_val_ratio)) if len(chaves_imagens) > 1 else 0
        val_keys = set(chaves_imagens[:num_val])

        total_anotacoes = 0
        for caminho_img, lista_anotacoes in grupos_imagem.items():
            eh_val = caminho_img in val_keys
            dest_img_dir = img_val_dir if eh_val else img_train_dir
            dest_lbl_dir = lbl_val_dir if eh_val else lbl_train_dir

            img = cv2.imread(caminho_img)
            if img is None:
                continue

            h, w = img.shape[:2]
            nome_base = Path(caminho_img).stem

            # Copiar imagem
            shutil.copy2(caminho_img, dest_img_dir / Path(caminho_img).name)

            # Criar arquivo de labels YOLO: <class_id> <x_center> <y_center> <width> <height> (normalizados 0 a 1)
            linhas_label = []
            for an in lista_anotacoes:
                esp_id = an["especie_id"]
                class_idx = id_para_idx.get(esp_id, 0)

                x1, y1, x2, y2 = an["bbox_x1"], an["bbox_y1"], an["bbox_x2"], an["bbox_y2"]
                # Normalização YOLO
                x_center = ((x1 + x2) / 2.0) / w
                y_center = ((y1 + y2) / 2.0) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                # Clamping de segurança
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.001, min(1.0, width))
                height = max(0.001, min(1.0, height))

                linhas_label.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                total_anotacoes += 1

            with open(dest_lbl_dir / f"{nome_base}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(linhas_label))

        # Criar data.yaml
        yaml_content = {
            "path": str(ds_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": idx_para_nome
        }
        yaml_file = ds_dir / "data.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

        print(f"[DATASET] Gerado com sucesso em: {yaml_file}")
        print(f" -> Imagens totais: {len(grupos_imagem)} | Anotações: {total_anotacoes}")
        print(f" -> Classes ({len(idx_para_nome)}): {idx_para_nome}")

        return str(yaml_file), idx_para_nome

    def treinar_modelo_cpu(self, data_yaml: str, modelo_base: str = "yolov8n.pt",
                           epocas: int = 15, img_size: int = 416, batch_size: int = 4,
                           nome_versao: Optional[str] = None) -> Dict[str, Any]:
        """
        Executa o treinamento do YOLO exclusivamente na CPU, respeitando os 8 GB de RAM.
        Salva e versiona os pesos resultantes no banco de dados.
        """
        from ultralytics import YOLO

        if nome_versao is None:
            nome_versao = f"modelo_insetos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n[TREINO] Iniciando fine-tuning em CPU...")
        print(f" -> Base: {modelo_base} | Épocas: {epocas} | Batch: {batch_size} | Resolução: {img_size}x{img_size}")

        modelo = YOLO(modelo_base)

        # Treinamento configurado especificamente para CPU convencional sem travar
        resultados = modelo.train(
            data=data_yaml,
            epochs=epocas,
            imgsz=img_size,
            batch=batch_size,
            device='cpu',
            workers=2,           # Poucos workers para economizar memória RAM
            patience=10,         # Early stopping
            save=True,
            val=True,
            verbose=True,
            plots=False          # Desativa plots pesados na CPU
        )

        # Localizar pesos gerados (best.pt)
        diretorio_saida = Path(resultados.save_dir) if hasattr(resultados, 'save_dir') else None
        caminho_best = None
        if diretorio_saida and (diretorio_saida / "weights" / "best.pt").exists():
            caminho_best = diretorio_saida / "weights" / "best.pt"
        else:
            # Fallback para busca nos runs
            runs_pesos = list(self.base_dir.glob("runs/**/best.pt"))
            if runs_pesos:
                caminho_best = runs_pesos[-1]

        # Copiar pesos para a pasta permanente de modelos
        destino_pesos = self.pasta_modelos / f"{nome_versao}.pt"
        if caminho_best and Path(caminho_best).exists():
            shutil.copy2(str(caminho_best), str(destino_pesos))
        else:
            # Salvar modelo diretamente
            modelo.save(str(destino_pesos))

        # Obter métricas de validação
        map50 = 0.0
        map50_95 = 0.0
        try:
            if hasattr(resultados, 'box'):
                map50 = float(resultados.box.map50)
                map50_95 = float(resultados.box.map)
        except Exception:
            pass

        # Ler classes do yaml
        with open(data_yaml, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        classes_dict = yaml_data.get("names", {})
        classes_lista = [classes_dict[k] for k in sorted(classes_dict.keys())]

        # Registrar no banco SQLite
        modelo_id = self.banco.registrar_modelo_treinado(
            nome_versao=nome_versao,
            caminho_pesos=str(destino_pesos),
            classes=classes_lista,
            modelo_base=modelo_base,
            epocas=epocas,
            img_size=img_size,
            batch_size=batch_size,
            qtd_imagens=0,
            qtd_anotacoes=0,
            map50=map50,
            map50_95=map50_95,
            observacoes="Treinado em CPU via EntomoTrack AI"
        )

        gc.collect()

        print(f"\n[SUCESSO] Treinamento finalizado!")
        print(f" -> Modelo salvo em: {destino_pesos}")
        print(f" -> Versão registrada no banco: #{modelo_id} ({nome_versao})")
        print(f" -> Métricas: mAP50 = {map50:.4f} | mAP50-95 = {map50_95:.4f}")

        return {
            "modelo_id": modelo_id,
            "nome_versao": nome_versao,
            "caminho_pesos": str(destino_pesos),
            "map50": map50,
            "map50_95": map50_95
        }


if __name__ == "__main__":
    print("[INFO] Testando módulo de treino...")
    treinador = TreinadorYOLO()
    print(f"[OK] Instância do Treinador inicializada. Datasets dir: {treinador.pasta_datasets}")
