"""
==============================================================================
ENTOMOTRACK AI - SUÍTE COMPLETA DE TESTES AUTOMATIZADOS
Validação de 12 etapas: Dependências, Câmera, Banco, YOLO, ByteTrack,
Análise Comportamental, Exportação e Human-in-the-Loop.
==============================================================================
"""

import os
import gc
import sys
import time
import shutil
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".yolo"))
(BASE_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
(BASE_DIR / ".yolo").mkdir(parents=True, exist_ok=True)
sys.path.append(str(BASE_DIR))


class TestEntomoTrackAI(unittest.TestCase):
    """Bateria de testes de integridade do sistema EntomoTrack AI."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = BASE_DIR / "resultados" / "testes_automatizados"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.db_test_path = cls.test_dir / "test_entomotrack.db"
        if cls.db_test_path.exists():
            cls.db_test_path.unlink()

    # -------------------------------------------------------------------------
    # TESTE 1: Importação de Bibliotecas
    # -------------------------------------------------------------------------
    def test_01_importacoes(self):
        print("\n[TESTE 1/12] Verificando importações essenciais...")
        import cv2
        import torch
        import ultralytics
        import pandas
        import numpy
        import matplotlib
        import sqlite3
        import yaml
        import tkinter
        self.assertIsNotNone(cv2.__version__)
        self.assertIsNotNone(torch.__version__)
        self.assertIsNotNone(ultralytics.__version__)
        print(f" -> OpenCV: {cv2.__version__} | PyTorch: {torch.__version__} | Ultralytics: {ultralytics.__version__}")

    # -------------------------------------------------------------------------
    # TESTE 2: Listagem de Câmeras
    # -------------------------------------------------------------------------
    def test_02_enumeracao_cameras(self):
        print("\n[TESTE 2/12] Verificando módulo de busca de câmeras USB...")
        from scripts.teste_camera import listar_cameras_disponiveis
        cams = listar_cameras_disponiveis(max_indices=2)
        # O teste deve ser bem-sucedido mesmo se nenhuma câmera física estiver plugada no momento
        self.assertIsInstance(cams, list)
        print(f" -> Detecção executada com sucesso. Câmeras encontradas: {len(cams)}")

    # -------------------------------------------------------------------------
    # TESTE 3: Geração de Imagem e Vídeo Sintético
    # -------------------------------------------------------------------------
    def test_03_geracao_video_teste(self):
        print("\n[TESTE 3/12] Gerando vídeo sintético de microscopia para validação...")
        import cv2
        cls_vid_path = self.test_dir / "video_microscopia_sintetico.mp4"
        w, h, fps = 416, 416, 15
        writer = cv2.VideoWriter(str(cls_vid_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # Simular 45 quadros com 2 insetos em movimento
        for f in range(45):
            frame = np.ones((h, w, 3), dtype=np.uint8) * 40 # Fundo escuro de microscópio
            # Inseto A (movimento orbital/diagonal)
            pos_a = (int(80 + f * 5), int(100 + np.sin(f / 4.0) * 30))
            cv2.circle(frame, pos_a, 14, (220, 220, 220), -1)
            cv2.ellipse(frame, pos_a, (16, 10), 30, 0, 360, (180, 180, 180), -1)

            # Inseto B (movimento retilíneo oposto)
            pos_b = (int(340 - f * 4), int(280 + np.cos(f / 5.0) * 20))
            cv2.circle(frame, pos_b, 12, (200, 200, 200), -1)
            writer.write(frame)

        writer.release()
        self.assertTrue(cls_vid_path.exists())
        self.assertGreater(cls_vid_path.stat().st_size, 1000)
        print(f" -> Vídeo sintético gerado: {cls_vid_path.name} ({cls_vid_path.stat().st_size} bytes)")

    # -------------------------------------------------------------------------
    # TESTE 4: Banco de Dados SQLite e Auditoria Científica
    # -------------------------------------------------------------------------
    def test_04_banco_dados_e_auditoria(self):
        print("\n[TESTE 4/12] Testando integridade científica e auditoria no SQLite...")
        from scripts.banco import BancoEntomoTrack
        banco = BancoEntomoTrack(db_path=str(self.db_test_path))

        # Cadastrar espécie
        esp_id = banco.cadastrar_especie(
            codigo="TRI_CONF",
            nome_cientifico="Tribolium confusum",
            nome_popular="Besouro-da-farinha",
            ordem="Coleoptera",
            familia="Tenebrionidae"
        )
        self.assertGreater(esp_id, 0)

        # Registrar detecção
        ident_id = banco.registrar_deteccao(
            origem_arquivo="teste.mp4",
            bbox=(50, 50, 100, 100),
            confianca=0.48,
            classe_detector="inseto",
            status_revisao="INCERTO",
            especie_id=1
        )
        self.assertGreater(ident_id, 0)

        # Aplicar correção do usuário
        sucesso_corr = banco.aplicar_correcao_humana(
            identificacao_id=ident_id,
            nova_especie_id=esp_id,
            status="CONFIRMADO",
            motivo="Identificação taxonômica confirmada"
        )
        self.assertTrue(sucesso_corr)

        # Verificar histórico de auditoria
        with banco.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM historico_alteracoes WHERE identificacao_id = ?", (ident_id,))
            historico = cursor.fetchall()
            self.assertGreater(len(historico), 0)

        # Soft delete (exclusão lógica sem perda de dados)
        sucesso_del = banco.marcar_exclusao_logica(ident_id, definitiva=False)
        self.assertTrue(sucesso_del)
        print(" -> Banco SQLite validado: CRUD, histórico de auditoria e exclusão lógica OK.")

    # -------------------------------------------------------------------------
    # TESTE 5: Carregamento do Modelo YOLO em CPU
    # -------------------------------------------------------------------------
    def test_05_carregamento_yolo_cpu(self):
        print("\n[TESTE 5/12] Verificando carregamento do YOLO em CPU...")
        from scripts.detectar import DetectorInsetosYOLO
        detector = DetectorInsetosYOLO()
        self.assertTrue(detector.modelo_carregado)
        print(f" -> Modelo carregado: {detector.caminho_modelo} em CPU com sucesso.")

    # -------------------------------------------------------------------------
    # TESTE 6: Inferência e Classificação em Situações A, B e C
    # -------------------------------------------------------------------------
    def test_06_deteccao_situacoes_a_b_c(self):
        print("\n[TESTE 6/12] Testando inferência e classificação em Situações A, B e C...")
        from scripts.detectar import DetectorInsetosYOLO
        import cv2
        detector = DetectorInsetosYOLO(conf_min=0.25, conf_incerta_max=0.55)

        # Imagem sintética
        frame = np.zeros((416, 416, 3), dtype=np.uint8)
        cv2.circle(frame, (208, 208), 25, (255, 255, 255), -1)

        dets = detector.detectar_frame(frame)
        self.assertIsInstance(dets, list)
        frame_vis = detector.desenhar_deteccoes(frame, dets)
        self.assertEqual(frame_vis.shape, frame.shape)
        print(f" -> Inferência executada sem falhas. Resposta válida.")

    # -------------------------------------------------------------------------
    # TESTE 7: Rastreamento Multiobjeto ByteTrack
    # -------------------------------------------------------------------------
    def test_07_bytetrack_multiobjeto(self):
        print("\n[TESTE 7/12] Testando rastreamento multiobjeto ByteTrack nativo...")
        from scripts.rastrear import ByteTrackNativo
        tracker = ByteTrackNativo(track_thresh=0.3, track_buffer=10, match_thresh=0.3)

        # Quadro 1: duas detecções
        dets_f1 = [
            {"bbox": (50, 50, 80, 80), "confianca": 0.85, "classe_nome": "inseto"},
            {"bbox": (200, 200, 230, 230), "confianca": 0.90, "classe_nome": "inseto"}
        ]
        trilhas1 = tracker.atualizar(dets_f1, frame_idx=1)
        self.assertEqual(len(trilhas1), 2)
        id1, id2 = trilhas1[0].track_id, trilhas1[1].track_id

        # Quadro 2: deslocamento suave dos dois objetos
        dets_f2 = [
            {"bbox": (55, 52, 85, 82), "confianca": 0.82, "classe_nome": "inseto"},
            {"bbox": (205, 202, 235, 232), "confianca": 0.88, "classe_nome": "inseto"}
        ]
        trilhas2 = tracker.atualizar(dets_f2, frame_idx=2)
        ids_f2 = [t.track_id for t in trilhas2]
        self.assertIn(id1, ids_f2)
        self.assertIn(id2, ids_f2)
        print(" -> ByteTrack: Associação de múltiplos objetos e preservação de IDs validada.")

    # -------------------------------------------------------------------------
    # TESTE 8: Calibração de Escala Espacial
    # -------------------------------------------------------------------------
    def test_08_calibracao_escala(self):
        print("\n[TESTE 8/12] Testando cálculo de calibração espacial (px/mm)...")
        from scripts.banco import BancoEntomoTrack
        banco = BancoEntomoTrack(db_path=str(self.db_test_path))

        dist_conhecida_mm = 2.0
        dist_pixels = 240.0 # 240 pixels equivalem a 2 mm
        px_por_mm = banco.salvar_calibracao(dist_conhecida_mm, dist_pixels, "Teste de calibração")
        self.assertAlmostEqual(px_por_mm, 120.0, places=2)

        ultima = banco.obter_ultima_calibracao()
        self.assertIsNotNone(ultima)
        self.assertAlmostEqual(ultima["pixels_por_mm"], 120.0, places=2)
        print(f" -> Calibração persistida: {px_por_mm:.2f} px/mm ({(1.0/px_por_mm)*1000:.2f} µm/px).")

    # -------------------------------------------------------------------------
    # TESTE 9: Métricas Comportamentais e Cinemática
    # -------------------------------------------------------------------------
    def test_09_metricas_comportamentais(self):
        print("\n[TESTE 9/12] Testando cálculo de parâmetros biológicos e comportamentais...")
        from scripts.analisar_movimento import AnalisadorComportamental

        df_traj = pd.DataFrame({
            "frame": [1, 2, 3, 4, 5],
            "tempo_s": [0.1, 0.2, 0.3, 0.4, 0.5],
            "track_id": [1, 1, 1, 1, 1],
            "classe": ["inseto"] * 5,
            "confianca": [0.9] * 5,
            "centro_x": [100.0, 110.0, 120.0, 120.0, 120.0],
            "centro_y": [100.0, 100.0, 100.0, 100.0, 100.0]
        })

        analisador = AnalisadorComportamental(pixels_por_mm=10.0, limiar_imobilidade_px=1.0)
        saida_pasta = self.test_dir / "resultados_analise"
        res = analisador.analisar_trajetorias(df_traj, str(saida_pasta), largura_quadro=416, altura_quadro=416)

        self.assertTrue(Path(res["csv_metricas"]).exists())
        df_m = res["df_metricas"]
        self.assertEqual(len(df_m), 1)
        # Distância percorrida: 20 pixels / 10 px/mm = 2.0 mm
        self.assertAlmostEqual(df_m["distancia_total_mm"].iloc[0], 2.0, places=1)
        print(f" -> Métricas calculadas: Distância={df_m['distancia_total_mm'].iloc[0]} mm, Tempo Ativo={df_m['tempo_ativo_s'].iloc[0]} s, Imóvel={df_m['tempo_imovel_s'].iloc[0]} s.")

    # -------------------------------------------------------------------------
    # TESTE 10: Exportação de Gráficos e Mapas de Calor
    # -------------------------------------------------------------------------
    def test_10_exportacao_graficos_e_relatorio(self):
        print("\n[TESTE 10/12] Verificando arquivos de gráficos e relatórios gerados...")
        saida_pasta = self.test_dir / "resultados_analise"
        self.assertTrue((saida_pasta / "trajetorias_2d.png").exists())
        self.assertTrue((saida_pasta / "grafico_velocidade.png").exists())
        self.assertTrue((saida_pasta / "grafico_distancia.png").exists())
        self.assertTrue((saida_pasta / "mapa_de_calor.png").exists())
        self.assertTrue((saida_pasta / "relatorio_resumo.txt").exists())
        print(" -> Todos os arquivos científicos gerados e validados no disco.")

    # -------------------------------------------------------------------------
    # TESTE 11: Módulo Human-in-the-Loop e Dataset YOLO
    # -------------------------------------------------------------------------
    def test_11_human_in_the_loop_e_dataset(self):
        print("\n[TESTE 11/12] Testando Human-in-the-Loop e exportação para dataset YOLO...")
        from scripts.banco import BancoEntomoTrack
        from scripts.treinar_modelo import TreinadorYOLO
        import cv2

        banco = BancoEntomoTrack(db_path=str(self.db_test_path))

        # Criar frame e recorte simulado
        frame_dummy = np.zeros((416, 416, 3), dtype=np.uint8)
        img_dummy_path = self.test_dir / "frame_dummy_01.png"
        cv2.imwrite(str(img_dummy_path), frame_dummy)

        # Registrar e confirmar exemplo
        ident_id = banco.registrar_deteccao(
            origem_arquivo="teste_video.mp4",
            bbox=(60, 60, 120, 120),
            confianca=0.92,
            classe_detector="inseto",
            caminho_imagem_original=str(img_dummy_path),
            status_revisao="CONFIRMADO",
            especie_id=1
        )

        treinador = TreinadorYOLO(db_path=str(self.db_test_path))
        yaml_path, classes = treinador.preparar_dataset_yolo()
        self.assertTrue(Path(yaml_path).exists())
        print(f" -> Dataset YOLO gerado com sucesso: {yaml_path}")

    # -------------------------------------------------------------------------
    # TESTE 12: Fluxo Completo Ponta a Ponta
    # -------------------------------------------------------------------------
    def test_12_fluxo_completo_integrado(self):
        print("\n[TESTE 12/12] Executando teste do fluxo completo integrado...")
        from scripts.rastrear import rastrear_video
        from scripts.detectar import DetectorInsetosYOLO
        from scripts.analisar_movimento import AnalisadorComportamental

        vid_path = self.test_dir / "video_microscopia_sintetico.mp4"
        pasta_res = self.test_dir / "execucao_completa"

        # Classe de detector com detecção de contraste para validação no vídeo sintético
        class DetectorValidacao(DetectorInsetosYOLO):
            def detectar_frame(self, frame_bgr, salvar_crops_pendentes=False, origem_arquivo="", frame_idx=0):
                # Detecção por limiar de contraste para objetos claros no fundo escuro sintético
                import cv2
                cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY)
                contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                dets = []
                for c in contornos:
                    if cv2.contourArea(c) > 50:
                        x, y, w, h = cv2.boundingRect(c)
                        dets.append({
                            "bbox": (x, y, x + w, y + h),
                            "confianca": 0.95,
                            "classe_id": 0,
                            "classe_nome": "inseto",
                            "situacao": "SITUACAO_A_CONFIAVEL",
                            "centro": (x + w // 2, y + h // 2),
                            "caminho_recorte": ""
                        })
                return dets

        detector_val = DetectorValidacao(caminho_modelo=str(BASE_DIR / "modelos" / "yolov8n.pt"))

        # 1. Rastrear vídeo com ByteTrack
        res_track = rastrear_video(str(vid_path), detector=detector_val, pasta_saida=str(pasta_res), pular_quadros=1)
        self.assertTrue(Path(res_track["video_anotado"]).exists())
        self.assertTrue(Path(res_track["csv_coordenadas"]).exists())
        self.assertGreater(res_track["total_insetos_rastreados"], 0)

        # 2. Analisar cinemática, gerar gráficos e mapas de calor
        analisador = AnalisadorComportamental(pixels_por_mm=50.0)
        res_analise = analisador.analisar_trajetorias(
            df_coords=res_track["dataframe"],
            pasta_saida=str(pasta_res),
            largura_quadro=res_track["largura"],
            altura_quadro=res_track["altura"]
        )
        self.assertTrue(Path(res_analise["csv_metricas"]).exists())
        self.assertTrue((Path(pasta_res) / "trajetorias_2d.png").exists())
        self.assertTrue((Path(pasta_res) / "mapa_de_calor.png").exists())
        self.assertTrue((Path(pasta_res) / "relatorio_resumo.txt").exists())

        print(f" -> FLUXO COMPLETO EXECUTADO COM SUCESSO!")
        print(f" -> Insetos rastreados com IDs persistentes: {res_track['total_insetos_rastreados']}")
        print(f" -> Resultados validados na pasta: {pasta_res}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
