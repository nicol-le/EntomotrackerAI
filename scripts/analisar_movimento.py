"""
==============================================================================
ENTOMOTRACK AI - MÓDULO DE ANÁLISE COMPORTAMENTAL E CINEMÁTICA
Cálculo de métricas biológicas, mapas de calor, gráficos e relatórios científicos.
==============================================================================
"""

import os
import sys
from pathlib import Path

# Garantir diretórios de cache locais graváveis
_BASE = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_BASE / ".matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_BASE / ".yolo"))
(_BASE / ".matplotlib").mkdir(parents=True, exist_ok=True)
(_BASE / ".yolo").mkdir(parents=True, exist_ok=True)

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Renderização não-interativa segura para threads e background
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Garantir import do diretório raiz
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.banco import BancoEntomoTrack


class AnalisadorComportamental:
    """
    Processa coordenadas temporais de rastreamento e calcula parâmetros cinemáticos,
    mapas de densidade espacial e gera gráficos científicos.
    """

    def __init__(self, pixels_por_mm: Optional[float] = None,
                 limiar_imobilidade_px: float = 2.0,
                 janela_suavizacao: int = 5):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.px_por_mm = pixels_por_mm
        self.limiar_imobilidade_px = limiar_imobilidade_px
        self.janela_suavizacao = janela_suavizacao
        self.escala_ativa = False

        self._carregar_calibracao()

    def _carregar_calibracao(self):
        """Carrega calibração ativa se não foi passada explicitamente."""
        if self.px_por_mm is not None and self.px_por_mm > 0:
            self.escala_ativa = True
            return

        config_path = self.base_dir / "configuracoes" / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cal = cfg.get("calibracao", {})
                if cal.get("escala_ativa", False):
                    self.px_por_mm = float(cal.get("pixels_por_milimetro", 1.0))
                    self.escala_ativa = True
            except Exception as e:
                print(f"[AVISO] Não foi possível ler calibração do config.yaml: {e}")

    def analisar_trajetorias(self, df_coords: pd.DataFrame, pasta_saida: str,
                            largura_quadro: int = 1280, altura_quadro: int = 720) -> Dict[str, Any]:
        """
        Calcula métricas individuais e gera todos os gráficos e relatórios na pasta_saida.
        """
        pasta_destino = Path(pasta_saida)
        pasta_destino.mkdir(parents=True, exist_ok=True)

        if df_coords.empty or "track_id" not in df_coords.columns:
            print("[AVISO] DataFrame de coordenadas vazio ou sem coluna 'track_id'.")
            caminho_csv_metricas = pasta_destino / "metricas_individuais.csv"
            pd.DataFrame(columns=["track_id", "classe", "distancia_total_px", "velocidade_media_px_s"]).to_csv(str(caminho_csv_metricas), index=False)
            return {
                "csv_metricas": str(caminho_csv_metricas),
                "df_metricas": pd.DataFrame(),
                "escala_utilizada": "Pixels (não calibrado)",
                "pasta_destino": str(pasta_destino)
            }

        metricas_lista = []
        ids_unicos = sorted(df_coords["track_id"].unique())

        unidade_dist = "mm" if self.escala_ativa else "px"
        unidade_vel = "mm/s" if self.escala_ativa else "px/s"
        fator_conv = self.px_por_mm if (self.escala_ativa and self.px_por_mm) else 1.0

        for tid in ids_unicos:
            sub = df_coords[df_coords["track_id"] == tid].sort_values("frame").copy()
            if len(sub) < 2:
                continue

            xs = sub["centro_x"].values.astype(float)
            ys = sub["centro_y"].values.astype(float)
            tempos = sub["tempo_s"].values.astype(float)

            # Deslocamento euclidiano quadro a quadro em pixels
            dx = np.diff(xs)
            dy = np.diff(ys)
            dt = np.diff(tempos)
            dt[dt == 0] = 0.001 # Prevenir divisão por zero

            distancias_px = np.sqrt(dx**2 + dy**2)
            velocidades_px_s = distancias_px / dt

            # Suavização por média móvel se houver pontos suficientes
            if len(velocidades_px_s) >= self.janela_suavizacao:
                v_suave = np.convolve(velocidades_px_s, np.ones(self.janela_suavizacao)/self.janela_suavizacao, mode='same')
            else:
                v_suave = velocidades_px_s

            # Conversão física se calibrado
            distancia_total = float(np.sum(distancias_px) / fator_conv)
            vel_media = float(np.mean(v_suave) / fator_conv)
            vel_maxima = float(np.max(v_suave) / fator_conv)

            # Tempo de atividade vs tempo de imobilidade
            # Imobilidade: deslocamento abaixo do limiar
            quadros_parado = np.sum(distancias_px <= self.limiar_imobilidade_px)
            tempo_parado = float(np.sum(dt[distancias_px <= self.limiar_imobilidade_px]))
            tempo_ativo = float(np.sum(dt[distancias_px > self.limiar_imobilidade_px]))
            tempo_total_observado = float(tempos[-1] - tempos[0])

            # Mudanças de direção (ângulos entre vetores sucessivos > 45 graus)
            angulos = np.arctan2(dy, dx)
            mudancas_angulares = np.abs(np.diff(angulos))
            # Ajustar descontinuidades cíclicas de pi
            mudancas_angulares = np.where(mudancas_angulares > np.pi, 2 * np.pi - mudancas_angulares, mudancas_angulares)
            limiar_radianos = np.radians(45.0)
            num_mudancas_direcao = int(np.sum(mudancas_angulares >= limiar_radianos))

            classe = sub["classe"].iloc[0] if "classe" in sub.columns else "inseto"
            conf_media = float(sub["confianca"].mean()) if "confianca" in sub.columns else 1.0

            metrica = {
                "track_id": tid,
                "classe": classe,
                "confianca_media": round(conf_media, 3),
                "total_quadros": len(sub),
                "tempo_total_s": round(tempo_total_observado, 2),
                f"distancia_total_{unidade_dist}": round(distancia_total, 2),
                f"velocidade_media_{unidade_vel}": round(vel_media, 2),
                f"velocidade_maxima_{unidade_vel}": round(vel_maxima, 2),
                "tempo_ativo_s": round(tempo_ativo, 2),
                "tempo_imovel_s": round(tempo_parado, 2),
                "indice_atividade_pct": round((tempo_ativo / tempo_total_observado * 100.0) if tempo_total_observado > 0 else 0, 1),
                "mudancas_direcao": num_mudancas_direcao,
                "calibrado": self.escala_ativa
            }
            metricas_lista.append(metrica)

        df_metricas = pd.DataFrame(metricas_lista)
        caminho_csv_metricas = pasta_destino / "metricas_individuais.csv"
        df_metricas.to_csv(str(caminho_csv_metricas), index=False, encoding="utf-8")
        print(f"[OK] Métricas individuais salvas em: {caminho_csv_metricas}")

        # GERAÇÃO DE GRÁFICOS CIENTÍFICOS
        self._gerar_grafico_trajetorias(df_coords, pasta_destino, largura_quadro, altura_quadro)
        self._gerar_grafico_velocidades(df_coords, pasta_destino, fator_conv, unidade_vel)
        self._gerar_grafico_distancia(df_coords, pasta_destino, fator_conv, unidade_dist)
        self._gerar_mapa_calor(df_coords, pasta_destino, largura_quadro, altura_quadro)
        self._gerar_relatorio_texto(df_metricas, pasta_destino, unidade_dist, unidade_vel)

        return {
            "csv_metricas": str(caminho_csv_metricas),
            "df_metricas": df_metricas,
            "escala_utilizada": f"{self.px_por_mm:.3f} px/mm" if self.escala_ativa else "Pixels (não calibrado)",
            "pasta_destino": str(pasta_destino)
        }

    def _gerar_grafico_trajetorias(self, df_coords: pd.DataFrame, pasta_destino: Path, w: int, h: int):
        """Gráfico 2D das trajetórias espaciais."""
        plt.figure(figsize=(9, 6), dpi=150)
        plt.title("EntomoTrack AI - Trajetórias Espaciais dos Insetos", fontsize=13, fontweight='bold')

        for tid in df_coords["track_id"].unique():
            sub = df_coords[df_coords["track_id"] == tid].sort_values("frame")
            plt.plot(sub["centro_x"], sub["centro_y"], marker='o', markersize=2, label=f"Inseto #{tid}")
            # Marcar início e fim
            plt.scatter(sub["centro_x"].iloc[0], sub["centro_y"].iloc[0], color='green', s=35, zorder=5)
            plt.scatter(sub["centro_x"].iloc[-1], sub["centro_y"].iloc[-1], color='red', s=35, zorder=5)

        plt.xlim(0, w)
        plt.ylim(h, 0) # Inverter eixo Y para coincidir com coordenadas de imagem
        plt.xlabel("Posição X (pixels)")
        plt.ylabel("Posição Y (pixels)")
        plt.grid(True, linestyle="--", alpha=0.5)
        if len(df_coords["track_id"].unique()) <= 15:
            plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(str(pasta_destino / "trajetorias_2d.png"))
        plt.close()

    def _gerar_grafico_velocidades(self, df_coords: pd.DataFrame, pasta_destino: Path, fator_conv: float, unidade: str):
        """Gráfico temporal de velocidade por indivíduo."""
        plt.figure(figsize=(9, 5), dpi=150)
        plt.title(f"Velocidade ao Longo do Tempo ({unidade})", fontsize=13, fontweight='bold')

        for tid in df_coords["track_id"].unique()[:10]: # Limitar a 10 para visualização limpa
            sub = df_coords[df_coords["track_id"] == tid].sort_values("frame")
            if len(sub) < 3:
                continue
            dx = np.diff(sub["centro_x"].values)
            dy = np.diff(sub["centro_y"].values)
            dt = np.diff(sub["tempo_s"].values)
            dt[dt == 0] = 0.001
            v = (np.sqrt(dx**2 + dy**2) / dt) / fator_conv
            t = sub["tempo_s"].values[1:]
            plt.plot(t, v, label=f"ID #{tid}", alpha=0.8)

        plt.xlabel("Tempo (segundos)")
        plt.ylabel(f"Velocidade ({unidade})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(str(pasta_destino / "grafico_velocidade.png"))
        plt.close()

    def _gerar_grafico_distancia(self, df_coords: pd.DataFrame, pasta_destino: Path, fator_conv: float, unidade: str):
        """Gráfico de distância acumulada ao longo do tempo."""
        plt.figure(figsize=(9, 5), dpi=150)
        plt.title(f"Distância Acumulada ({unidade})", fontsize=13, fontweight='bold')

        for tid in df_coords["track_id"].unique()[:10]:
            sub = df_coords[df_coords["track_id"] == tid].sort_values("frame")
            if len(sub) < 2:
                continue
            dx = np.diff(sub["centro_x"].values)
            dy = np.diff(sub["centro_y"].values)
            d_acum = np.insert(np.cumsum(np.sqrt(dx**2 + dy**2)), 0, 0.0) / fator_conv
            plt.plot(sub["tempo_s"], d_acum, label=f"ID #{tid}", linewidth=2)

        plt.xlabel("Tempo (segundos)")
        plt.ylabel(f"Distância Acumulada ({unidade})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(str(pasta_destino / "grafico_distancia.png"))
        plt.close()

    def _gerar_mapa_calor(self, df_coords: pd.DataFrame, pasta_destino: Path, w: int, h: int):
        """Gera mapa de calor da densidade espacial de ocupação dos insetos."""
        plt.figure(figsize=(9, 6), dpi=150)
        plt.title("Mapa de Calor - Densidade de Ocupação Espacial", fontsize=13, fontweight='bold')

        x = df_coords["centro_x"].values
        y = df_coords["centro_y"].values

        # Histograma 2D de densidade
        bins = [min(60, max(10, w // 20)), min(60, max(10, h // 20))]
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, w], [0, h]])

        # Plotar mapa de calor
        plt.imshow(heatmap.T, origin='upper', extent=[0, w, h, 0], cmap='inferno', interpolation='gaussian', aspect='auto')
        plt.colorbar(label="Frequência de Ocupação (detecções)")
        plt.xlabel("Coordenada X (pixels)")
        plt.ylabel("Coordenada Y (pixels)")
        plt.tight_layout()
        plt.savefig(str(pasta_destino / "mapa_de_calor.png"))
        plt.close()

    def _gerar_relatorio_texto(self, df_metricas: pd.DataFrame, pasta_destino: Path, unid_dist: str, unid_vel: str):
        """Gera relatório resumido estruturado em texto plano."""
        caminho_relatorio = pasta_destino / "relatorio_resumo.txt"
        with open(caminho_relatorio, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("ENTOMOTRACK AI - RELATÓRIO DE ANÁLISE COMPORTAMENTAL DE INSETOS\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Data da análise: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Status da Calibração Espacial: {'CALIBRADO (' + str(round(self.px_por_mm, 3)) + ' px/mm)' if self.escala_ativa else 'NÃO CALIBRADO (Medições em pixels)'}\n")
            f.write(f"Total de insetos individuais analisados: {len(df_metricas)}\n")
            f.write(f"Limiar de imobilidade configurado: {self.limiar_imobilidade_px} pixels/frame\n\n")

            f.write("AVISO CIENTÍFICO:\n")
            f.write("Os tempos de imobilidade e demais indicadores derivam de critérios geométricos\n")
            f.write("e parâmetros configuráveis de visão computacional, devendo ser interpretados\n")
            f.write("com validação metodológica para cada grupo taxonômico específico.\n\n")

            f.write("-" * 70 + "\n")
            f.write("RESUMO ESTATÍSTICO GERAL:\n")
            f.write("-" * 70 + "\n")

            if not df_metricas.empty:
                col_dist = f"distancia_total_{unid_dist}"
                col_vel = f"velocidade_media_{unid_vel}"
                f.write(f"Distância média percorrida: {df_metricas[col_dist].mean():.2f} {unid_dist} (Desvio Padrão: {df_metricas[col_dist].std():.2f})\n")
                f.write(f"Velocidade média geral: {df_metricas[col_vel].mean():.2f} {unid_vel}\n")
                f.write(f"Média de atividade: {df_metricas['indice_atividade_pct'].mean():.1f}%\n")
                f.write(f"Média de mudanças de direção: {df_metricas['mudancas_direcao'].mean():.1f}\n\n")

                f.write("-" * 70 + "\n")
                f.write("MÉTRICAS INDIVIDUAIS POR INSETO:\n")
                f.write("-" * 70 + "\n")
                f.write(df_metricas.to_string(index=False))
                f.write("\n\n" + "=" * 70 + "\n")

        print(f"[OK] Relatório resumido salvo em: {caminho_relatorio}")


if __name__ == "__main__":
    # Teste unitário do analisador comportamental com dados simulados
    df_simulado = pd.DataFrame({
        "frame": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "tempo_s": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
        "track_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "classe": ["inseto"] * 10,
        "confianca": [0.85] * 10,
        "centro_x": [100, 110, 120, 125, 130, 200, 200, 201, 202, 202],
        "centro_y": [100, 105, 110, 112, 115, 300, 300, 300, 301, 301]
    })
    analisador = AnalisadorComportamental()
    pasta_teste = Path(__file__).resolve().parent.parent / "resultados" / "teste_analise"
    analisador.analisar_trajetorias(df_simulado, str(pasta_teste))
    print("[SUCESSO] Módulo de análise comportamental validado.")
