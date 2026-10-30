"""
==============================================================================
ENTOMOTRACK AI - MÓDULO DE CALIBRAÇÃO DE ESCALA ESPACIAL
Conversão entre pixels e milímetros reais com referência micrométrica.
==============================================================================
"""

import cv2
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Garantir import do diretório raiz
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.banco import BancoEntomoTrack


class CalibradorEscala:
    """
    Interface interativa de calibração espacial de microscopia.
    Permite clicar em 2 pontos de uma referência conhecida (ex: micrômetro ou régua)
    e obter a constante exata de pixels por milímetro (px/mm).
    """

    def __init__(self, caminho_imagem: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.caminho_imagem = caminho_imagem
        self.pontos = []
        self.imagem_original = None
        self.imagem_display = None
        self.px_por_mm = 1.0

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pontos) < 2:
                self.pontos.append((x, y))
                cv2.circle(self.imagem_display, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.imagem_display, f"P{len(self.pontos)} ({x},{y})",
                            (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                if len(self.pontos) == 2:
                    p1, p2 = self.pontos[0], self.pontos[1]
                    cv2.line(self.imagem_display, p1, p2, (0, 255, 0), 2)
                    dist_px = np.linalg.norm(np.array(p1) - np.array(p2))
                    p_meio = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    cv2.putText(self.imagem_display, f"{dist_px:.1f} px",
                                p_meio, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Calibracao de Escala - EntomoTrack AI", self.imagem_display)

    def calibrar(self, distancia_conhecida_mm: float = 1.0) -> Optional[float]:
        """
        Executa a calibração interativa na imagem fornecida.
        """
        if self.caminho_imagem and Path(self.caminho_imagem).exists():
            self.imagem_original = cv2.imread(str(self.caminho_imagem))
        else:
            # Capturar um quadro da câmera HAYEAR se nenhuma imagem foi passada
            print("[INFO] Nenhuma imagem informada. Abrindo câmera para captura do padrão de calibração...")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("[ERRO] Não foi possível acessar a câmera para calibração.")
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                print("[ERRO] Falha ao capturar imagem da câmera.")
                return None
            self.imagem_original = frame

        self.imagem_display = self.imagem_original.copy()
        self.pontos = []

        nome_janela = "Calibracao de Escala - EntomoTrack AI"
        cv2.namedWindow(nome_janela, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(nome_janela, self._mouse_callback)

        print("\n================ CALIBRAÇÃO DE ESCALA ================")
        print("1. Clique com o botão esquerdo no ponto inicial da referência conhecida.")
        print("2. Clique no ponto final da referência conhecida.")
        print("3. Pressione 'r' para reiniciar os pontos.")
        print("4. Pressione 'ENTER' ou 'ESPAÇO' para confirmar a distância.")
        print("5. Pressione 'ESC' para cancelar.")
        print("=====================================================\n")

        while True:
            cv2.imshow(nome_janela, self.imagem_display)
            tecla = cv2.waitKey(20) & 0xFF

            if tecla == 27:  # ESC
                print("[CALIBRAÇÃO] Operação cancelada pelo usuário.")
                cv2.destroyAllWindows()
                return None
            elif tecla == ord('r'):  # Reset
                self.pontos = []
                self.imagem_display = self.imagem_original.copy()
                print("[RESET] Pontos reiniciados. Clique novamente.")
            elif tecla in [13, 32]:  # ENTER ou ESPAÇO
                if len(self.pontos) == 2:
                    break
                else:
                    print("[AVISO] Selecione 2 pontos na imagem antes de confirmar.")

        cv2.destroyAllWindows()

        p1, p2 = self.pontos[0], self.pontos[1]
        distancia_px = float(np.linalg.norm(np.array(p1) - np.array(p2)))

        if distancia_px <= 0 or distancia_conhecida_mm <= 0:
            print("[ERRO] Distância em pixels ou mm inválida.")
            return None

        self.px_por_mm = distancia_px / distancia_conhecida_mm

        print(f"\n[SUCESSO] Calibração concluída:")
        print(f" -> Distância marcada: {distancia_px:.2f} pixels")
        print(f" -> Distância real: {distancia_conhecida_mm:.3f} mm")
        print(f" -> Escala calculada: {self.px_por_mm:.4f} pixels/mm")
        print(f" -> Resolução espacial: {(1.0 / self.px_por_mm) * 1000:.2f} micrômetros/pixel")

        # Salvar no banco de dados SQLite
        banco = BancoEntomoTrack()
        banco.salvar_calibracao(distancia_mm=distancia_conhecida_mm,
                               distancia_px=distancia_px,
                               observacoes=f"Calibrado via imagem: {Path(self.caminho_imagem).name if self.caminho_imagem else 'Camera ao vivo'}")

        # Atualizar arquivo config.yaml
        config_path = self.base_dir / "configuracoes" / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

                if "calibracao" not in cfg:
                    cfg["calibracao"] = {}
                cfg["calibracao"]["escala_ativa"] = True
                cfg["calibracao"]["pixels_por_milimetro"] = float(round(self.px_por_mm, 4))
                cfg["calibracao"]["distancia_referencia_mm"] = float(distancia_conhecida_mm)

                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
                print(f"[CONFIG] Escala salva no arquivo: {config_path}")
            except Exception as e:
                print(f"[AVISO] Falha ao atualizar config.yaml: {e}")

        return self.px_por_mm


if __name__ == "__main__":
    caminho = sys.argv[1] if len(sys.argv) > 1 else None
    dist_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    calibrador = CalibradorEscala(caminho_imagem=caminho)
    calibrador.calibrar(distancia_conhecida_mm=dist_mm)
