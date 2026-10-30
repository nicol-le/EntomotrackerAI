"""
==============================================================================
ENTOMOTRACK AI - MÓDULO DE TESTE E DIAGNÓSTICO DE CÂMERA
Verificação de câmeras USB (HAYEAR), enumeração de dispositivos e preview.
==============================================================================
"""

import cv2
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Garantir import do diretório raiz
sys.path.append(str(Path(__file__).resolve().parent.parent))


def listar_cameras_disponiveis(max_indices: int = 5) -> List[Dict[str, Any]]:
    """
    Enumera portas de câmera disponíveis no Windows usando cv2.CAP_DSHOW.
    Evita travamentos e testa abertura individualmente.
    """
    cameras_detectadas = []
    print("[INFO] Buscando dispositivos de captura conectados via USB...")

    for idx in range(max_indices):
        # Utiliza CAP_DSHOW para compatibilidade direta com DirectShow no Windows
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or fps > 120:
                    fps = 30.0

                backend_name = cap.getBackendName()
                info = {
                    "indice": idx,
                    "largura": largura,
                    "altura": altura,
                    "fps": fps,
                    "backend": backend_name,
                    "possivel_hayear": True  # Câmera USB funcional
                }
                cameras_detectadas.append(info)
                print(f"[OK] Câmera encontrada no Índice {idx}: {largura}x{altura} @ {fps:.1f} FPS (Backend: {backend_name})")
            cap.release()
        else:
            # Não é erro, apenas índice não conectado
            pass

    return cameras_detectadas


def testar_preview_camera(indice: int = 0, largura: int = 1280, altura: int = 720) -> bool:
    """
    Abre o preview em tempo real da câmera selecionada.
    Permite capturar foto com tecla 's' e sair com tecla 'q'.
    """
    base_dir = Path(__file__).resolve().parent.parent
    pasta_fotos = base_dir / "imagens"
    pasta_fotos.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Conectando à câmera {indice} via DirectShow ({largura}x{altura})...")
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[ERRO CRÍTICO] Falha ao abrir a câmera no índice {indice}.")
        print("[DICA] Verifique se o cabo USB está conectado ou se outro programa (ex: software HAYEAR oficial) está utilizando a câmera.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

    largura_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[CONECTADO] Resolução ativa: {largura_real}x{altura_real}")
    print("[COMANDOS NA JANELA DE VÍDEO]")
    print(" -> Pressione 's' para salvar uma foto de teste")
    print(" -> Pressione 'q' ou 'ESC' para fechar a visualização\n")

    tempo_anterior = time.time()
    contador_frames = 0
    fps_exibido = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[AVISO] Perda temporária de sinal do sensor de imagem...")
                time.sleep(0.05)
                continue

            contador_frames += 1
            tempo_atual = time.time()
            delta = tempo_atual - tempo_anterior
            if delta >= 1.0:
                fps_exibido = contador_frames / delta
                contador_frames = 0
                tempo_anterior = tempo_atual

            # Adicionar painel informativo na tela
            cv2.putText(frame, f"EntomoTrack AI - Cam {indice} ({largura_real}x{altura_real}) | FPS: {fps_exibido:.1f}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Pressione 's': Foto | 'q': Sair",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("EntomoTrack AI - Teste de Camera HAYEAR", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q') or tecla == 27:
                break
            elif tecla == ord('s'):
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                nome_foto = pasta_fotos / f"captura_teste_{timestamp_str}.png"
                cv2.imwrite(str(nome_foto), frame)
                print(f"[SALVO] Fotografia de teste salva em: {nome_foto}")

        return True
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Conexão com a câmera liberada com sucesso.")


if __name__ == "__main__":
    dispositivos = listar_cameras_disponiveis()
    if not dispositivos:
        print("\n[ALERTA] Nenhuma câmera USB detectada. Conecte a câmera HAYEAR e tente novamente.")
    else:
        print(f"\n[SUCESSO] Total de {len(dispositivos)} dispositivo(s) encontrado(s).")
        indice_escolhido = dispositivos[0]["indice"]
        print(f"[INFO] Iniciando pré-visualização no dispositivo {indice_escolhido}...")
        testar_preview_camera(indice=indice_escolhido)
