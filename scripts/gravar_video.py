"""
==============================================================================
ENTOMOTRACK AI - MÓDULO DE GRAVAÇÃO DE VÍDEO EM MICROSCOPIA
Gravação controlada com câmera HAYEAR, prevenção de corrupção e registro no banco.
==============================================================================
"""

import cv2
import sys
import time
from pathlib import Path
from typing import Optional

# Garantir import do diretório raiz
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.banco import BancoEntomoTrack


def gravar_video_camera(indice_camera: int = 0, largura: int = 1280, altura: int = 720,
                        fps_alvo: float = 30.0, duracao_max_segundos: Optional[float] = None,
                        nome_customizado: Optional[str] = None) -> Optional[str]:
    """
    Grava vídeo a partir da câmera digital HAYEAR USB com controle de FPS
    e gravação estável sem estouro de memória RAM.
    """
    base_dir = Path(__file__).resolve().parent.parent
    pasta_videos = base_dir / "videos"
    pasta_videos.mkdir(parents=True, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    if nome_customizado:
        caminho_video = pasta_videos / f"{nome_customizado}_{timestamp_str}.mp4"
    else:
        caminho_video = pasta_videos / f"experimento_{timestamp_str}.mp4"

    print(f"\n[INFO] Inicializando câmera {indice_camera} ({largura}x{altura} @ {fps_alvo:.0f} FPS)...")
    cap = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[ERRO] Câmera no índice {indice_camera} indisponível.")
        print("[DICA] Feche o aplicativo oficial da HAYEAR se estiver em execução.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    cap.set(cv2.CAP_PROP_FPS, fps_alvo)

    largura_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_sensor = cap.get(cv2.CAP_PROP_FPS)
    if fps_sensor <= 0 or fps_sensor > 120:
        fps_sensor = fps_alvo

    # Configuração do codec de vídeo (mp4v é universal e compatível nativamente no Windows)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(caminho_video), fourcc, fps_sensor, (largura_real, altura_real))

    if not out.isOpened():
        print("[ERRO] Falha ao inicializar o codificador de vídeo MP4.")
        cap.release()
        return None

    print(f"[GRAVANDO] Arquivo: {caminho_video}")
    print("[COMANDOS]")
    print(" -> 'p': Pausar / Continuar gravação")
    print(" -> 'q' ou 'ESC': Finalizar e salvar gravação\n")

    inicio_gravacao = time.time()
    total_quadros = 0
    em_pausa = False
    tempo_pausa_inicio = 0.0
    tempo_total_pausado = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            tempo_atual = time.time()
            tempo_decorrido = (tempo_atual - inicio_gravacao) - tempo_total_pausado

            if duracao_max_segundos and tempo_decorrido >= duracao_max_segundos:
                print(f"[INFO] Tempo máximo atingido: {duracao_max_segundos:.1f} s")
                break

            # Gravar o quadro apenas se não estiver em pausa
            if not em_pausa:
                out.write(frame)
                total_quadros += 1

            # Preparar overlay visual
            status_txt = "PAUSADO" if em_pausa else "GRAVANDO"
            cor_status = (0, 165, 255) if em_pausa else (0, 0, 255)

            display_frame = frame.copy()
            cv2.circle(display_frame, (25, 25), 8, cor_status, -1)
            cv2.putText(display_frame, f"{status_txt} | Tempo: {tempo_decorrido:.1f}s | Quadros: {total_quadros}",
                        (45, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display_frame, "Teclas: 'p' = Pausar/Retomar | 'q' = Salvar e Sair",
                        (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("EntomoTrack AI - Gravacao de Experimento", display_frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q') or tecla == 27:
                break
            elif tecla == ord('p'):
                em_pausa = not em_pausa
                if em_pausa:
                    tempo_pausa_inicio = time.time()
                    print("[PAUSA] Gravação pausada temporariamente.")
                else:
                    tempo_total_pausado += (time.time() - tempo_pausa_inicio)
                    print("[RETOMADO] Gravação em andamento.")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    duracao_final = time.time() - inicio_gravacao - tempo_total_pausado
    print(f"\n[CONCLUÍDO] Vídeo gravado com sucesso!")
    print(f" -> Local: {caminho_video}")
    print(f" -> Duração ativa: {duracao_final:.2f} s")
    print(f" -> Total de quadros: {total_quadros}")
    print(f" -> Resolução: {largura_real}x{altura_real}")

    # Registrar sessão no banco de dados SQLite
    try:
        banco = BancoEntomoTrack()
        with banco.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessoes_experimentais (tipo, caminho_arquivo, data_registro, resolucao, fps, duracao_segundos, observacoes)
                VALUES ('video', ?, ?, ?, ?, ?, 'Gravação via EntomoTrack AI')
            """, (str(caminho_video), time.strftime("%Y-%m-%d %H:%M:%S"),
                  f"{largura_real}x{altura_real}", float(fps_sensor), float(duracao_final)))
            conn.commit()
            print("[INFO] Sessão experimental registrada no banco de dados SQLite.")
    except Exception as e:
        print(f"[AVISO] Não foi possível registrar a sessão no banco: {e}")

    return str(caminho_video)


if __name__ == "__main__":
    gravar_video_camera(indice_camera=0, largura=1280, altura=720)
