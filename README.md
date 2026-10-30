# EntomoTrack AI — Rastreamento Automático de Insetos em Microscopia

Sistema completo de visão computacional e inteligência artificial para monitoramento, detecção e rastreamento multiobjeto de múltiplos insetos simultaneamente em imagens e vídeos capturados por microscópio estereoscópico BiFocus com câmera digital HAYEAR USB.

Otimizado para execução em **Windows 10/11**, com **8 GB de RAM** e **CPU convencional** (sem necessidade de GPU dedicada).

Consulte o guia completo de instalação para computadores novos em:
👉 **[`INSTRUCOES_NOVO_PC.md`](./INSTRUCOES_NOVO_PC.md)**

---

## Como Iniciar Rapidamente
Dê um duplo clique no arquivo:
```text
iniciar_entomotrack.bat
```
Ou execute via terminal PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe interface.py
```

---

## Módulos Principais
- `scripts/teste_camera.py`: Enumeração e pré-visualização da câmera digital HAYEAR USB.
- `scripts/gravar_video.py`: Gravação estável de ensaios em microscopia.
- `scripts/calibrar_escala.py`: Calibração espacial interativa de pixels por milímetro ($px/mm$).
- `scripts/detectar.py`: Detector YOLO em CPU com suporte a Situações A (Confiável), B (Incerta) e C (Não Reconhecido).
- `scripts/rastrear.py`: Rastreador multiobjeto ByteTrack com persistência de ID.
- `scripts/analisar_movimento.py`: Cálculo de parâmetros cinemáticos, mapas de calor e relatórios.
- `scripts/treinar_modelo.py`: Fine-tuning em CPU com divisão de dataset sem vazamento de dados (*data leakage*).
- `scripts/revisao_manual.py`: Ciclo *Human-in-the-Loop* com banco relacional SQLite e histórico de auditoria.
- `interface.py`: Interface gráfica em Tkinter desacoplada em threads.
- `testes/test_suite.py`: Suíte de validação automatizada de 12 etapas.
