"""
==============================================================================
ENTOMOTRACK AI - INTERFACE GRÁFICA PRINCIPAL (TKINTER OTIMIZADO)
Aquisição com HAYEAR, Rastreamento ByteTrack, Análise Comportamental e Human-in-the-Loop.
==============================================================================
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".yolo"))
(BASE_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
(BASE_DIR / ".yolo").mkdir(parents=True, exist_ok=True)

import gc
import cv2
import sys
import yaml
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import pandas as pd
from typing import Optional, Dict, Any, List
sys.path.append(str(BASE_DIR))

from scripts.banco import BancoEntomoTrack
from scripts.teste_camera import listar_cameras_disponiveis
from scripts.detectar import DetectorInsetosYOLO
from scripts.rastrear import rastrear_video
from scripts.analisar_movimento import AnalisadorComportamental
from scripts.calibrar_escala import CalibradorEscala
from scripts.treinar_modelo import TreinadorYOLO
from scripts.revisao_manual import GerenciadorRevisaoManual


class EntomoTrackApp(tk.Tk):
    """Aplicação Principal EntomoTrack AI com Tkinter e execução segura em background."""

    def __init__(self):
        super().__init__()
        self.title("EntomoTrack AI — Rastreamento e Análise de Insetos em Microscopia")
        self.geometry("1120x760")
        self.minsize(980, 650)

        # Configurar tema limpo
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Instâncias de módulos
        self.banco = BancoEntomoTrack()
        self.revisao_mgr = GerenciadorRevisaoManual()
        self.config_path = BASE_DIR / "configuracoes" / "config.yaml"
        self.cfg = self.carregar_configuracao()

        # Controle de câmera
        self.cap_preview: Optional[cv2.VideoCapture] = None
        self.preview_ativo = False
        self.gravacao_ativa = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.caminho_video_gravando = None

        # Controle de rastreamento
        self.rastreamento_ativo = False
        self.cancelar_rastreamento = False
        self.thread_rastreamento: Optional[threading.Thread] = None

        # Imagem de preview temporária
        self.preview_img_tk = None
        self.crop_img_tk = None

        # Inicializar Interface
        self.criar_widgets()
        self.protocol("WM_DELETE_WINDOW", self.ao_fechar)

    def carregar_configuracao(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}

    def criar_widgets(self):
        # Barra superior com título
        header_frame = tk.Frame(self, bg="#1E293B", height=50)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        lbl_titulo = tk.Label(header_frame, text="🔬 EntomoTrack AI — Monitoramento e Rastreamento de Insetos",
                              font=("Segoe UI", 13, "bold"), fg="#F8FAFC", bg="#1E293B")
        lbl_titulo.pack(side=tk.LEFT, padx=15, pady=8)

        lbl_hardware = tk.Label(header_frame, text="Modo: CPU Convencional | 8 GB RAM Otimizado",
                                font=("Segoe UI", 9), fg="#94A3B8", bg="#1E293B")
        lbl_hardware.pack(side=tk.RIGHT, padx=15, pady=8)

        # Abas do sistema
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.aba_camera = ttk.Frame(self.notebook)
        self.aba_rastreio = ttk.Frame(self.notebook)
        self.aba_revisao = ttk.Frame(self.notebook)
        self.aba_analise = ttk.Frame(self.notebook)

        self.notebook.add(self.aba_camera, text="  1. Câmera HAYEAR & Gravação  ")
        self.notebook.add(self.aba_rastreio, text="  2. Rastreamento & Detecção  ")
        self.notebook.add(self.aba_revisao, text="  3. Revisão & Treinamento (Human-in-the-Loop)  ")
        self.notebook.add(self.aba_analise, text="  4. Calibração & Resultados  ")

        self.montar_aba_camera()
        self.montar_aba_rastreio()
        self.montar_aba_revisao()
        self.montar_aba_analise()

        # Barra de status inferior
        self.status_var = tk.StringVar(value="Sistema pronto. Conecte a câmera ou selecione um vídeo gravado.")
        status_bar = tk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W,
                              font=("Segoe UI", 9), bg="#F1F5F9", fg="#334155", padx=10, pady=4)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # =========================================================================
    # ABA 1: CÂMERA HAYEAR & GRAVAÇÃO
    # =========================================================================

    def montar_aba_camera(self):
        painel_controles = ttk.LabelFrame(self.aba_camera, text="Controles da Câmera USB", padding=10)
        painel_controles.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        ttk.Button(painel_controles, text="🔍 Detectar Câmeras USB", command=self.detectar_cameras).pack(fill=tk.X, pady=4)

        ttk.Label(painel_controles, text="Dispositivo:").pack(anchor=tk.W, pady=(8, 2))
        self.combo_cameras = ttk.Combobox(painel_controles, values=["Câmera 0 (HAYEAR)"], state="readonly")
        self.combo_cameras.current(0)
        self.combo_cameras.pack(fill=tk.X, pady=2)

        ttk.Label(painel_controles, text="Resolução de Captura:").pack(anchor=tk.W, pady=(8, 2))
        self.combo_res = ttk.Combobox(painel_controles, values=["1280x720 (Recomendado)", "1920x1080", "640x480"], state="readonly")
        self.combo_res.current(0)
        self.combo_res.pack(fill=tk.X, pady=2)

        ttk.Separator(painel_controles, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        self.btn_preview = ttk.Button(painel_controles, text="▶ Iniciar Pré-Visualização", command=self.alternar_preview)
        self.btn_preview.pack(fill=tk.X, pady=4)

        self.btn_foto = ttk.Button(painel_controles, text="📸 Capturar Foto", command=self.capturar_foto, state=tk.DISABLED)
        self.btn_foto.pack(fill=tk.X, pady=4)

        self.btn_gravar = ttk.Button(painel_controles, text="⏺ Iniciar Gravação", command=self.alternar_gravacao, state=tk.DISABLED)
        self.btn_gravar.pack(fill=tk.X, pady=4)

        self.lbl_tempo_rec = ttk.Label(painel_controles, text="Duração: 00:00", font=("Segoe UI", 10, "bold"), foreground="red")
        self.lbl_tempo_rec.pack(pady=6)

        # Painel central de visualização
        painel_visao = ttk.LabelFrame(self.aba_camera, text="Imagem do Microscópio BiFocus", padding=5)
        painel_visao.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.canvas_cam = tk.Canvas(painel_visao, bg="#0F172A", highlightthickness=0)
        self.canvas_cam.pack(fill=tk.BOTH, expand=True)

    def detectar_cameras(self):
        self.status_var.set("Buscando dispositivos USB...")
        cams = listar_cameras_disponiveis()
        if cams:
            lista = [f"Índice {c['indice']}: {c['largura']}x{c['altura']} @ {c['fps']:.0f} FPS" for c in cams]
            self.combo_cameras["values"] = lista
            self.combo_cameras.current(0)
            messagebox.showinfo("Câmeras Detectadas", f"Encontrada(s) {len(cams)} câmera(s) USB disponível(is).")
        else:
            messagebox.showwarning("Aviso", "Nenhuma câmera USB detectada. Conecte a câmera HAYEAR.")

    def alternar_preview(self):
        if not self.preview_ativo:
            idx = self.combo_cameras.current()
            if idx < 0:
                idx = 0
            res_str = self.combo_res.get()
            w, h = (1280, 720)
            if "1920" in res_str:
                w, h = (1920, 1080)
            elif "640" in res_str:
                w, h = (640, 480)

            self.cap_preview = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not self.cap_preview.isOpened():
                messagebox.showerror("Erro de Acesso", f"Não foi possível abrir a câmera {idx}.\nVerifique se outro programa (ex: HAYEAR oficial) está utilizando a câmera.")
                return

            self.cap_preview.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap_preview.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

            self.preview_ativo = True
            self.btn_preview.config(text="⏹ Parar Pré-Visualização")
            self.btn_foto.config(state=tk.NORMAL)
            self.btn_gravar.config(state=tk.NORMAL)
            self.status_var.set("Câmera conectada. Exibindo imagem em tempo real.")

            # Iniciar loop de atualização no Canvas
            self.atualizar_quadro_camera()
        else:
            self.parar_preview()

    def parar_preview(self):
        self.preview_ativo = False
        if self.gravacao_ativa:
            self.alternar_gravacao()
        if self.cap_preview:
            self.cap_preview.release()
            self.cap_preview = None

        self.btn_preview.config(text="▶ Iniciar Pré-Visualização")
        self.btn_foto.config(state=tk.DISABLED)
        self.btn_gravar.config(state=tk.DISABLED)
        self.canvas_cam.delete("all")
        self.status_var.set("Câmera desconectada.")

    def atualizar_quadro_camera(self):
        if not self.preview_ativo or self.cap_preview is None:
            return

        ret, frame = self.cap_preview.read()
        if ret and frame is not None:
            if self.gravacao_ativa and self.video_writer:
                self.video_writer.write(frame)

            # Redimensionar para o Canvas de forma proporcional
            c_w = self.canvas_cam.winfo_width()
            c_h = self.canvas_cam.winfo_height()
            if c_w > 50 and c_h > 50:
                f_h, f_w = frame.shape[:2]
                escala = min(c_w / f_w, c_h / f_h)
                nw, nh = int(f_w * escala), int(f_h * escala)
                frame_res = cv2.resize(frame, (nw, nh))
                frame_rgb = cv2.cvtColor(frame_res, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                self.preview_img_tk = ImageTk.PhotoImage(image=img_pil)

                self.canvas_cam.delete("all")
                self.canvas_cam.create_image(c_w // 2, c_h // 2, anchor=tk.CENTER, image=self.preview_img_tk)

        # Agendar próximo quadro respeitando limite de FPS
        if self.preview_ativo:
            self.after(33, self.atualizar_quadro_camera)

    def capturar_foto(self):
        if self.cap_preview and self.cap_preview.isOpened():
            ret, frame = self.cap_preview.read()
            if ret and frame is not None:
                pasta = BASE_DIR / "imagens"
                pasta.mkdir(parents=True, exist_ok=True)
                nome = pasta / f"foto_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(nome), frame)
                self.status_var.set(f"Fotografia salva: {nome.name}")
                messagebox.showinfo("Foto Capturada", f"Imagem salva com sucesso em:\n{nome}")

    def alternar_gravacao(self):
        if not self.gravacao_ativa:
            if not self.cap_preview:
                return
            pasta = BASE_DIR / "videos"
            pasta.mkdir(parents=True, exist_ok=True)
            self.caminho_video_gravando = pasta / f"experimento_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

            w = int(self.cap_preview.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap_preview.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(self.caminho_video_gravando), fourcc, 30.0, (w, h))

            if not self.video_writer.isOpened():
                messagebox.showerror("Erro", "Não foi possível inicializar o codificador de vídeo.")
                return

            self.gravacao_ativa = True
            self.btn_gravar.config(text="⏹ Finalizar Gravação")
            self.status_var.set("Gravando vídeo do experimento...")
        else:
            self.gravacao_ativa = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.btn_gravar.config(text="⏺ Iniciar Gravação")
            self.status_var.set(f"Vídeo gravado: {self.caminho_video_gravando.name if self.caminho_video_gravando else ''}")
            if self.caminho_video_gravando:
                messagebox.showinfo("Vídeo Concluído", f"Vídeo salvo com sucesso em:\n{self.caminho_video_gravando}")

    # =========================================================================
    # ABA 2: RASTREAMENTO & DETECÇÃO
    # =========================================================================

    def montar_aba_rastreio(self):
        painel_config = ttk.LabelFrame(self.aba_rastreio, text="Parâmetros de Rastreamento (Otimizado CPU)", padding=10)
        painel_config.pack(fill=tk.X, padx=8, pady=8)

        # Seleção de Vídeo
        f_vid = ttk.Frame(painel_config)
        f_vid.pack(fill=tk.X, pady=4)
        ttk.Label(f_vid, text="Vídeo de Entrada:", width=18).pack(side=tk.LEFT)
        self.entry_video = ttk.Entry(f_vid)
        self.entry_video.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(f_vid, text="Buscar Vídeo...", command=self.selecionar_video_arquivo).pack(side=tk.RIGHT)

        # Seleção de Modelo
        f_mod = ttk.Frame(painel_config)
        f_mod.pack(fill=tk.X, pady=4)
        ttk.Label(f_mod, text="Modelo YOLO (.pt):", width=18).pack(side=tk.LEFT)
        self.entry_modelo = ttk.Entry(f_mod)
        self.entry_modelo.insert(0, "yolov8n.pt")
        self.entry_modelo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(f_mod, text="Buscar Pesos...", command=self.selecionar_modelo_arquivo).pack(side=tk.RIGHT)

        # Sliders de ajuste CPU
        f_sliders = ttk.Frame(painel_config)
        f_sliders.pack(fill=tk.X, pady=8)

        ttk.Label(f_sliders, text="Confiança Mínima:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.scale_conf = ttk.Scale(f_sliders, from_=0.15, to=0.90, value=0.35)
        self.scale_conf.grid(row=0, column=1, sticky=tk.EW, padx=5)

        ttk.Label(f_sliders, text="Quadros Alternados (Pulo):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.spin_pulo = ttk.Spinbox(f_sliders, from_=1, to=10, width=5)
        self.spin_pulo.set("2")
        self.spin_pulo.grid(row=0, column=3, sticky=tk.W, padx=5)

        f_sliders.columnconfigure(1, weight=1)

        # Botões de Ação
        f_botoes = ttk.Frame(painel_config)
        f_botoes.pack(fill=tk.X, pady=6)

        self.btn_iniciar_rastreio = ttk.Button(f_botoes, text="🚀 Iniciar Rastreamento ByteTrack", command=self.iniciar_rastreamento_thread)
        self.btn_iniciar_rastreio.pack(side=tk.LEFT, padx=5)

        self.btn_parar_rastreio = ttk.Button(f_botoes, text="⏹ Interromper", command=self.interromper_rastreamento, state=tk.DISABLED)
        self.btn_parar_rastreio.pack(side=tk.LEFT, padx=5)

        self.btn_abrir_resultados = ttk.Button(f_botoes, text="📂 Abrir Pasta de Resultados", command=self.abrir_pasta_resultados)
        self.btn_abrir_resultados.pack(side=tk.RIGHT, padx=5)

        # Barra de progresso
        self.progresso_var = tk.DoubleVar()
        self.barra_progresso = ttk.Progressbar(painel_config, variable=self.progresso_var, maximum=100)
        self.barra_progresso.pack(fill=tk.X, pady=8)

        # Log de eventos de rastreamento
        painel_log = ttk.LabelFrame(self.aba_rastreio, text="Registro do Processamento", padding=8)
        painel_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.txt_log = tk.Text(painel_log, bg="#0F172A", fg="#F8FAFC", font=("Consolas", 9), height=12)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def log_msg(self, mensagem: str):
        self.txt_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {mensagem}\n")
        self.txt_log.see(tk.END)

    def selecionar_video_arquivo(self):
        caminho = filedialog.askopenfilename(
            title="Selecione o vídeo do experimento",
            filetypes=[("Arquivos de Vídeo", "*.mp4;*.avi;*.mov;*.mkv"), ("Todos os Arquivos", "*.*")]
        )
        if caminho:
            self.entry_video.delete(0, tk.END)
            self.entry_video.insert(0, caminho)

    def selecionar_modelo_arquivo(self):
        caminho = filedialog.askopenfilename(
            title="Selecione os pesos do modelo YOLO",
            filetypes=[("Pesos PyTorch YOLO", "*.pt"), ("Todos os Arquivos", "*.*")]
        )
        if caminho:
            self.entry_modelo.delete(0, tk.END)
            self.entry_modelo.insert(0, caminho)

    def iniciar_rastreamento_thread(self):
        caminho_vid = self.entry_video.get().strip()
        if not caminho_vid or not Path(caminho_vid).exists():
            messagebox.showwarning("Aviso", "Selecione um arquivo de vídeo válido antes de iniciar.")
            return

        self.rastreamento_ativo = True
        self.cancelar_rastreamento = False
        self.btn_iniciar_rastreio.config(state=tk.DISABLED)
        self.btn_parar_rastreio.config(state=tk.NORMAL)
        self.progresso_var.set(0)

        self.thread_rastreamento = threading.Thread(target=self._executar_rastreamento_worker, daemon=True)
        self.thread_rastreamento.start()

    def _executar_rastreamento_worker(self):
        caminho_vid = self.entry_video.get().strip()
        caminho_mod = self.entry_modelo.get().strip()
        conf_min = float(self.scale_conf.get())
        try:
            pulo = int(self.spin_pulo.get())
        except Exception:
            pulo = 2

        self.log_msg(f"Iniciando rastreamento multiobjeto em CPU...")
        self.log_msg(f"Vídeo: {Path(caminho_vid).name} | Modelo: {Path(caminho_mod).name}")
        self.log_msg(f"Confiança: {conf_min:.2f} | Salto de quadros: {pulo}")

        try:
            detector = DetectorInsetosYOLO(caminho_modelo=caminho_mod, conf_min=conf_min)

            def atualizar_progresso(frac):
                self.progresso_var.set(frac * 100)
                self.status_var.set(f"Rastreando vídeo: {frac * 100:.1f}% concluído")

            def deve_cancelar():
                return self.cancelar_rastreamento

            res_rastreio = rastrear_video(
                caminho_video=caminho_vid,
                detector=detector,
                pular_quadros=pulo,
                callback_progresso=atualizar_progresso,
                flag_cancelar=deve_cancelar
            )

            if not self.cancelar_rastreamento:
                self.log_msg(f"Rastreamento concluído com sucesso!")
                self.log_msg(f"Total de insetos únicos identificados: {res_rastreio['total_insetos_rastreados']}")
                self.log_msg(f"Vídeo rastreado salvo em: {res_rastreio['video_anotado']}")

                # Executar análise comportamental automaticamente
                self.log_msg("Processando métricas comportamentais, gráficos e mapas de calor...")
                analisador = AnalisadorComportamental()
                analisador.analisar_trajetorias(
                    df_coords=res_rastreio["dataframe"],
                    pasta_saida=res_rastreio["pasta_resultados"],
                    largura_quadro=res_rastreio["largura"],
                    altura_quadro=res_rastreio["altura"]
                )
                self.log_msg(f"Todos os resultados foram consolidados na pasta:\n{res_rastreio['pasta_resultados']}")
                self.status_var.set("Análise finalizada com sucesso! Resultados prontos.")
                self.pasta_ultimos_resultados = res_rastreio["pasta_resultados"]
                messagebox.showinfo("Sucesso", f"Análise comportamental concluída!\n\nResultados salvos em:\n{res_rastreio['pasta_resultados']}")
            else:
                self.log_msg("Processamento cancelado pelo operador.")
                self.status_var.set("Rastreamento interrompido.")

        except Exception as e:
            self.log_msg(f"ERRO durante o processamento: {e}")
            messagebox.showerror("Erro de Processamento", f"Ocorreu uma falha durante o rastreamento:\n{e}")
        finally:
            self.rastreamento_ativo = False
            self.btn_iniciar_rastreio.config(state=tk.NORMAL)
            self.btn_parar_rastreio.config(state=tk.DISABLED)

    def interromper_rastreamento(self):
        self.cancelar_rastreamento = True
        self.log_msg("Solicitação de cancelamento enviada. Finalizando quadro atual...")

    def abrir_pasta_resultados(self):
        pasta = getattr(self, 'pasta_ultimos_resultados', None)
        if not pasta or not Path(pasta).exists():
            pasta = BASE_DIR / "resultados"
            pasta.mkdir(parents=True, exist_ok=True)
        os.startfile(str(pasta))

    # =========================================================================
    # ABA 3: REVISÃO & TREINAMENTO (HUMAN-IN-THE-LOOP)
    # =========================================================================

    def montar_aba_revisao(self):
        # Painel esquerdo: Tabela de pendentes
        painel_esquerdo = ttk.LabelFrame(self.aba_revisao, text="Insetos Pendentes de Identificação", padding=8)
        painel_esquerdo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        cols = ("id", "arquivo", "frame", "conf", "status", "especie")
        self.tree_pendentes = ttk.Treeview(painel_esquerdo, columns=cols, show="headings", selectmode="browse")
        self.tree_pendentes.heading("id", text="ID")
        self.tree_pendentes.heading("arquivo", text="Origem")
        self.tree_pendentes.heading("frame", text="Quadro")
        self.tree_pendentes.heading("conf", text="Confiança")
        self.tree_pendentes.heading("status", text="Situação")
        self.tree_pendentes.heading("especie", text="Espécie Atribuída")

        self.tree_pendentes.column("id", width=40)
        self.tree_pendentes.column("arquivo", width=120)
        self.tree_pendentes.column("frame", width=60)
        self.tree_pendentes.column("conf", width=70)
        self.tree_pendentes.column("status", width=90)
        self.tree_pendentes.column("especie", width=120)

        scroll_tree = ttk.Scrollbar(painel_esquerdo, orient=tk.VERTICAL, command=self.tree_pendentes.yview)
        self.tree_pendentes.configure(yscroll=scroll_tree.set)
        self.tree_pendentes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_tree.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_pendentes.bind("<<TreeviewSelect>>", self.ao_selecionar_pendente)

        # Painel direito: Visualização e Correção
        painel_direito = ttk.LabelFrame(self.aba_revisao, text="Painel de Decisão Científica (Human-in-the-Loop)", padding=8)
        painel_direito.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        # Canvas para exibir recorte do inseto
        self.canvas_crop = tk.Canvas(painel_direito, bg="#1E293B", width=220, height=220)
        self.canvas_crop.pack(pady=4)

        ttk.Label(painel_direito, text="Atribuir Espécie / Grupo:").pack(anchor=tk.W, pady=(8, 2))
        self.combo_especies_rev = ttk.Combobox(painel_direito, state="readonly", width=28)
        self.combo_especies_rev.pack(fill=tk.X, pady=2)
        self.atualizar_combobox_especies()

        ttk.Button(painel_direito, text="➕ Cadastrar Nova Espécie", command=self.janela_cadastrar_especie).pack(fill=tk.X, pady=4)

        ttk.Separator(painel_direito, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(painel_direito, text="✅ Confirmar Espécie", command=self.confirmar_revisao).pack(fill=tk.X, pady=3)
        ttk.Button(painel_direito, text="❌ Falso Positivo (Sujeira/Sombra)", command=self.marcar_falso_positivo).pack(fill=tk.X, pady=3)
        ttk.Button(painel_direito, text="🗑 Excluir Anotação", command=self.excluir_anotacao).pack(fill=tk.X, pady=3)

        ttk.Separator(painel_direito, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(painel_direito, text="📦 Exportar Dataset YOLO", command=self.exportar_dataset_yolo).pack(fill=tk.X, pady=4)
        ttk.Button(painel_direito, text="🧠 Treinar Modelo em CPU", command=self.iniciar_treinamento_cpu).pack(fill=tk.X, pady=4)

        ttk.Button(painel_direito, text="🔄 Atualizar Lista", command=self.carregar_pendentes_tabela).pack(fill=tk.X, pady=8)

        self.carregar_pendentes_tabela()

    def atualizar_combobox_especies(self):
        especies = self.banco.listar_especies()
        valores = [f"{e['id']} - {e['nome_cientifico']} ({e['codigo']})" for e in especies]
        self.combo_especies_rev["values"] = valores
        if valores:
            self.combo_especies_rev.current(0)

    def carregar_pendentes_tabela(self):
        for item in self.tree_pendentes.get_children():
            self.tree_pendentes.delete(item)

        pendentes = self.banco.listar_pendentes_revisao(limite=200)
        for p in pendentes:
            self.tree_pendentes.insert("", tk.END, values=(
                p["id"],
                Path(p["origem_arquivo"]).name,
                p["frame_idx"],
                f"{p['confianca_detector']:.2f}" if p['confianca_detector'] else "N/A",
                p["status_revisao"],
                p["nome_cientifico"] or "Não Atribuído"
            ))

    def ao_selecionar_pendente(self, event):
        sel = self.tree_pendentes.selection()
        if not sel:
            return
        item = self.tree_pendentes.item(sel[0])
        ident_id = item["values"][0]

        with self.banco.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT caminho_recorte FROM identificacoes WHERE id = ?", (ident_id,))
            row = cursor.fetchone()

        if row and row["caminho_recorte"] and Path(row["caminho_recorte"]).exists():
            img = cv2.imread(row["caminho_recorte"])
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb).resize((200, 200), Image.Resampling.LANCZOS)
                self.crop_img_tk = ImageTk.PhotoImage(image=img_pil)
                self.canvas_crop.delete("all")
                self.canvas_crop.create_image(110, 110, anchor=tk.CENTER, image=self.crop_img_tk)

    def janela_cadastrar_especie(self):
        janela = tk.Toplevel(self)
        janela.title("Cadastro Taxonômico de Espécie")
        janela.geometry("400x380")
        janela.transient(self)
        janela.grab_set()

        ttk.Label(janela, text="Código Único (ex: SP_001):").pack(anchor=tk.W, padx=15, pady=(10, 2))
        e_cod = ttk.Entry(janela)
        e_cod.pack(fill=tk.X, padx=15)

        ttk.Label(janela, text="Nome Científico:").pack(anchor=tk.W, padx=15, pady=(8, 2))
        e_cient = ttk.Entry(janela)
        e_cient.pack(fill=tk.X, padx=15)

        ttk.Label(janela, text="Nome Popular / Grupo:").pack(anchor=tk.W, padx=15, pady=(8, 2))
        e_pop = ttk.Entry(janela)
        e_pop.pack(fill=tk.X, padx=15)

        ttk.Label(janela, text="Ordem Taxonômica:").pack(anchor=tk.W, padx=15, pady=(8, 2))
        e_ord = ttk.Entry(janela)
        e_ord.pack(fill=tk.X, padx=15)

        ttk.Label(janela, text="Observações Biológicas:").pack(anchor=tk.W, padx=15, pady=(8, 2))
        e_obs = ttk.Entry(janela)
        e_obs.pack(fill=tk.X, padx=15)

        def salvar():
            cod = e_cod.get().strip()
            cient = e_cient.get().strip()
            if not cod or not cient:
                messagebox.showerror("Erro", "Código e Nome Científico são obrigatórios.")
                return
            try:
                self.banco.cadastrar_especie(
                    codigo=cod,
                    nome_cientifico=cient,
                    nome_popular=e_pop.get().strip(),
                    ordem=e_ord.get().strip(),
                    observacoes=e_obs.get().strip()
                )
                self.atualizar_combobox_especies()
                messagebox.showinfo("Sucesso", f"Espécie '{cient}' cadastrada!")
                janela.destroy()
            except Exception as ex:
                messagebox.showerror("Erro", f"Falha ao cadastrar: {ex}")

        ttk.Button(janela, text="💾 Salvar Espécie", command=salvar).pack(pady=15)

    def confirmar_revisao(self):
        sel = self.tree_pendentes.selection()
        if not sel:
            messagebox.showwarning("Aviso", "Selecione uma detecção na tabela.")
            return
        ident_id = self.tree_pendentes.item(sel[0])["values"][0]
        esp_str = self.combo_especies_rev.get()
        if not esp_str:
            return
        esp_id = int(esp_str.split(" - ")[0])

        self.revisao_mgr.confirmar_especie(ident_id, esp_id)
        self.carregar_pendentes_tabela()
        self.status_var.set(f"Detecção #{ident_id} confirmada com sucesso.")

    def marcar_falso_positivo(self):
        sel = self.tree_pendentes.selection()
        if not sel:
            return
        ident_id = self.tree_pendentes.item(sel[0])["values"][0]
        self.revisao_mgr.marcar_falso_positivo(ident_id)
        self.carregar_pendentes_tabela()
        self.status_var.set(f"Detecção #{ident_id} marcada como falso positivo.")

    def excluir_anotacao(self):
        sel = self.tree_pendentes.selection()
        if not sel:
            return
        ident_id = self.tree_pendentes.item(sel[0])["values"][0]
        if messagebox.askyesno("Confirmar Exclusão", f"Deseja excluir a anotação #{ident_id}?\nO vídeo original será preservado."):
            self.revisao_mgr.excluir_anotacao(ident_id, definitiva=False)
            self.carregar_pendentes_tabela()
            self.status_var.set(f"Anotação #{ident_id} excluída logicamente.")

    def exportar_dataset_yolo(self):
        try:
            treinador = TreinadorYOLO()
            yaml_path, classes = treinador.preparar_dataset_yolo()
            messagebox.showinfo("Dataset Gerado", f"Dataset YOLO gerado com sucesso!\n\nArquivo de configuração:\n{yaml_path}\nClasses: {len(classes)}")
        except Exception as e:
            messagebox.showerror("Erro ao Gerar Dataset", str(e))

    def iniciar_treinamento_cpu(self):
        if not messagebox.askyesno("Confirmar Treinamento", "Deseja iniciar o fine-tuning do modelo YOLO em CPU?\nO processo usará 4 threads e respeitará a RAM."):
            return

        def worker():
            try:
                self.status_var.set("Preparando dataset e iniciando treinamento na CPU...")
                treinador = TreinadorYOLO()
                yaml_path, _ = treinador.preparar_dataset_yolo()
                res = treinador.treinar_modelo_cpu(data_yaml=yaml_path, epocas=15, batch_size=4, img_size=416)
                self.status_var.set(f"Treinamento concluído: {res['nome_versao']} (mAP50: {res['map50']:.3f})")
                messagebox.showinfo("Treino Concluído", f"Novo modelo treinado salvo em:\n{res['caminho_pesos']}")
            except Exception as e:
                self.status_var.set("Falha no treinamento.")
                messagebox.showerror("Erro de Treinamento", str(e))

        threading.Thread(target=worker, daemon=True).start()

    # =========================================================================
    # ABA 4: CALIBRAÇÃO & RESULTADOS
    # =========================================================================

    def montar_aba_analise(self):
        painel_calib = ttk.LabelFrame(self.aba_analise, text="Calibração Espacial de Escala (Milímetros Reais)", padding=10)
        painel_calib.pack(fill=tk.X, padx=8, pady=8)

        f_cal = ttk.Frame(painel_calib)
        f_cal.pack(fill=tk.X, pady=4)

        ttk.Label(f_cal, text="Distância Conhecida de Referência (mm):").pack(side=tk.LEFT, padx=5)
        self.entry_dist_mm = ttk.Entry(f_cal, width=8)
        self.entry_dist_mm.insert(0, "1.0")
        self.entry_dist_mm.pack(side=tk.LEFT, padx=5)

        ttk.Button(f_cal, text="📐 Calibrar Interativamente", command=self.iniciar_calibracao_janela).pack(side=tk.LEFT, padx=10)

        self.lbl_status_escala = ttk.Label(f_cal, text="Status: Carregando calibração...", font=("Segoe UI", 9, "bold"))
        self.lbl_status_escala.pack(side=tk.RIGHT, padx=10)
        self.atualizar_label_escala()

        # Visualização de gráficos gerados
        painel_graficos = ttk.LabelFrame(self.aba_analise, text="Galeria de Gráficos e Mapas Comportamentais", padding=8)
        painel_graficos.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        f_botoes_g = ttk.Frame(painel_graficos)
        f_botoes_g.pack(fill=tk.X, pady=4)

        ttk.Button(f_botoes_g, text="📈 Trajetórias 2D", command=lambda: self.carregar_grafico_galeria("trajetorias_2d.png")).pack(side=tk.LEFT, padx=4)
        ttk.Button(f_botoes_g, text="🔥 Mapa de Calor", command=lambda: self.carregar_grafico_galeria("mapa_de_calor.png")).pack(side=tk.LEFT, padx=4)
        ttk.Button(f_botoes_g, text="⚡ Gráfico de Velocidade", command=lambda: self.carregar_grafico_galeria("grafico_velocidade.png")).pack(side=tk.LEFT, padx=4)
        ttk.Button(f_botoes_g, text="📏 Distância Acumulada", command=lambda: self.carregar_grafico_galeria("grafico_distancia.png")).pack(side=tk.LEFT, padx=4)

        self.canvas_grafico = tk.Canvas(painel_graficos, bg="#1E293B")
        self.canvas_grafico.pack(fill=tk.BOTH, expand=True, pady=6)

    def atualizar_label_escala(self):
        cal = self.banco.obter_ultima_calibracao()
        if cal:
            px_mm = cal["pixels_por_mm"]
            self.lbl_status_escala.config(text=f"Escala Ativa: {px_mm:.2f} px/mm ({(1.0/px_mm)*1000:.1f} µm/px)", foreground="#16A34A")
        else:
            self.lbl_status_escala.config(text="Escala: Não Calibrada (Medições em pixels)", foreground="#DC2626")

    def iniciar_calibracao_janela(self):
        try:
            dist_mm = float(self.entry_dist_mm.get().replace(",", "."))
        except Exception:
            messagebox.showerror("Erro", "Informe um valor numérico válido para a distância em mm.")
            return

        messagebox.showinfo("Calibração", "A janela de calibração será aberta.\n\n1. Clique em dois pontos com distância conhecida.\n2. Pressione ENTER para confirmar.\n3. Pressione ESC para cancelar.")
        calibrador = CalibradorEscala()
        resultado = calibrador.calibrar(distancia_conhecida_mm=dist_mm)
        if resultado:
            self.atualizar_label_escala()
            messagebox.showinfo("Sucesso", f"Calibração salva com sucesso!\nFator: {resultado:.3f} px/mm")

    def carregar_grafico_galeria(self, nome_arquivo: str):
        pasta = getattr(self, 'pasta_ultimos_resultados', None)
        caminho = None
        if pasta and (Path(pasta) / nome_arquivo).exists():
            caminho = Path(pasta) / nome_arquivo
        else:
            # Buscar o mais recente na pasta resultados
            arquivos = list((BASE_DIR / "resultados").glob(f"**/{nome_arquivo}"))
            if arquivos:
                caminho = arquivos[-1]

        if caminho and caminho.exists():
            img = Image.open(str(caminho))
            c_w = self.canvas_grafico.winfo_width()
            c_h = self.canvas_grafico.winfo_height()
            if c_w > 50 and c_h > 50:
                img.thumbnail((c_w - 20, c_h - 20), Image.Resampling.LANCZOS)
                self.grafico_img_tk = ImageTk.PhotoImage(img)
                self.canvas_grafico.delete("all")
                self.canvas_grafico.create_image(c_w // 2, c_h // 2, anchor=tk.CENTER, image=self.grafico_img_tk)
                self.status_var.set(f"Exibindo: {nome_arquivo}")
        else:
            messagebox.showinfo("Informação", f"Nenhum gráfico '{nome_arquivo}' encontrado.\nExecute um rastreamento primeiro.")

    def ao_fechar(self):
        self.parar_preview()
        self.destroy()


if __name__ == "__main__":
    app = EntomoTrackApp()
    app.mainloop()
