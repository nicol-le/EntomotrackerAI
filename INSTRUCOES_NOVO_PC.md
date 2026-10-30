# GUIA DE INSTALAÇÃO E EXECUÇÃO DO ENTOMOTRACK AI EM UM PC NOVO

Este documento contém todas as instruções necessárias para configurar, instalar e executar o **EntomoTrack AI** em qualquer computador com **Windows 10 ou Windows 11**, equipado com **8 GB de RAM** e **processamento exclusivamente por CPU** (sem necessidade de placa de vídeo dedicada ou GPU NVIDIA).

---

## 1. REQUISITOS DO SISTEMA

- **Sistema Operacional:** Windows 10 ou Windows 11 (64 bits).
- **Memória RAM:** 8 GB (otimizado com controle de buffer e coleta periódica de memória).
- **Processador:** CPU convencional (Intel Core i3/i5/i7, AMD Ryzen 3/5 ou superior).
- **Placa de Vídeo:** Não é necessária GPU dedicada.
- **Python:** Versão 3.11, 3.12 ou 3.13 (64 bits).
- **Dispositivo de Captura:** Câmera digital HAYEAR USB conectada ao microscópio estereoscópico BiFocus.
- **Espaço em Disco:** Mínimo de 3 GB livres para ambiente virtual, dependências e gravações.

---

## 2. PREPARAÇÃO DO NOVO COMPUTADOR (PASSO ÚNICO INICIAL)

### 2.1. Instalar o Python (se ainda não estiver instalado)
1. Acesse o site oficial do Python: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2. Baixe o instalador do **Python 3.11**, **3.12** ou **3.13** (versão *Windows installer 64-bit*).
3. **MUITO IMPORTANTE:** Na primeira tela de instalação, marque a caixa:
   - ✅ **"Add Python to PATH"** (ou "Add python.exe to PATH").
4. Clique em **Install Now**.
5. Ao concluir, se aparecer a opção **"Disable path length limit"**, clique nela para evitar restrições de nomes de caminhos longos no Windows.

---

## 3. TRANSFERÊNCIA DO PROJETO PARA O NOVO PC

1. Copie a pasta inteira do projeto (por pendrive, rede local ou compactada em `.zip`) para o novo computador.
2. Sugestão de pastas:
   - `C:\EntomoTrackAI\` ou
   - Na sua Área de Trabalho / Documentos.
3. Certifique-se de que a estrutura básica está presente:
   ```text
   EntomoTrackAI/
   ├── configuracoes/
   │   └── config.yaml
   ├── scripts/
   │   ├── banco.py
   │   ├── teste_camera.py
   │   ├── gravar_video.py
   │   ├── calibrar_escala.py
   │   ├── detectar.py
   │   ├── rastrear.py
   │   ├── analisar_movimento.py
   │   ├── treinar_modelo.py
   │   └── revisao_manual.py
   ├── modelos/
   │   └── yolov8n.pt
   ├── interface.py
   ├── requirements.txt
   ├── iniciar_entomotrack.bat
   └── testes/
       └── test_suite.py
   ```

> [!NOTE]
> Não é necessário copiar a pasta `.venv` do PC antigo. Cada computador deve criar seu próprio ambiente virtual para evitar incompatibilidades de caminhos absolutos do Windows.

---

## 4. INICIALIZAÇÃO AUTOMÁTICA (MÉTODO RECOMENDADO - 1 CLIQUE)

O sistema conta com um inicializador inteligente para Windows:

1. Conecte a câmera digital HAYEAR na porta USB do computador.
2. Dê um **duplo clique** no arquivo:
   ```text
   iniciar_entomotrack.bat
   ```
3. O script irá automaticamente:
   - Verificar se as pastas necessárias existem e criá-las se faltarem.
   - Criar o ambiente virtual isolado `.venv`.
   - Baixar e instalar todas as bibliotecas necessárias para CPU (`torch+cpu`, `ultralytics`, `opencv`, `pandas`, `matplotlib`, `tkinter`).
   - Abrir a interface gráfica do **EntomoTrack AI**.

Nas próximas vezes que quiser usar o programa, basta dar um duplo clique novamente em `iniciar_entomotrack.bat` e ele abrirá a interface imediatamente.

---

## 5. INSTALAÇÃO E EXECUÇÃO MANUAL (VIA TERMINAL POWERSHELL / CMD)

Se preferir realizar a instalação manualmente ou acompanhar todos os logs detalhados:

### Passo 1: Abrir o terminal na pasta do projeto
Abra o **PowerShell** ou **Prompt de Comando (CMD)** dentro da pasta do projeto:
```powershell
cd "C:\EntomoTrackAI"
```

### Passo 2: Criar o ambiente virtual Python
```powershell
python -m venv .venv
```

### Passo 3: Ativar o ambiente virtual
- **No PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Se o PowerShell bloquear a execução de scripts, execute antes: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **No Prompt de Comando (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Passo 4: Instalar as dependências do projeto
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> O arquivo `requirements.txt` já inclui a diretiva `--extra-index-url https://download.pytorch.org/whl/cpu`, garantindo que o PyTorch instalado seja a versão leve para CPU, sem os 4 GB de pacotes CUDA desnecessários para quem não possui placa NVIDIA.

### Passo 5: Baixar os pesos base do modelo YOLOv8n (se ainda não estiver na pasta `modelos/`)
```powershell
python -c "from ultralytics import YOLO; import shutil; m = YOLO('yolov8n.pt'); shutil.copy2('yolov8n.pt', 'modelos/yolov8n.pt'); print('Pesos base prontos!')"
```

### Passo 6: Iniciar o EntomoTrack AI
```powershell
python interface.py
```

---

## 6. VALIDAÇÃO DOS TESTES NO NOVO PC

Para certificar-se de que todos os módulos (câmera, banco de dados, detector YOLO, ByteTrack, cálculos comportamentais e exportação de resultados) estão operando 100% no novo computador, execute a suíte de testes:

```powershell
.\.venv\Scripts\python.exe testes\test_suite.py
```

### Resultado esperado:
```text
test_01_importacoes: OK
test_02_enumeracao_cameras: OK
test_03_geracao_video_teste: OK
test_04_banco_dados_e_auditoria: OK
test_05_carregamento_yolo_cpu: OK
test_06_deteccao_situacoes_a_b_c: OK
test_07_bytetrack_multiobjeto: OK
test_08_calibracao_escala: OK
test_09_metricas_comportamentais: OK
test_10_exportacao_graficos_e_relatorio: OK
test_11_human_in_the_loop_e_dataset: OK
test_12_fluxo_completo_integrado: OK

----------------------------------------------------------------------
Ran 12 tests in 18s -> OK
```

---

## 7. GUIA DE OPERAÇÃO DA INTERFACE

A interface é dividida em 4 abas funcionais:

```text
┌────────────────────────────────────────────────────────────────────────┐
│  🔬 EntomoTrack AI — Monitoramento e Rastreamento de Insetos          │
├───────────────────┬───────────────────┬───────────────────┬────────────┤
│ 1. Câmera HAYEAR  │ 2. Rastreamento   │ 3. Revisão Manual │ 4. Gráficos│
└───────────────────┴───────────────────┴───────────────────┴────────────┘
```

### 7.1. Aba 1: Câmera HAYEAR & Gravação
1. Clique em **🔍 Detectar Câmeras USB** para enumerar os dispositivos conectados.
2. Selecione a resolução desejada (padrão recomendado: `1280x720`).
3. Clique em **▶ Iniciar Pré-Visualização** para conferir o foco do microscópio BiFocus em tempo real.
4. Utilize **📸 Capturar Foto** para registrar imagens isoladas (salvas em `imagens/`).
5. Utilize **⏺ Iniciar Gravação** para gravar os ensaios experimentais (salvos automaticamente em `videos/`).

> [!WARNING]
> **Atenção:** Feche o aplicativo oficial da HAYEAR antes de abrir a câmera no EntomoTrack AI. O Windows impede que dois programas acessem a mesma câmera USB simultaneamente via DirectShow.

### 7.2. Aba 2: Rastreamento & Detecção
1. Clique em **Buscar Vídeo...** e selecione o vídeo gravado.
2. Selecione os pesos do modelo (padrão: `modelos/yolov8n.pt` ou o modelo específico de insetos treinado).
3. Ajuste a **Confiança Mínima** (padrão inicial: `0.35`).
4. Defina o **Salto de Quadros Alternados** (padrão: `2` para acelerar em CPUs convencionais).
5. Clique em **🚀 Iniciar Rastreamento ByteTrack**.
6. Acompanhe a barra de progresso. Ao término, os resultados serão processados automaticamente.
7. Clique em **📂 Abrir Pasta de Resultados** para inspecionar os vídeos e planilhas geradas.

### 7.3. Aba 3: Revisão & Treinamento (Human-in-the-Loop)
1. Insetos com confiança incerta ou não reconhecidos aparecem listados na tabela.
2. Clique em qualquer linha para inspecionar o recorte ampliado do inseto.
3. Se for uma espécie conhecida, selecione-a e clique em **✅ Confirmar Espécie**.
4. Se for uma espécie nova, clique em **➕ Cadastrar Nova Espécie** e preencha os dados taxonômicos.
5. Se for sujeira, poeira ou reflexo, clique em **❌ Falso Positivo**.
6. Se quiser remover uma detecção errada sem apagar o vídeo original, clique em **🗑 Excluir Anotação**.
7. Para gerar um novo treinamento, clique em **📦 Exportar Dataset YOLO** e depois em **🧠 Treinar Modelo em CPU**. O modelo será treinado respeitando os limites da CPU e salvo na pasta `modelos/`.

### 7.4. Aba 4: Calibração & Resultados
1. Posicione uma lâmina micrométrica ou régua milimetrada sob o microscópio.
2. Digite a distância real em milímetros no campo (ex.: `1.0`).
3. Clique em **📐 Calibrar Interativamente**:
   - Clique no ponto inicial da escala.
   - Clique no ponto final da escala.
   - Pressione **ENTER** para salvar o fator de calibração em $px/mm$.
4. Use os botões da galeria para alternar a visualização de:
   - **Trajetórias 2D** (`trajetorias_2d.png`).
   - **Mapa de Calor Espacial** (`mapa_de_calor.png`).
   - **Velocidade ao Longo do Tempo** (`grafico_velocidade.png`).
   - **Distância Acumulada** (`grafico_distancia.png`).

---

## 8. RESOLUÇÃO DE PROBLEMAS COMUNS (TROUBLESHOOTING)

| Problema | Causa Provável | Solução |
| :--- | :--- | :--- |
| **"Não foi possível abrir a câmera"** | A câmera está em uso por outro software ou cabo USB desconectado. | Feche o software oficial da HAYEAR. Reconecte o cabo USB e clique em "Detectar Câmeras USB". |
| **"python não é reconhecido como comando"** | O Python não foi adicionado ao PATH do Windows durante a instalação. | Reinstale o Python marcando a caixa **"Add Python to PATH"**. |
| **"Script Execution Policy Error" no PowerShell** | Política de segurança padrão do Windows bloqueando scripts `.ps1`. | Abra o PowerShell como Administrador e digite: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` e confirme com `S`. |
| **"Processamento do vídeo muito lento"** | Resolução do vídeo muito alta para a CPU. | Na Aba 2, altere o campo **"Quadros Alternados (Pulo)"** para `3` ou `4`, reduzindo o esforço da CPU pela metade. |
| **"Certificado SSL inválido ao baixar pesos"** | Data e hora do relógio do Windows desajustadas. | Ajuste a data e hora do Windows para o horário automático nas configurações do sistema. |
| **"Resultados aparecem em pixels e não em mm"** | A escala espacial ainda não foi calibrada. | Acesse a Aba 4 e execute a **Calibração Interativa** com uma referência de distância conhecida. |

---

## 9. RESULTADOS GERADOS EM CADA EXPERIMENTO

Ao finalizar qualquer rastreamento, uma nova pasta com data e hora é gerada dentro de `resultados/` contendo:

1. `video_rastreado.mp4` — Vídeo com caixas delimitadoras e trajetórias coloridas por ID.
2. `coordenadas_por_quadro.csv` — Tabela completa com posição $(X, Y)$, quadro a quadro, tempo em segundos, confiança e ID de cada indivíduo.
3. `metricas_individuais.csv` — Resumo estatístico por inseto: distância total percorrida ($mm$ ou $px$), velocidade média ($mm/s$ ou $px/s$), velocidade máxima, tempo em atividade ($s$), tempo imóvel ($s$) e mudanças de direção.
4. `trajetorias_2d.png` — Gráfico vetorial bidimensional de todas as rotas percorridas.
5. `mapa_de_calor.png` — Mapa de densidade de ocupação espacial por cores térmicas.
6. `grafico_velocidade.png` — Perfil cinemático de velocidade no tempo.
7. `grafico_distancia.png` — Curva de deslocamento acumulado.
8. `relatorio_resumo.txt` — Relatório científico consolidado com notas metodológicas e advertências taxonômicas.
