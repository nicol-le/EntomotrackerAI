@echo off
chcp 65001 >nul
title EntomoTrack AI - Inicializador Automatico

echo ==============================================================================
echo                      ENTOMOTRACK AI — INICIALIZADOR
echo           Rastreamento Automatico de Insetos em Microscopia
echo               Otimizado para CPU Convencional e 8 GB RAM
echo ==============================================================================
echo.

cd /d "%~dp0"

:: 1. Criar pastas necessárias se não existirem
echo [*] Verificando estrutura de pastas do sistema...
if not exist "modelos" mkdir modelos
if not exist "videos" mkdir videos
if not exist "imagens\recortes" mkdir imagens\recortes
if not exist "imagens\pendentes" mkdir imagens\pendentes
if not exist "imagens\frames_originais" mkdir imagens\frames_originais
if not exist "resultados" mkdir resultados
if not exist "datasets" mkdir datasets
if not exist "banco_dados" mkdir banco_dados
if not exist "configuracoes" mkdir configuracoes
echo [OK] Pastas verificadas com sucesso.
echo.

:: 2. Verificar Ambiente Virtual .venv
if not exist ".venv\Scripts\python.exe" (
    echo [*] Ambiente virtual .venv nao encontrado.
    echo [*] Criando ambiente virtual...
    python -m venv .venv 2>nul
    if not exist ".venv\Scripts\python.exe" (
        py -3.11 -m venv .venv 2>nul
    )
    if not exist ".venv\Scripts\python.exe" (
        py -3 -m venv .venv 2>nul
    )
    if not exist ".venv\Scripts\python.exe" (
        echo [ERRO] Nao foi possivel criar o ambiente virtual automaticamente.
        echo Certifique-se de que o Python 3.11+ esta instalado no sistema.
        pause
        exit /b 1
    )
    echo [OK] Ambiente virtual criado com sucesso.
    echo [*] Instalando dependencias necessarias (aguarde)...
    ".venv\Scripts\pip.exe" install -r requirements.txt
)

:: 3. Teste rápido de integridade
echo [*] Verificando integridade das bibliotecas...
".venv\Scripts\python.exe" -c "import cv2, torch, ultralytics, pandas, matplotlib, tkinter; print('[OK] Todas as dependencias essenciais estao operacionais.')" 2>nul
if %errorlevel% neq 0 (
    echo [AVISO] Algumas dependencias precisam ser atualizadas ou concluidas.
    echo Tentando instalar requirements.txt...
    ".venv\Scripts\pip.exe" install -r requirements.txt
)

:: 4. Iniciar Interface Gráfica
echo.
echo [*] Iniciando EntomoTrack AI...
start "" ".venv\Scripts\pythonw.exe" interface.py 2>nul
if %errorlevel% neq 0 (
    ".venv\Scripts\python.exe" interface.py
)

exit /b 0
