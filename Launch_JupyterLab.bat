@echo off
REM ──────────────────────────────────────────────────────────
REM (1) Minicondaのパス（ユーザー名に合わせて設定）
SET CONDA_ROOT=C:\Users\denyu\miniconda3

REM (2) conda の初期化
CALL "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"

REM (3) 目的の環境「pygraph-app」をアクティベート
CALL conda activate pygraph-app

REM (4) プロジェクトのフォルダに移動
REM ここはプロジェクトフォルダの正しいパスに修正してください
cd C:\Users\denyu\Desktop\pygraph-app

REM (5) JupyterLabを起動
jupyter lab
REM ──────────────────────────────────────────────────────────