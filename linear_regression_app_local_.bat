@echo off
REM ----------------------------------------------------
REM Streamlitアプリを起動するバッチファイル

REM (1) Minicondaのパス
SET CONDA_ROOT=C:\Users\denyu\miniconda3

REM (2) condaを初期化して、基本環境をアクティベート
CALL "%CONDA_ROOT%\Scripts\activate.bat"

REM (3) 目的の環境「pygraph-app」をアクティベート
CALL conda activate pygraph-app

REM (4) プロジェクトフォルダに移動
REM ここはアプリのapp.pyがあるフォルダの正しいパスに修正してください
cd C:\Users\denyu\Desktop\pygraph-app\linear_regression_app

REM (5) Streamlitアプリを起動
streamlit run app.py
REM ----------------------------------------------------