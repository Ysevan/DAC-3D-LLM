@echo off
chcp 65001 >nul
cd /d "%~dp0\LLM助手源码"
python -m pip install -e .
python app.py
pause
