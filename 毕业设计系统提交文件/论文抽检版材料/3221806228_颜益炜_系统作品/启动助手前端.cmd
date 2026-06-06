@echo off
chcp 65001 >nul
cd /d "%~dp0\LLM助手源码\ui2"
set VITE_API_BASE_URL=http://127.0.0.1:7890
npm install
npm run dev
pause
