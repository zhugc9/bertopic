@echo off
cd /d "%~dp0"
call conda activate bertopic_env
streamlit run web_ui.py --server.port 8502
pause