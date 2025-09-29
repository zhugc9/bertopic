@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo BerTopic Web Interface Launcher
echo ========================================
echo Activating bertopic_env environment...
call conda activate bertopic_env 2>nul || call "%CONDA_PREFIX%\..\Scripts\activate.bat" bertopic_env
echo Starting Web Interface...
streamlit run tools\web_ui.py --server.port 8502
pause
