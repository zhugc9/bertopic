@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo BerTopic Analysis System Launcher
echo ========================================
echo 💡 提示：修改 config.yaml 设置分析参数
echo ========================================
echo Activating bertopic_env environment...
call conda activate bertopic_env 2>nul || call "%CONDA_PREFIX%\..\Scripts\activate.bat" bertopic_env
echo Starting Analysis System...
python main.py --run
pause
