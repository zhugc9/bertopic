@echo off
chcp 65001 > nul

:: =================================================================
::
::      BERTopic 项目智能启动器 (v3.0)
::      - 首次运行自动安装环境，之后自动启动分析 -
::
:: =================================================================

:: 定义环境名称
set ENV_NAME=bertopic_lab
set PYTHON_VERSION=3.9
set REQUIREMENTS_FILE=requirements.txt

:: 检查Anaconda是否在PATH中
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [!!!] 致命错误: 找不到 "conda" 命令。
    echo.
    echo      请确保您已经正确安装了Anaconda，并且在安装时勾选了
    echo      "Add Anaconda to my PATH environment variable"。
    echo.
    echo      如果未勾选，请手动打开 "Anaconda Prompt" 并运行此脚本。
    echo.
    pause
    exit /b 1
)

:: 检查环境是否存在
conda env list | findstr /B "%ENV_NAME% " >nul
if %errorlevel% neq 0 (
    :: --- 环境不存在，提示运行配置脚本 ---
    echo.
    echo [提示] 检测到您是首次运行，需要先配置环境。
    echo =================================================================
    echo.
    echo      请先运行环境配置程序
    echo.
    echo =================================================================
    echo.
    echo 请按以下步骤操作:
    echo   1. 关闭此窗口
    echo   2. 双击运行 "一键智能配置.bat"
    echo   3. 等待配置完成后，再运行此脚本
    echo.
    echo [提示] 配置过程大约需要10-20分钟，只需要运行一次。
    echo.
    pause
    exit /b 1
)

:: --- 环境已存在或安装完成，执行启动流程 ---
echo.
echo =================================================================
echo.
echo      BERTopic 项目一键启动器
echo.
echo =================================================================
echo.
echo [i] 正在激活专属环境 "%ENV_NAME%"...

call conda.bat activate %ENV_NAME% && (
    echo [OK] 环境激活成功！
    echo.
    echo [i] 正在启动 "quick_start.py" 智能安全助手...
    echo.
    python quick_start.py
)

if %errorlevel% neq 0 (
    echo.
    echo [!!!] 错误: 程序运行失败。
    echo.
    echo      可能的原因:
    echo      1. 环境 "%ENV_NAME%" 未正确安装。
    echo      2. Python脚本本身存在错误。
    echo.
)

echo.
echo =================================================================
echo.
echo      分析已结束。您可以关闭此窗口。
echo.
echo =================================================================
pause
