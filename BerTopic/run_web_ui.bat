@echo off
chcp 65001 > nul

:: =================================================================
::
::      BERTopic Web界面启动器
::      启动Streamlit交互式界面
::
:: =================================================================

echo.
echo =================================================================
echo.
echo      🚀 BERTopic Web界面启动器
echo.
echo =================================================================
echo.

:: 检查conda环境
set ENV_NAME=bertopic_lab

conda env list | findstr /B "%ENV_NAME% " >nul
if %errorlevel% neq 0 (
    echo [!!!] 错误: 找不到conda环境 "%ENV_NAME%"
    echo.
    echo      请先双击运行 "一键智能配置.bat" 配置环境
    echo.
    pause
    exit /b 1
)

:: 激活环境并启动Web界面
echo [i] 正在启动Web界面...
echo [i] 浏览器将自动打开，如果没有请手动访问显示的地址
echo.

call conda.bat activate %ENV_NAME% && (
    echo [OK] 环境激活成功！
    echo.
    echo [i] 启动Streamlit应用...
    echo [i] 按 Ctrl+C 停止服务
    echo.
    streamlit run web_ui.py
)

if %errorlevel% neq 0 (
    echo.
    echo [!!!] 错误: Web界面启动失败
    echo.
    echo      可能的原因:
    echo      1. Streamlit未正确安装
    echo      2. 端口被占用
    echo      3. 其他网络问题
    echo.
    echo      请尝试手动运行: streamlit run web_ui.py
    echo.
)

echo.
echo =================================================================
echo.
echo      Web界面已关闭
echo.
echo =================================================================
pause
