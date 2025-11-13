@echo off
chcp 65001 >nul
title ICT Swing Trading Bot
echo.

echo ========================================
echo    ICT SWING TRADING BOT LAUNCHER
echo ========================================
echo.

:: Set paths
set VENV_PATH=C:\Users\Bamidele\Documents\trading_ai\ict_swing_ai\ict_swing_trading_bot\ict_scripts
set PROJECT_PATH=C:\Users\Bamidele\Documents\trading_ai\ict_swing_ai\ict_jpy_swing_trader
set BOT_SCRIPT=live\live_demo_bot.py

echo [INFO] Virtual Environment: %VENV_PATH%
echo [INFO] Project Path: %PROJECT_PATH%
echo [INFO] Bot Script: %BOT_SCRIPT%
echo.

:: Check if virtual environment exists
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at:
    echo         %VENV_PATH%
    echo.
    echo Please check your virtual environment path.
    pause
    exit /b 1
)

:: Check if project directory exists
if not exist "%PROJECT_PATH%" (
    echo [ERROR] Project directory not found at:
    echo         %PROJECT_PATH%
    echo.
    echo Please check your project path.
    pause
    exit /b 1
)

:: Check if bot script exists
if not exist "%PROJECT_PATH%\%BOT_SCRIPT%" (
    echo [ERROR] Bot script not found at:
    echo         %PROJECT_PATH%\%BOT_SCRIPT%
    echo.
    echo Please check your bot script path.
    pause
    exit /b 1
)

echo [SUCCESS] All paths verified!
echo.

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

:: Change to project directory
echo [INFO] Changing to project directory...
cd /d "%PROJECT_PATH%"

if errorlevel 1 (
    echo [ERROR] Failed to change to project directory
    pause
    exit /b 1
)

echo [SUCCESS] Environment setup complete!
echo.
echo ========================================
echo        STARTING TRADING BOT
echo ========================================
echo.
echo Bot: %BOT_SCRIPT%
echo Date: %date% %time%
echo.
echo Press Ctrl+C to stop the bot
echo ========================================
echo.

:: Run the bot
python "%BOT_SCRIPT%"

:: Check if bot exited with error
if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] Bot exited with error code: %errorlevel%
    echo ========================================
) else (
    echo.
    echo ========================================
    echo [INFO] Bot exited normally
    echo ========================================
)

:: Deactivate virtual environment
echo.
echo [INFO] Deactivating virtual environment...
call "%VENV_PATH%\Scripts\deactivate.bat"

echo.
echo ========================================
echo          BOT SESSION ENDED
echo ========================================
echo.
pause