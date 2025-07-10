@echo off
echo 🚀 Stock Price Analyzer - Clean Launcher
echo ========================================
echo.
echo 📱 Starting application...
echo ⏹️  Press Ctrl+C to stop
echo.

REM Set environment variables to suppress errors
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
set STREAMLIT_LOGGER_LEVEL=error
set PYTHONWARNINGS=ignore

REM Run the application
python launch_app.py

pause 