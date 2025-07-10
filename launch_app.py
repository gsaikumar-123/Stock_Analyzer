#!/usr/bin/env python3
"""
Clean launcher for Stock Price Analyzer
Runs Streamlit without PyTorch file watcher errors
"""

import subprocess
import sys
import os

def main():
    print("🚀 Launching Stock Price Analyzer...")
    print("📱 This will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    # Set environment variables to suppress warnings
    env = os.environ.copy()
    env['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
    env['STREAMLIT_LOGGER_LEVEL'] = 'error'
    env['PYTHONWARNINGS'] = 'ignore'
    
    # Run Streamlit with clean configuration
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.fileWatcherType", "none",
            "--logger.level", "error",
            "--server.headless", "false"
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 