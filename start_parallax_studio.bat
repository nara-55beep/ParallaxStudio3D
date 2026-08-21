@echo off
cd /d "%~dp0"
python -c "import cv2, numpy, PIL, torch, transformers" >nul 2>&1
if errorlevel 1 (
  echo Installing the free local AI cutout engine. This is needed only once...
  python -m pip install -e ".[ai]"
  if errorlevel 1 (
    echo Installation failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)
python -m parallax3d app
if errorlevel 1 pause
