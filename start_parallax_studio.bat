@echo off
cd /d "%~dp0"
python -m parallax3d app
if errorlevel 1 pause
