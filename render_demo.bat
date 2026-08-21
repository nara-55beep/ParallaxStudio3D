@echo off
cd /d "%~dp0"
python -m parallax3d render scenes\child_mother.json --output output\demo
if errorlevel 1 pause
