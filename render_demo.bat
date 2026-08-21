@echo off
cd /d "%~dp0"
python -m parallax3d image assets\child_mother_source.jpeg --output output\demo
if errorlevel 1 pause
