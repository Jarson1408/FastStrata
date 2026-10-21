@echo off
rem Uninstall FastStrata: what setup put outside this folder goes to the Recycle Bin (the FastStrata Gateway's settings
rem in %APPDATA%\faststrata-gateway, the desktop shortcut FastStrata), and OpenCode's config loses the provider
rem faststrata (a .bak of it is kept first). It lists everything and asks before it changes anything. This folder and
rem the model files are yours to delete afterwards: it says where they are and how big. Options are passed on to
rem setup.py (--yes: no question).
setlocal
title FastStrata - uninstall
cd /d "%~dp0"
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe" & goto run
rem no environment of setup's here: any Python 3 runs this part of setup (it needs nothing installed)
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3" & goto run
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python" & goto run
echo  No Python here: setup never ran in this folder, so it put nothing outside it. Delete the folder yourself.
pause
exit /b 1

:run
%PY% setup.py --uninstall %*
pause
exit /b
