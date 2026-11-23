@echo off
setlocal
cd /d "%~dp0.."
set "PROJECT=%CD%"
set "APP=%PROJECT%\leather-ui"
set "EXPERIMENT=%PROJECT%\lstm-day1"
set "RUNTIME=%PROJECT%\work\lstm-day1"
for %%I in ("%PROJECT%") do if /i "%%~nxI"=="outputs" set "RUNTIME=%PROJECT%\..\work\lstm-day1"
for %%I in ("%RUNTIME%") do set "RUNTIME=%%~fI"
set "UV_CACHE_DIR=%RUNTIME%\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%RUNTIME%\python"
set "UV_PROJECT_ENVIRONMENT=%RUNTIME%\venv"
set "TORCH_HOME=%RUNTIME%\torch-cache"
set "MPLCONFIGDIR=%RUNTIME%\matplotlib"
set "PYTHONDONTWRITEBYTECODE=1"
set "PORT=%~1"
if not defined PORT set "PORT=8765"
if /i "%~2"=="rebuild" goto build
if exist "%APP%\frontend\dist\index.html" goto serve

:build
pushd "%APP%\frontend"
call npm.cmd ci --no-audit --no-fund
if errorlevel 1 exit /b 1
call npm.cmd run build
if errorlevel 1 exit /b 1
popd

:serve
uv sync --locked --project "%EXPERIMENT%" --python 3.12 --group demo
if errorlevel 1 exit /b 1
"%UV_PROJECT_ENVIRONMENT%\Scripts\python.exe" -m uvicorn api:app --app-dir "%APP%" --host 127.0.0.1 --port "%PORT%"
exit /b %errorlevel%
