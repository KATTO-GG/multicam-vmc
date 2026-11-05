@echo off
REM Builds multicam_vmc.exe (+ portable zip, + installer if Inno Setup found).
REM Needs: Python 3.10-3.12 on PATH. No admin needed.
setlocal
cd /d "%~dp0"

python -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install -U pip || goto :err
pip install -r requirements.txt pyinstaller || goto :err

if not exist face_landmarker.task (
  echo downloading face model ...
  powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile face_landmarker.task" || goto :err
)

pyinstaller --noconfirm --clean --windowed --name multicam_vmc ^
  --collect-all mediapipe --add-data "face_landmarker.task;." multicam_vmc.py || goto :err

echo making portable zip ...
powershell -NoProfile -Command "Compress-Archive -Path dist\multicam_vmc -DestinationPath multicam_vmc_portable.zip -Force"

set ISCC=
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if defined ISCC (
  "%ISCC%" installer.iss || goto :err
  echo.
  echo DONE: dist_installer\multicam_vmc_setup.exe   (installer, no admin)
) else (
  echo.
  echo Inno Setup not found - skipped installer. Get it from https://jrsoftware.org/isdl.php
  echo ^(choose "Install for me only" = no admin^), then run build.bat again.
)
echo DONE: dist\multicam_vmc\multicam_vmc.exe   and   multicam_vmc_portable.zip
pause
exit /b 0

:err
echo.
echo BUILD FAILED. See messages above.
pause
exit /b 1
