# multicam_vmc

Multi-webcam face tracking fused into one VMC stream. Better tracking at
side angles, fewer dropouts. Works with any app that has a VMC receiver
(VNyan, Warudo, VSeeFace, ...). No admin needed anywhere.

## Get the exe / installer
Windows only. Pick one:

**A. GitHub builds it for you (no setup)**
1. Push this folder to a GitHub repo (keep `.github/workflows/build.yml`).
2. Repo > Actions > latest run > Artifacts:
   `multicam_vmc_setup` (installer) or `multicam_vmc_portable` (zip, unzip + run).

**B. Build locally**
1. Install Python 3.10-3.12 (tick "Add to PATH").
2. Optional: Inno Setup (https://jrsoftware.org/isdl.php, choose "Install for me only").
3. Double-click `build.bat`.
4. Outputs: `dist_installer\multicam_vmc_setup.exe`, `multicam_vmc_portable.zip`,
   `dist\multicam_vmc\multicam_vmc.exe`.

Installer is per-user (goes in `%LOCALAPPDATA%\Programs`). Uninstall from
Windows Settings > Apps.

## Use
Run `multicam_vmc.exe`. Launcher window:
1. Tick your cameras (auto-detected).
2. Phone cam: start phone app, press **Find on WiFi** (or paste stream URL).
3. **Start**.

In your VTuber app: enable VMC/OSC receiver, same port (default 39539).

First start calibrates cam placement: sit in view of ALL cams, turn head slowly
left/right/up/down (~6 s). Saved. Moved a cam? Tick Recalibrate or press `c`.

Preview keys: `q` quit (or close window) | `c` recalibrate | `y`/`p`/`r` flip head
yaw/pitch/roll | `m` mirror L/R | `f` dense dots | `a d w s` orbit rig view.
If head turns the wrong way in your app, press y/p/r until right.

## Phone as cam
- DroidCam / Iriun / Camo with PC client: phone appears as a normal camera in the list.
- IP stream apps (Android "IP Webcam", DroidCam MJPEG): **Find on WiFi** scans your
  /24 subnet for known stream ports (plain TCP, no admin). Password-protected streams
  are not auto-found; paste `http://user:pass@ip:8080/video` by hand.
- USB or 5 GHz WiFi, 640x480. Lag between cams hurts fusion.

## Files / data
Per-user data in `%APPDATA%\multicam_vmc`: `calibration.json`, `settings.json`,
`log.txt` (check this if the exe crashes), model cache.

## CLI (source or exe via terminal)
    python multicam_vmc.py --list-cams
    python multicam_vmc.py --cams 0 1 --vrm
    python multicam_vmc.py --discover
    python multicam_vmc.py --cams 0 --auto-phone
Flags: `--nogui --smooth --port --host --width --height --fps --calib-secs
--calib-file --recalibrate --mirror --vrm --no-preview`.

## Windows notes
- Camera blocked? Settings > Privacy > Camera > allow desktop apps.
- SmartScreen "unknown publisher" warning on first run: unsigned build. More info > Run anyway.
- Firewall: app only sends UDP to localhost and probes LAN ports outbound. No listening ports.

## Status
Fusion, calibration, rendering, OSC output, discovery probe, launcher window tested
offline. NOT tested: real cams, real phone stream, real VMC receiver, PyInstaller
build on Windows (mediapipe + PyInstaller can need tweaks).
