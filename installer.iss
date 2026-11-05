; Inno Setup script. Per-user install: NO admin needed.
[Setup]
AppName=multicam_vmc
AppVersion=1.0
AppPublisher=multicam_vmc
DefaultDirName={autopf}\multicam_vmc
DefaultGroupName=multicam_vmc
PrivilegesRequired=lowest
OutputDir=dist_installer
OutputBaseFilename=multicam_vmc_setup
Compression=lzma2
SolidCompression=yes
DisableProgramGroupPage=yes
UninstallDisplayName=multicam_vmc

[Tasks]
Name: "desktopicon"; Description: "Desktop shortcut"; Flags: unchecked

[Files]
Source: "dist\multicam_vmc\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\multicam_vmc"; Filename: "{app}\multicam_vmc.exe"
Name: "{autodesktop}\multicam_vmc"; Filename: "{app}\multicam_vmc.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\multicam_vmc.exe"; Description: "Launch multicam_vmc"; Flags: postinstall nowait skipifsilent
