; Values are supplied by scripts/build-installers.py. No elevated install needed.
[Setup]
AppId={{F82B9577-D75C-4EBB-97A6-CB51CF2E498C}
AppName=PlayChop
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\PlayChop
DefaultGroupName=PlayChop
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename={#OutputName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\PlayChop.exe
LicenseFile={#RepoRoot}\LICENSE
[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\PlayChop"; Filename: "{app}\PlayChop.exe"
Name: "{userdesktop}\PlayChop"; Filename: "{app}\PlayChop.exe"; Tasks: desktopicon
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked
[Run]
Filename: "{app}\PlayChop.exe"; Description: "Launch PlayChop"; Flags: nowait postinstall skipifsilent
