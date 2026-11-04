; Values are supplied by scripts/build-installers.py. No elevated install needed.
[Setup]
AppId={{F82B9577-D75C-4EBB-97A6-CB51CF2E498C}
AppName=Scout Studio
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\Scout-Studio
DefaultGroupName=Scout Studio
UsePreviousGroup=no
UsePreviousAppDir=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename={#OutputName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=Scout Studio
UninstallDisplayIcon={app}\Scout-Studio.exe
LicenseFile={#RepoRoot}\LICENSE
[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\Scout Studio"; Filename: "{app}\Scout-Studio.exe"
Name: "{userdesktop}\Scout Studio"; Filename: "{app}\Scout-Studio.exe"; Tasks: desktopicon
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked
[Run]
Filename: "{app}\Scout-Studio.exe"; Description: "Launch Scout Studio"; Flags: nowait postinstall skipifsilent

[Code]
var
  PreviousName, PreviousExecutable, PreviousDirectory, PreviousGroup: String;

function InitializeSetup(): Boolean;
var
  Key: String;
begin
  { Keep the existing AppId and install location so upgrades share one uninstaller. }
  Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\' +
    '{F82B9577-D75C-4EBB-97A6-CB51CF2E498C}_is1';
  RegQueryStringValue(HKCU, Key, 'DisplayIcon', PreviousExecutable);
  PreviousExecutable := RemoveQuotes(PreviousExecutable);
  if CompareText(ExtractFileExt(PreviousExecutable), '.exe') = 0 then
    PreviousName := ChangeFileExt(ExtractFileName(PreviousExecutable), '');
  RegQueryStringValue(HKCU, Key, 'InstallLocation', PreviousDirectory);
  RegQueryStringValue(HKCU, Key, 'Inno Setup: Icon Group', PreviousGroup);
  Result := True;
end;

procedure RemovePreviousShortcut(const Folder, Name: String);
var
  Shell, Directory, Item: Variant;
  Shortcut, Target: String;
begin
  { Remove only a shortcut that still targets this prior installation. }
  if (Name = '') or (ExtractFileName(Name) <> Name) then
    Exit;
  Shortcut := AddBackslash(Folder) + Name + '.lnk';
  if not FileExists(Shortcut) then
    Exit;
  try
    Shell := CreateOleObject('Shell.Application');
    Directory := Shell.NameSpace(Folder);
    Item := Directory.ParseName(Name + '.lnk');
    Target := Item.GetLink.Path;
    if (CompareText(RemoveBackslashUnlessRoot(ExtractFileDir(Target)),
                    RemoveBackslashUnlessRoot(PreviousDirectory)) = 0) and
       (CompareText(Target, PreviousExecutable) = 0) then
      DeleteFile(Shortcut);
  except
    Log('Could not inspect previous shortcut: ' + Shortcut);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and (PreviousDirectory <> '') and
     (PreviousName <> '') and (PreviousName <> 'Scout-Studio') then
  begin
    RemovePreviousShortcut(ExpandConstant('{userdesktop}'), PreviousName);
    if (PreviousGroup <> '') and (ExtractFileName(PreviousGroup) = PreviousGroup) then
      RemovePreviousShortcut(AddBackslash(ExpandConstant('{userprograms}')) +
        PreviousGroup, PreviousName);
  end;
end;
