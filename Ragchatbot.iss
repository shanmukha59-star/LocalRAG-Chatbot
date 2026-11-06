
[Setup]
AppId={{8F3B2C1A-5D4E-4F6A-9B7C-2E1D0A3F4B5C}
AppName=RAG Chatbot
AppVersion=1.0
AppPublisher=Ashan
DefaultDirName={localappdata}\RAG Chatbot
DefaultGroupName=RAG Chatbot
DisableProgramGroupPage=yes
PrivilegesRequired=lowest

OutputDir=C:\Users\Ashan\OneDrive\Desktop
OutputBaseFilename=RAGChatbot-Setup

Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "C:\Users\Ashan\OneDrive\Desktop\RAG - Pipeline\dist\RAGChatbot\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\RAG Chatbot"; Filename: "{app}\RAGChatbot.exe"
Name: "{autodesktop}\RAG Chatbot"; Filename: "{app}\RAGChatbot.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\RAGChatbot.exe"; Description: "Launch RAG Chatbot"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\db"
