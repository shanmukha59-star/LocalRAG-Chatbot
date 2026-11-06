$ErrorActionPreference = 'Stop'

$projectDir = $PSScriptRoot
$python = Join-Path $projectDir '.venv\Scripts\python.exe'
$spec = Join-Path $projectDir 'RAGChatbot.spec'

if (-not (Test-Path $python)) {
    throw "Project Python environment not found: $python"
}

& $python -c 'import PyInstaller' *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller is missing from .venv. Install it with: .venv\Scripts\python.exe -m pip install pyinstaller'
}

Push-Location $projectDir
try {
    & $python -m PyInstaller --clean --noconfirm $spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$appDirectory = Join-Path $projectDir 'dist\RAGChatbot'
$exePath = Join-Path $appDirectory 'RAGChatbot.exe'
$databasePath = Join-Path $appDirectory '_internal\db\chroma_db\chroma.sqlite3'
if (-not (Test-Path $exePath)) {
    throw "Build completed but executable was not found: $exePath"
}
if (-not (Test-Path $databasePath)) {
    throw "Build completed but the Chroma database was not bundled: $databasePath"
}

$desktopPath = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktopPath 'RAGChatbot.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $exePath
$shortcut.WorkingDirectory = $appDirectory
$shortcut.Description = 'RAG Chatbot desktop app'
$shortcut.IconLocation = "$exePath,0"
$shortcut.Save()

Write-Host "Desktop app ready: $exePath"
Write-Host "Desktop shortcut created: $shortcutPath"