[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [ValidateSet("cu126-py312", "cu130-py313")]
    [string]$PortableBuild = "cu126-py312",
    [switch]$SkipExternalCustomNodes,
    [switch]$SkipFFmpeg,
    [switch]$SkipVCRedist,
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ComfyRepoApi = "https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest"
$VRGDGRepo = "https://github.com/vrgamegirl19/comfyui-vrgamedevgirl"
$ComfyManagerRepo = "https://github.com/ltdrdata/ComfyUI-Manager"

$ExternalCustomNodes = @(
    @{ Name = "ComfyUI-Manager"; Repo = $ComfyManagerRepo; Branch = "main"; RequiredFor = "installing and fixing missing nodes from inside ComfyUI" },
    @{ Name = "ComfyUI-VideoHelperSuite"; Repo = "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"; Branch = "main"; RequiredFor = "VHS video/audio load and combine nodes" },
    @{ Name = "ComfyUI-LTXVideo"; Repo = "https://github.com/Lightricks/ComfyUI-LTXVideo"; Branch = "master"; RequiredFor = "LTXV and Gemma/LTX video workflow nodes" },
    @{ Name = "rgthree-comfy"; Repo = "https://github.com/rgthree/rgthree-comfy"; Branch = "main"; RequiredFor = "workflow utility nodes used by many shared workflows" },
    @{ Name = "ComfyUI-Crystools"; Repo = "https://github.com/crystian/ComfyUI-Crystools"; Branch = "main"; RequiredFor = "VRAM/RAM cleanup and system helper nodes" },
    @{ Name = "ComfyUI-KJNodes"; Repo = "https://github.com/kijai/ComfyUI-KJNodes"; Branch = "main"; RequiredFor = "common video and scheduler helper nodes" }
)

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "WARN: $Message" -ForegroundColor Yellow
}

function Require-Windows {
    if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
        throw "This installer is for Windows portable ComfyUI."
    }
}

function Select-Folder {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = "Choose where to install ComfyUI portable"
    $dialog.ShowNewFolderButton = $true
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        return $dialog.SelectedPath
    }
    throw "No install folder was selected."
}

function Get-InstallRoot {
    if ($InstallRoot -and $InstallRoot.Trim()) {
        return (Resolve-OrCreate-Path $InstallRoot)
    }

    try {
        return (Resolve-OrCreate-Path (Select-Folder))
    }
    catch {
        Write-Warn "Folder picker was not available: $($_.Exception.Message)"
        $typed = Read-Host "Type or paste the folder where ComfyUI should be installed"
        if (-not $typed.Trim()) {
            throw "No install folder was provided."
        }
        return (Resolve-OrCreate-Path $typed)
    }
}

function Resolve-OrCreate-Path {
    param([string]$Path)
    $expanded = [Environment]::ExpandEnvironmentVariables($Path)
    New-Item -ItemType Directory -Force -Path $expanded | Out-Null
    return (Resolve-Path -LiteralPath $expanded).Path
}

function Invoke-FileDownload {
    param(
        [string]$Uri,
        [string]$OutFile
    )
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null
    Write-Host "Downloading $Uri"
    Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
}

function Get-PortableAsset {
    param([string]$Build)
    $release = Invoke-RestMethod -Uri $ComfyRepoApi -Headers @{ "User-Agent" = "VRGDG-ComfyUI-Installer" }
    $assets = @($release.assets)

    $patterns = if ($Build -eq "cu130-py313") {
        @(
            "^ComfyUI_windows_portable_nvidia.*cu130.*\.7z$",
            "^ComfyUI_windows_portable_nvidia\.7z$"
        )
    }
    else {
        @(
            "^ComfyUI_windows_portable_nvidia.*cu126.*\.7z$",
            "^ComfyUI_windows_portable_nvidia_cu126\.7z$"
        )
    }

    foreach ($pattern in $patterns) {
        $match = $assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1
        if ($match) {
            return $match
        }
    }

    $available = ($assets | Where-Object { $_.name -like "*windows*portable*nvidia*.7z" } | Select-Object -ExpandProperty name) -join ", "
    throw "Could not find a matching ComfyUI portable asset for $Build. Available Nvidia portable assets: $available"
}

function Get-7zr {
    param([string]$ToolsDir)
    $sevenZip = Join-Path $ToolsDir "7zr.exe"
    if (-not (Test-Path -LiteralPath $sevenZip)) {
        Invoke-FileDownload -Uri "https://www.7-zip.org/a/7zr.exe" -OutFile $sevenZip
    }
    return $sevenZip
}

function Expand-7zArchive {
    param(
        [string]$Archive,
        [string]$Destination,
        [string]$ToolsDir
    )
    $sevenZip = Get-7zr -ToolsDir $ToolsDir
    & $sevenZip x $Archive "-o$Destination" -y
    if ($LASTEXITCODE -ne 0) {
        throw "7-Zip failed to extract $Archive"
    }
}

function Find-ComfyPortableRoot {
    param([string]$InstallRoot)
    $direct = Join-Path $InstallRoot "ComfyUI_windows_portable"
    if (Test-Path -LiteralPath $direct) {
        return $direct
    }
    $nested = Get-ChildItem -LiteralPath $InstallRoot -Directory -Recurse -Depth 2 -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq "ComfyUI_windows_portable" } |
        Select-Object -First 1
    if ($nested) {
        return $nested.FullName
    }
    throw "Could not find ComfyUI_windows_portable after extraction."
}

function Get-EmbeddedPython {
    param([string]$ComfyPortableRoot)
    $candidates = @(
        (Join-Path $ComfyPortableRoot "python_embeded\python.exe"),
        (Join-Path $ComfyPortableRoot "python_embedded\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    throw "Could not find the embedded Python executable."
}

function Get-ComfyRoot {
    param([string]$ComfyPortableRoot)
    $candidate = Join-Path $ComfyPortableRoot "ComfyUI"
    if (Test-Path -LiteralPath $candidate) {
        return $candidate
    }
    throw "Could not find the ComfyUI folder inside the portable install."
}

function Get-RepoParts {
    param([string]$Repo)
    if ($Repo -notmatch "github\.com/([^/]+)/([^/]+?)(\.git)?$") {
        throw "Only GitHub repository URLs are supported by this installer: $Repo"
    }
    return @{ Owner = $Matches[1]; RepoName = $Matches[2] }
}

function Install-GitHubRepoZip {
    param(
        [string]$Repo,
        [string]$Branch,
        [string]$Destination,
        [string]$TempDir
    )

    $parts = Get-RepoParts -Repo $Repo
    $zipUrl = "https://github.com/$($parts.Owner)/$($parts.RepoName)/archive/refs/heads/$Branch.zip"
    $zipPath = Join-Path $TempDir "$($parts.RepoName)-$Branch.zip"
    $extractDir = Join-Path $TempDir "$($parts.RepoName)-$Branch"

    if (Test-Path -LiteralPath $extractDir) {
        Remove-Item -LiteralPath $extractDir -Recurse -Force
    }

    Invoke-FileDownload -Uri $zipUrl -OutFile $zipPath
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force

    $source = Get-ChildItem -LiteralPath $extractDir -Directory | Select-Object -First 1
    if (-not $source) {
        throw "Downloaded repo zip did not contain a folder: $Repo"
    }

    if (Test-Path -LiteralPath $Destination) {
        $backup = "$Destination.backup_$(Get-Date -Format yyyyMMdd_HHmmss)"
        Write-Warn "Existing folder found. Moving it to $backup"
        Move-Item -LiteralPath $Destination -Destination $backup
    }
    Move-Item -LiteralPath $source.FullName -Destination $Destination
}

function Install-CustomNodeRepo {
    param(
        [string]$Repo,
        [string]$Branch,
        [string]$Destination,
        [string]$TempDir
    )

    if (Get-Command git -ErrorAction SilentlyContinue) {
        if (Test-Path -LiteralPath (Join-Path $Destination ".git")) {
            Write-Host "Updating $(Split-Path -Leaf $Destination)"
            & git -C $Destination fetch origin $Branch
            & git -C $Destination checkout $Branch
            & git -C $Destination pull --ff-only origin $Branch
            if ($LASTEXITCODE -eq 0) {
                return
            }
            Write-Warn "Git update failed for $Destination. Leaving existing folder in place."
            return
        }
        if (-not (Test-Path -LiteralPath $Destination)) {
            Write-Host "Cloning $Repo"
            & git clone --depth 1 --branch $Branch $Repo $Destination
            if ($LASTEXITCODE -eq 0) {
                return
            }
            Write-Warn "Git clone failed for $Repo. Falling back to zip download."
        }
    }

    if (Test-Path -LiteralPath $Destination) {
        Write-Warn "Skipping $Destination because it already exists and is not a git checkout."
        return
    }
    Install-GitHubRepoZip -Repo $Repo -Branch $Branch -Destination $Destination -TempDir $TempDir
}

function Install-NodeRequirements {
    param(
        [string]$PythonExe,
        [string]$CustomNodesDir
    )

    $requirementFiles = Get-ChildItem -LiteralPath $CustomNodesDir -Directory |
        ForEach-Object { Join-Path $_.FullName "requirements.txt" } |
        Where-Object { Test-Path -LiteralPath $_ }

    & $PythonExe -m ensurepip --upgrade
    & $PythonExe -m pip install --upgrade pip setuptools wheel

    foreach ($requirements in $requirementFiles) {
        Write-Host "Installing Python requirements from $requirements"
        & $PythonExe -m pip install -r $requirements
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Requirement install had errors: $requirements"
        }
    }
}

function Install-FFmpeg {
    param(
        [string]$InstallRoot,
        [string]$ToolsDir
    )
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        Write-Host "FFmpeg is already on PATH."
        return (Split-Path -Parent (Get-Command ffmpeg).Source)
    }

    $ffmpegZip = Join-Path $ToolsDir "ffmpeg-release-essentials.zip"
    $ffmpegDir = Join-Path $InstallRoot "ffmpeg"
    Invoke-FileDownload -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $ffmpegZip

    if (Test-Path -LiteralPath $ffmpegDir) {
        Remove-Item -LiteralPath $ffmpegDir -Recurse -Force
    }
    Expand-Archive -LiteralPath $ffmpegZip -DestinationPath $ffmpegDir -Force

    $bin = Get-ChildItem -LiteralPath $ffmpegDir -Directory -Recurse |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "ffmpeg.exe") } |
        Select-Object -First 1

    if (-not $bin) {
        throw "Could not find ffmpeg.exe after extracting FFmpeg."
    }
    return $bin.FullName
}

function Test-VCRedistInstalled {
    $paths = @(
        "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
    )
    foreach ($path in $paths) {
        if (Test-Path $path) {
            $props = Get-ItemProperty $path
            if ($props.Installed -eq 1) {
                return $true
            }
        }
    }
    return $false
}

function Install-VCRedist {
    param([string]$ToolsDir)
    if (Test-VCRedistInstalled) {
        Write-Host "Microsoft Visual C++ 2015-2022 x64 runtime is already installed."
        return
    }

    $vcPath = Join-Path $ToolsDir "vc_redist.x64.exe"
    Invoke-FileDownload -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $vcPath
    Write-Host "Installing Microsoft Visual C++ runtime. Windows may ask for admin permission."
    Start-Process -FilePath $vcPath -ArgumentList "/install", "/quiet", "/norestart" -Wait
}

function New-Launchers {
    param(
        [string]$ComfyPortableRoot,
        [string]$FFmpegBin
    )

    $runNvidia = Join-Path $ComfyPortableRoot "run_nvidia_gpu.bat"
    $vrgdgRun = Join-Path $ComfyPortableRoot "run_vrgdg_nvidia_gpu.bat"
    $lines = @("@echo off")
    if ($FFmpegBin) {
        $lines += "set ""PATH=$FFmpegBin;%PATH%"""
    }
    $lines += "call ""$runNvidia"""
    Set-Content -LiteralPath $vrgdgRun -Value $lines -Encoding ASCII
}

function Write-InstallNotes {
    param(
        [string]$ComfyPortableRoot,
        [string]$ComfyRoot,
        [string]$PythonExe,
        [string]$FFmpegBin,
        [string]$Build
    )

    $notes = @"
VRGDG ComfyUI portable install notes
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

Portable build: $Build
ComfyUI portable folder: $ComfyPortableRoot
ComfyUI folder: $ComfyRoot
Python: $PythonExe
FFmpeg bin: $FFmpegBin

Start ComfyUI with:
  $ComfyPortableRoot\run_vrgdg_nvidia_gpu.bat

Installed VRGDG node repo:
  $VRGDGRepo

External custom nodes installed by this helper:
$($ExternalCustomNodes | ForEach-Object { "  - $($_.Name): $($_.Repo)" } | Out-String)

If a workflow still reports missing nodes, open ComfyUI Manager and use Install Missing Custom Nodes.
Models are not downloaded by this script because workflow model choices and license pages vary.
"@

    Set-Content -LiteralPath (Join-Path $ComfyPortableRoot "VRGDG_INSTALL_NOTES.txt") -Value $notes -Encoding UTF8
}

function Main {
    Require-Windows

    Write-Host "VRGDG ComfyUI Portable Installer" -ForegroundColor Magenta
    Write-Host "Default build: $PortableBuild"
    if ($PortableBuild -eq "cu126-py312") {
        Write-Host "This is the compatibility pick for VRGDG workflows and llama-cpp-python."
    }

    $root = Get-InstallRoot
    $toolsDir = Join-Path $root "_vrgdg_installer_tools"
    $tempDir = Join-Path $root "_vrgdg_installer_temp"
    New-Item -ItemType Directory -Force -Path $toolsDir, $tempDir | Out-Null

    Write-Step "Finding latest ComfyUI portable release"
    $asset = Get-PortableAsset -Build $PortableBuild
    Write-Host "Selected asset: $($asset.name)"

    $archive = Join-Path $tempDir $asset.name
    $portableRoot = Join-Path $root "ComfyUI_windows_portable"

    if (-not (Test-Path -LiteralPath $portableRoot)) {
        Write-Step "Downloading and extracting ComfyUI portable"
        Invoke-FileDownload -Uri $asset.browser_download_url -OutFile $archive
        Expand-7zArchive -Archive $archive -Destination $root -ToolsDir $toolsDir
    }
    else {
        Write-Warn "ComfyUI_windows_portable already exists. The installer will update custom nodes and requirements only."
    }

    $portableRoot = Find-ComfyPortableRoot -InstallRoot $root
    $comfyRoot = Get-ComfyRoot -ComfyPortableRoot $portableRoot
    $pythonExe = Get-EmbeddedPython -ComfyPortableRoot $portableRoot
    $customNodesDir = Join-Path $comfyRoot "custom_nodes"
    New-Item -ItemType Directory -Force -Path $customNodesDir | Out-Null

    Write-Step "Installing VRGDG custom nodes from main branch"
    Install-CustomNodeRepo -Repo $VRGDGRepo -Branch "main" -Destination (Join-Path $customNodesDir "comfyui-vrgamedevgirl") -TempDir $tempDir

    if (-not $SkipExternalCustomNodes) {
        Write-Step "Installing helpful external custom nodes"
        foreach ($node in $ExternalCustomNodes) {
            Write-Host "$($node.Name): $($node.RequiredFor)"
            Install-CustomNodeRepo -Repo $node.Repo -Branch $node.Branch -Destination (Join-Path $customNodesDir $node.Name) -TempDir $tempDir
        }
    }

    Write-Step "Installing Python requirements"
    Install-NodeRequirements -PythonExe $pythonExe -CustomNodesDir $customNodesDir

    $ffmpegBin = ""
    if (-not $SkipFFmpeg) {
        Write-Step "Checking FFmpeg"
        $ffmpegBin = Install-FFmpeg -InstallRoot $root -ToolsDir $toolsDir
    }

    if (-not $SkipVCRedist) {
        Write-Step "Checking Microsoft Visual C++ runtime"
        Install-VCRedist -ToolsDir $toolsDir
    }

    Write-Step "Creating VRGDG launch helper"
    New-Launchers -ComfyPortableRoot $portableRoot -FFmpegBin $ffmpegBin
    Write-InstallNotes -ComfyPortableRoot $portableRoot -ComfyRoot $comfyRoot -PythonExe $pythonExe -FFmpegBin $ffmpegBin -Build $PortableBuild

    Write-Step "Quick environment check"
    $checkCode = @'
import sys
print("Python:", sys.version.replace("\n", " "))
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Torch check failed:", exc)
'@
    & $pythonExe -c $checkCode

    Write-Host ""
    Write-Host "Done. Start ComfyUI with:" -ForegroundColor Green
    Write-Host "  $portableRoot\run_vrgdg_nvidia_gpu.bat"

    if (-not $NoStart) {
        $answer = Read-Host "Start ComfyUI now? Type Y and press Enter"
        if ($answer -match "^[Yy]") {
            Start-Process -FilePath (Join-Path $portableRoot "run_vrgdg_nvidia_gpu.bat") -WorkingDirectory $portableRoot
        }
    }
}

try {
    Main
}
catch {
    Write-Host ""
    Write-Host "Installer failed:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Tip: send the text above to VRGameDevGirl so she can see exactly where setup stopped."
    exit 1
}
