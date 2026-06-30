param(
    [string]$TargetRoot = "A:\MUSUBI\musubi-tuner-ltx2",
    [string]$SourceRoot = "A:\MUSUBI\_upstream_musubi_tuner_krea2",
    [string]$RepoUrl = "https://github.com/kohya-ss/musubi-tuner.git",
    [string]$Branch = "main",
    [switch]$DryRun,
    [switch]$SkipClone,
    [switch]$NoSharedPatch,
    [switch]$NoVerify
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Do {
    param([string]$Message)
    if ($DryRun) {
        Write-Host "[dry-run] $Message" -ForegroundColor DarkYellow
    } else {
        Write-Host $Message
    }
}

function Ensure-Directory {
    param([string]$Path)
    if ($DryRun) {
        Write-Do "Create directory: $Path"
        return
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Run-Command {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory = (Get-Location).Path
    )
    $line = "$FilePath $($Arguments -join ' ')"
    Write-Do $line
    if ($DryRun) {
        return
    }

    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Command failed with exit code $($process.ExitCode): $line"
    }
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

function Get-RelativePathCompat {
    param(
        [string]$BasePath,
        [string]$ChildPath
    )

    if ([System.IO.Path].GetMethod("GetRelativePath", [type[]]@([string], [string]))) {
        return [System.IO.Path]::GetRelativePath($BasePath, $ChildPath)
    }

    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    $childFull = [System.IO.Path]::GetFullPath($ChildPath)
    if (-not $baseFull.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $baseFull += [System.IO.Path]::DirectorySeparatorChar
    }

    $baseUri = [System.Uri]::new($baseFull)
    $childUri = [System.Uri]::new($childFull)
    $relativeUri = $baseUri.MakeRelativeUri($childUri)
    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace("/", [System.IO.Path]::DirectorySeparatorChar)
}

function Backup-TargetPath {
    param(
        [string]$TargetPath,
        [string]$BackupRoot
    )
    if (-not (Test-Path -LiteralPath $TargetPath)) {
        return
    }

    $relative = Get-RelativePathCompat -BasePath (Resolve-Path -LiteralPath $TargetRoot).Path -ChildPath (Resolve-Path -LiteralPath $TargetPath).Path
    $backupPath = Join-Path $BackupRoot $relative
    $backupParent = Split-Path -Parent $backupPath
    Ensure-Directory $backupParent

    if ($DryRun) {
        Write-Do "Backup $TargetPath -> $backupPath"
        return
    }

    if (Test-Path -LiteralPath $backupPath) {
        return
    }

    Copy-Item -LiteralPath $TargetPath -Destination $backupPath -Recurse -Force
}

function Copy-BackportPath {
    param(
        [string]$RelativePath,
        [string]$BackupRoot
    )

    $sourcePath = Join-Path $SourceRoot $RelativePath
    $targetPath = Join-Path $TargetRoot $RelativePath
    Assert-PathExists $sourcePath "Upstream backport source"

    Backup-TargetPath -TargetPath $targetPath -BackupRoot $BackupRoot
    Ensure-Directory (Split-Path -Parent $targetPath)

    Write-Do "Copy $sourcePath -> $targetPath"
    if (-not $DryRun) {
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Recurse -Force
    }
}

function Write-TextFile {
    param(
        [string]$Path,
        [string]$Content
    )
    Ensure-Directory (Split-Path -Parent $Path)
    Write-Do "Write $Path"
    if (-not $DryRun) {
        Set-Content -LiteralPath $Path -Value $Content -Encoding UTF8
    }
}

function Save-PatchedFile {
    param(
        [string]$Path,
        [string]$Content,
        [string]$BackupRoot
    )
    Backup-TargetPath -TargetPath $Path -BackupRoot $BackupRoot
    Write-Do "Patch $Path"
    if (-not $DryRun) {
        Set-Content -LiteralPath $Path -Value $Content -Encoding UTF8
    }
}

function Add-Krea2SharedCompatibility {
    param([string]$BackupRoot)

    $datasetPath = Join-Path $TargetRoot "src\musubi_tuner\dataset\image_video_dataset.py"
    Assert-PathExists $datasetPath "Target dataset file"

    $content = Get-Content -LiteralPath $datasetPath -Raw
    $changed = $false

    if ($content -notmatch "ARCHITECTURE_KREA2\s*=") {
        $content = $content -replace '(ARCHITECTURE_Z_IMAGE_FULL\s*=\s*"[^"]+"\r?\n)', "`$1ARCHITECTURE_KREA2 = `"k2`"`r`nARCHITECTURE_KREA2_FULL = `"krea2`"`r`n"
        $changed = $true
    }

    if ($content -notmatch "RESOLUTION_STEPS_KREA2\s*=") {
        $content = $content -replace '(RESOLUTION_STEPS_Z_IMAGE\s*=\s*\d+\r?\n)', "`$1    RESOLUTION_STEPS_KREA2 = 16`r`n"
        $changed = $true
    }

    if ($content -notmatch "ARCHITECTURE_KREA2:\s*RESOLUTION_STEPS_KREA2") {
        $content = $content -replace '(ARCHITECTURE_Z_IMAGE:\s*RESOLUTION_STEPS_Z_IMAGE,\r?\n)', "`$1        ARCHITECTURE_KREA2: RESOLUTION_STEPS_KREA2,`r`n"
        $changed = $true
    }

    if ($content -notmatch "def save_latent_cache_krea2\(") {
        $latentFunction = @'

def save_latent_cache_krea2(item_info: ItemInfo, latent: torch.Tensor):
    """Krea 2 architecture. Text-to-image target latent only."""
    assert latent.dim() == 4, "latent should be 4D tensor (channel, frame, height, width)"

    C, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_KREA2_FULL)

'@
        $content = $content -replace '(\r?\ndef save_latent_cache_common\(item_info: ItemInfo, sd: dict\[str, torch\.Tensor\], arch_fullname: str\):)', "$latentFunction`$1"
        $changed = $true
    }

    if ($content -notmatch "def save_text_encoder_output_cache_krea2\(") {
        $textEncoderFunction = @'

def save_text_encoder_output_cache_krea2(item_info: ItemInfo, embed: torch.Tensor):
    """Krea 2 architecture. Varlen Qwen3-VL hidden-state stack."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_krea2_vl_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_KREA2_FULL)

'@
        $content = $content -replace '(\r?\ndef save_text_encoder_output_cache_common\(item_info: ItemInfo, sd: dict\[str, torch\.Tensor\], arch_fullname: str\):)', "$textEncoderFunction`$1"
        $changed = $true
    }

    if ($changed) {
        Save-PatchedFile -Path $datasetPath -Content $content -BackupRoot $BackupRoot
    } else {
        Write-Host "Krea 2 dataset compatibility already present."
    }

    $architecturesPath = Join-Path $TargetRoot "src\musubi_tuner\dataset\architectures.py"
    if (-not (Test-Path -LiteralPath $architecturesPath)) {
        $architecturesContent = @'
"""Compatibility constants for newer musubi-tuner architecture-specific scripts.

This file is generated by VRGDG's Krea 2 backport script for older tuner installs
that still keep architecture constants in image_video_dataset.py.
"""

from musubi_tuner.dataset.image_video_dataset import *  # noqa: F401,F403

'@
        Write-TextFile -Path $architecturesPath -Content $architecturesContent
    } else {
        Write-Host "Architecture compatibility module already exists: $architecturesPath"
    }
}

function Add-Krea2MetadataCompatibility {
    param([string]$BackupRoot)

    $metadataPath = Join-Path $TargetRoot "src\musubi_tuner\utils\sai_model_spec.py"
    Assert-PathExists $metadataPath "Target SAI model metadata file"

    $content = Get-Content -LiteralPath $metadataPath -Raw
    $changed = $false

    if ($content -notmatch "ARCHITECTURE_KREA2") {
        $content = $content -replace '(\s+ARCHITECTURE_LTX2,\r?\n\s+ARCHITECTURE_Z_IMAGE,)', "`$1`r`n    ARCHITECTURE_KREA2,"
        $changed = $true
    }

    if ($content -notmatch "ARCH_KREA2\s*=") {
        $content = $content -replace '(ARCH_Z_IMAGE\s*=\s*"Z-Image"\r?\n)', "`$1ARCH_KREA2 = `"Krea-2`"`r`n"
        $changed = $true
    }

    if ($content -notmatch "IMPL_KREA2\s*=") {
        $content = $content -replace '(IMPL_Z_IMAGE\s*=\s*"https://github\.com/Tongyi-MAI/Z-Image"\r?\n)', "`$1IMPL_KREA2 = `"https://github.com/krea-ai/krea-2`"`r`n"
        $changed = $true
    }

    if ($content -notmatch "architecture == ARCHITECTURE_KREA2") {
        $content = $content -replace '(elif architecture == ARCHITECTURE_Z_IMAGE:\r?\n\s+arch = ARCH_Z_IMAGE\r?\n\s+impl = IMPL_Z_IMAGE\r?\n)', "`$1    elif architecture == ARCHITECTURE_KREA2:`r`n        arch = ARCH_KREA2`r`n        impl = IMPL_KREA2`r`n"
        $changed = $true
    }

    if ($changed) {
        Save-PatchedFile -Path $metadataPath -Content $content -BackupRoot $BackupRoot
    } else {
        Write-Host "Krea 2 SAI metadata compatibility already present."
    }
}

function Add-TrainingBaseCompatibility {
    param([string]$BackupRoot)

    $trainerPath = Join-Path $TargetRoot "src\musubi_tuner\hv_train_network.py"
    Assert-PathExists $trainerPath "Target shared trainer file"

    $content = Get-Content -LiteralPath $trainerPath -Raw
    if ($content -match "class\s+DiTOutput\b") {
        Write-Host "DiTOutput compatibility shim already present."
        return
    }

    $shim = @'


class DiTOutput:
    """Compatibility wrapper for newer architecture trainers.

    Older VRGDG/musubi training loops unpack call_dit() as (pred, target).
    Newer architecture modules return DiTOutput(pred=..., target=..., extra=...).
    Iteration preserves the old unpacking behavior.
    """

    def __init__(self, pred, target, extra=None):
        self.pred = pred
        self.target = target
        self.extra = extra or {}

    def __iter__(self):
        yield self.pred
        yield self.target

'@

    if ($content -match "SharedEpoch\s*=") {
        $content = $content -replace '(\r?\nSharedEpoch\s*=)', "$shim`$1"
    } elseif ($content -match "logger\s*=\s*logging\.getLogger") {
        $content = $content -replace '(\r?\nlogger\s*=\s*logging\.getLogger)', "$shim`$1"
    } else {
        throw "Could not find a safe insertion point for DiTOutput in $trainerPath"
    }

    Save-PatchedFile -Path $trainerPath -Content $content -BackupRoot $BackupRoot
}

function Add-Krea2ModelCompatibility {
    param([string]$BackupRoot)

    $kreaModelPath = Join-Path $TargetRoot "src\musubi_tuner\krea2\krea2_mmdit.py"
    Assert-PathExists $kreaModelPath "Target Krea 2 MMDiT file"

    $content = Get-Content -LiteralPath $kreaModelPath -Raw
    if ($content -match "def\s+enable_gradient_checkpointing\(\s*self,\s*cpu_offload:\s*bool\s*=\s*False,\s*blocks_to_checkpoint=None,\s*\*\*kwargs\s*\)") {
        Write-Host "Krea 2 gradient checkpointing compatibility already present."
        return
    }

    $patched = $content -replace "def\s+enable_gradient_checkpointing\(\s*self,\s*cpu_offload:\s*bool\s*=\s*False\s*\):", "def enable_gradient_checkpointing(self, cpu_offload: bool = False, blocks_to_checkpoint=None, **kwargs):"
    if ($patched -eq $content) {
        throw "Could not patch Krea 2 enable_gradient_checkpointing signature in $kreaModelPath"
    }

    Save-PatchedFile -Path $kreaModelPath -Content $patched -BackupRoot $BackupRoot
}

function Test-Krea2Backport {
    $pythonCandidates = @(
        (Join-Path $TargetRoot "venv\Scripts\python.exe"),
        (Join-Path $TargetRoot ".venv\Scripts\python.exe")
    )
    $python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $python) {
        Write-Warning "Could not find target venv Python. Skipping import verification."
        return
    }

    $checks = @(
        "import musubi_tuner.dataset.architectures as a; print('krea arch', a.ARCHITECTURE_KREA2, a.ARCHITECTURE_KREA2_FULL)",
        "from musubi_tuner.dataset.image_video_dataset import save_latent_cache_krea2, save_text_encoder_output_cache_krea2; print('krea cache helpers ok')",
        "from musubi_tuner.utils import sai_model_spec; from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_KREA2; print(sai_model_spec.build_metadata(None, ARCHITECTURE_KREA2, 0)['modelspec.architecture'])",
        "import musubi_tuner.networks.lora_krea2; print('lora_krea2 ok')",
        "import musubi_tuner.krea2_train_network; print('krea2_train_network ok')"
    )

    foreach ($code in $checks) {
        Write-Host "Verify: $code" -ForegroundColor Cyan
        & $python -c $code
        if ($LASTEXITCODE -ne 0) {
            throw "Verification failed: $code"
        }
    }
}

$TargetRoot = [System.IO.Path]::GetFullPath($TargetRoot)
$SourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)

Write-Step "Validate target tuner"
Assert-PathExists $TargetRoot "Target musubi root"
Assert-PathExists (Join-Path $TargetRoot "zimage_train_network.py") "Target Z-Image trainer wrapper"
Assert-PathExists (Join-Path $TargetRoot "src\musubi_tuner\zimage") "Target Z-Image package"

if (-not $SkipClone) {
    Write-Step "Clone or update upstream musubi-tuner"
    if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot ".git"))) {
        Ensure-Directory (Split-Path -Parent $SourceRoot)
        Run-Command -FilePath "git" -Arguments @("clone", "--depth", "1", "--branch", $Branch, $RepoUrl, $SourceRoot)
    } else {
        Run-Command -FilePath "git" -Arguments @("-C", $SourceRoot, "fetch", "--depth", "1", "origin", $Branch)
        Run-Command -FilePath "git" -Arguments @("-C", $SourceRoot, "checkout", $Branch)
        Run-Command -FilePath "git" -Arguments @("-C", $SourceRoot, "pull", "--ff-only", "origin", $Branch)
    }
}

if ($DryRun -and -not (Test-Path -LiteralPath $SourceRoot)) {
    Write-Step "Dry run stopped before upstream validation"
    Write-Host "The upstream clone does not exist yet. Re-run without -DryRun to clone and apply, or pre-clone into SourceRoot and run again with -DryRun -SkipClone." -ForegroundColor Yellow
    exit 0
}

Write-Step "Validate upstream Krea 2 files"
$requiredUpstream = @(
    "krea2_cache_latents.py",
    "krea2_cache_text_encoder_outputs.py",
    "krea2_train_network.py",
    "src\musubi_tuner\krea2_cache_latents.py",
    "src\musubi_tuner\krea2_cache_text_encoder_outputs.py",
    "src\musubi_tuner\krea2_train_network.py",
    "src\musubi_tuner\krea2",
    "src\musubi_tuner\modules\attention.py",
    "src\musubi_tuner\modules\custom_offloading_utils.py",
    "src\musubi_tuner\networks\lora_krea2.py"
)
foreach ($relative in $requiredUpstream) {
    Assert-PathExists (Join-Path $SourceRoot $relative) "Required upstream file"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupRoot = Join-Path $TargetRoot "_vrgdg_backups\krea2_backport_$stamp"
Write-Step "Backup root"
Write-Host $backupRoot -ForegroundColor Green

Write-Step "Copy Krea 2-specific upstream files"
$copyPaths = @(
    "krea2_cache_latents.py",
    "krea2_cache_text_encoder_outputs.py",
    "krea2_generate_image.py",
    "krea2_train_network.py",
    "src\musubi_tuner\krea2_cache_latents.py",
    "src\musubi_tuner\krea2_cache_text_encoder_outputs.py",
    "src\musubi_tuner\krea2_generate_image.py",
    "src\musubi_tuner\krea2_train_network.py",
    "src\musubi_tuner\krea2",
    "src\musubi_tuner\modules\attention.py",
    "src\musubi_tuner\modules\custom_offloading_utils.py",
    "src\musubi_tuner\networks\lora_krea2.py",
    "docs\krea2.md"
)
foreach ($relative in $copyPaths) {
    if (Test-Path -LiteralPath (Join-Path $SourceRoot $relative)) {
        Copy-BackportPath -RelativePath $relative -BackupRoot $backupRoot
    } else {
        Write-Warning "Optional upstream path not found, skipping: $relative"
    }
}

if (-not $NoSharedPatch) {
    Write-Step "Patch shared compatibility in target tuner"
    Add-Krea2SharedCompatibility -BackupRoot $backupRoot
    Add-Krea2MetadataCompatibility -BackupRoot $backupRoot
    Add-TrainingBaseCompatibility -BackupRoot $backupRoot
    Add-Krea2ModelCompatibility -BackupRoot $backupRoot
} else {
    Write-Warning "Skipping shared patch. Krea 2 scripts will likely fail on older tuner layouts."
}

if (-not $NoVerify -and -not $DryRun) {
    Write-Step "Verify Python imports"
    Test-Krea2Backport
}

Write-Step "Done"
if ($DryRun) {
    Write-Host "Dry run complete. Re-run without -DryRun to apply." -ForegroundColor Yellow
} else {
    Write-Host "Krea 2 backport applied. Backup saved under: $backupRoot" -ForegroundColor Green
}
