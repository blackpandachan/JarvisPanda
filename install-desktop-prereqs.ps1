# Run this in an ADMIN PowerShell terminal:
#   Set-ExecutionPolicy Bypass -Scope Process -Force
#   .\install-desktop-prereqs.ps1
#
# Installs: Rust (rustup), Node.js LTS, MSVC Build Tools (C++ required by Rust)
# WebView2 already detected on this system — no action needed.

$ErrorActionPreference = "Stop"
$tmp = "$env:TEMP\jarvis-setup"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# ── 1. MSVC Build Tools ────────────────────────────────────────────────────────
Write-Host "`n[1/3] Installing Visual Studio Build Tools (C++ workload)..." -ForegroundColor Cyan
$vsInstaller = "$tmp\vs_buildtools.exe"
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_buildtools.exe" -OutFile $vsInstaller
# Silent install: C++ build tools workload only (~3 GB, takes a few minutes)
Start-Process -FilePath $vsInstaller -ArgumentList @(
    "--quiet", "--wait", "--norestart",
    "--add", "Microsoft.VisualStudio.Workload.VCTools",
    "--includeRecommended"
) -Wait
Write-Host "  MSVC build tools installed." -ForegroundColor Green

# ── 2. Rust (rustup) ──────────────────────────────────────────────────────────
Write-Host "`n[2/3] Installing Rust via rustup..." -ForegroundColor Cyan
$rustupInit = "$tmp\rustup-init.exe"
Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustupInit
# -y: no prompts, default stable toolchain
Start-Process -FilePath $rustupInit -ArgumentList "-y" -Wait
# Add cargo to PATH for this session
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
Write-Host "  Rust installed: $(rustc --version)" -ForegroundColor Green

# ── 3. Node.js LTS ────────────────────────────────────────────────────────────
Write-Host "`n[3/3] Installing Node.js LTS..." -ForegroundColor Cyan
# Get latest LTS version number
$nodeIndex = Invoke-RestMethod "https://nodejs.org/dist/index.json"
$lts = $nodeIndex | Where-Object { $_.lts } | Select-Object -First 1
$nodeVersion = $lts.version
$nodeMsi = "$tmp\node-$nodeVersion-x64.msi"
Invoke-WebRequest -Uri "https://nodejs.org/dist/$nodeVersion/node-$nodeVersion-x64.msi" -OutFile $nodeMsi
Start-Process msiexec.exe -ArgumentList "/i `"$nodeMsi`" /quiet /norestart" -Wait
# Reload PATH
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
Write-Host "  Node.js installed: $(node --version)" -ForegroundColor Green

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host "`n=== All prerequisites installed! ===" -ForegroundColor Green
Write-Host "Close and reopen your terminal so PATH updates take effect, then run:"
Write-Host "  cd C:\GitHub\PandaJarvis\desktop"
Write-Host "  npm install"
Write-Host "  npm run dev"
Write-Host ""
Write-Host "Make sure jarvis-server (Docker) is running on port 8000 first."
