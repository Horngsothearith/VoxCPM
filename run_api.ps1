<#
.SYNOPSIS
    Starts the VoxCPM Local API server in PowerShell.
.EXAMPLE
    .\run_api.ps1
    .\run_api.ps1 -Port 8080 -Device cpu
#>
param (
    [string]$HostIP = "0.0.0.0",
    [int]$Port = 8000,
    [string]$Device = "auto",
    [string]$ModelId = "openbmb/VoxCPM2",
    [switch]$LoadDenoiser,
    [switch]$Preload
)

$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment: $VenvPython" -ForegroundColor Green
    $PyCmd = $VenvPython
} else {
    Write-Host "Using system python" -ForegroundColor Yellow
    $PyCmd = "python"
}

$ArgsList = @("api_server.py", "--host", $HostIP, "--port", $Port, "--device", $Device, "--model-id", $ModelId)
if ($LoadDenoiser) { $ArgsList += "--load-denoiser" }
if ($Preload) { $ArgsList += "--preload" }

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "           VoxCPM Local API Server Starting             " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Web UI Playground : http://localhost:$Port/ui" -ForegroundColor White
Write-Host " OpenAPI / Docs    : http://localhost:$Port/docs" -ForegroundColor White
Write-Host " OpenAI Endpoint   : http://localhost:$Port/v1/audio/speech" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Cyan

& $PyCmd $ArgsList
