param(
    [string]$CudaVisibleDevices = "0"
)

$ErrorActionPreference = "Stop"

$Scripts = @(
    @{
        Name = "ADDA"
        Path = Join-Path $PSScriptRoot "examples/domain_adaptation/image_classification/adda.ps1"
    },
    @{
        Name = "ADDA Improve"
        Path = Join-Path $PSScriptRoot "ADDA_improve/run_office.ps1"
    },
    @{
        Name = "CDAN"
        Path = Join-Path $PSScriptRoot "examples/domain_adaptation/image_classification/cdan.ps1"
    },
    @{
        Name = "CDAN Improve"
        Path = Join-Path $PSScriptRoot "CDAN_imporve/run_office.ps1"
    }
)

foreach ($Script in $Scripts) {
    if (-not (Test-Path $Script.Path)) {
        throw "Script not found: $($Script.Path)"
    }

    Write-Host ""
    Write-Host "===== Running $($Script.Name) ====="
    Write-Host $Script.Path

    & $Script.Path -CudaVisibleDevices $CudaVisibleDevices

    if ($LASTEXITCODE -ne 0) {
        throw "$($Script.Name) failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "All Office31 and OfficeHome runs completed."
