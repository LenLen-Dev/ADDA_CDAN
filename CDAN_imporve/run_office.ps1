param(
    [string]$CudaVisibleDevices = "0"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices

function Invoke-Training {
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "python $($args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    # ResNet50, Office31, Single Source
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/Office31_A2W
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/Office31_D2W
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/Office31_W2D
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/Office31_A2D
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/Office31_D2A
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/Office31_W2A

    # ResNet50, Office-Home, Single Source
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Ar2Cl
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Ar2Pr
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Ar2Rw
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Cl2Ar
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Cl2Pr
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Cl2Rw
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Pr2Ar
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Pr2Cl
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Pr2Rw
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Rw2Ar
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Rw2Cl
    Invoke-Training CDAN_imporve/cdan_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 30 --seed 0 --condition-dim 1024 --max-temperature 2.0 --min-temperature 1.0 --log logs/tlc_cdan/OfficeHome_Rw2Pr
}
finally {
    Pop-Location
}
