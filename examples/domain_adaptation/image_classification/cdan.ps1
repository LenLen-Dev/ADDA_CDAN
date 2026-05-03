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

Push-Location $PSScriptRoot
try {
    # ResNet50, Office31, Single Source
    Invoke-Training cdan.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 2 --log logs/cdan/Office31_A2W
    Invoke-Training cdan.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 2 --log logs/cdan/Office31_D2W
    Invoke-Training cdan.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 2 --log logs/cdan/Office31_W2D
    Invoke-Training cdan.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 2 --log logs/cdan/Office31_A2D
    Invoke-Training cdan.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 2 --log logs/cdan/Office31_D2A
    Invoke-Training cdan.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 2 --log logs/cdan/Office31_W2A

    # ResNet50, Office-Home, Single Source
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Ar2Cl
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Ar2Pr
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Ar2Rw
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Cl2Ar
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Cl2Pr
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Cl2Rw
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Pr2Ar
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Pr2Cl
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Pr2Rw
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Rw2Ar
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Rw2Cl
    Invoke-Training cdan.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Rw2Pr

    # Vision Transformer, Office-Home, Single Source
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Ar -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Ar2Cl
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Ar -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Ar2Pr
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Ar -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Ar2Rw
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Cl -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Cl2Ar
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Cl -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Cl2Pr
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Cl -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Cl2Rw
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Pr -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Pr2Ar
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Pr -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Pr2Cl
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Pr -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Pr2Rw
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Rw -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Rw2Ar
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Rw -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Rw2Cl
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Rw -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Rw2Pr

    # ResNet50, Office-Home, Multi Source
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Cl Pr Rw -t Ar -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_multi2Ar
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Ar Pr Rw -t Cl -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_multi2Cl
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_multi2Pr
    # Invoke-Training cdan.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_multi2Rw
}
finally {
    Pop-Location
}
