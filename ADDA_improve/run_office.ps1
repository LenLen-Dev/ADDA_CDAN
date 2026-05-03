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
    # ResNet18, Office31, Single Source
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s A -t W -a resnet18 --epochs 20 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 1 --log logs/pl_epa_adda/Office31_A2W
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s D -t W -a resnet18 --epochs 20 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 1 --log logs/pl_epa_adda/Office31_D2W
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s W -t D -a resnet18 --epochs 20 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 1 --log logs/pl_epa_adda/Office31_W2D
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s A -t D -a resnet18 --epochs 20 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 1 --log logs/pl_epa_adda/Office31_A2D
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s D -t A -a resnet18 --epochs 20 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 1 --log logs/pl_epa_adda/Office31_D2A
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office31 -d Office31 -s W -t A -a resnet18 --epochs 20 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 1 --log logs/pl_epa_adda/Office31_W2A

    # ResNet18, Office-Home, Single Source
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Ar -t Cl -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Ar2Cl
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Ar -t Pr -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Ar2Pr
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Ar -t Rw -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Ar2Rw
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Cl -t Ar -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Cl2Ar
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Cl -t Pr -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Cl2Pr
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Cl -t Rw -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Cl2Rw
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Pr -t Ar -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Pr2Ar
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Pr -t Cl -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Pr2Cl
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Pr -t Rw -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Pr2Rw
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Rw -t Ar -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Rw2Ar
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Rw -t Cl -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Rw2Cl
    Invoke-Training ADDA_improve/adda_improve.py examples/domain_adaptation/image_classification/data/office-home -d OfficeHome -s Rw -t Pr -a resnet18 --epochs 30 --pretrain-epochs 3 --pseudo-threshold 0.95 --pseudo-label-weight 0.1 --entropy-weight 0.01 --prototype-weight 0.1 --regularizer-start-epoch 1 --regularizer-rampup-epochs 5 --seed 0 --log logs/pl_epa_adda/OfficeHome_Rw2Pr
}
finally {
    Pop-Location
}
