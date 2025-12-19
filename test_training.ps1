# Test Training Script
# Runs 2 epochs for each model on electronics dataset
# Purpose: Quick verification before full pipeline run

param(
    [int]$Epochs = 2,
    [string]$Dataset = "electronics",
    [switch]$SkipAblation
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Test Training (Quick Verification)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Dataset: $Dataset" -ForegroundColor Yellow
Write-Host "Epochs: $Epochs" -ForegroundColor Yellow
Write-Host ""

$Models = @("lattice", "micro", "diffmm")
$failedRuns = @()

function Clear-Vram {
    python -c "import torch; torch.cuda.empty_cache(); print(f'VRAM cleared. Available: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB')"
}

# Verify data exists
$dataPath = "data/processed/$Dataset"
if (-not (Test-Path "$dataPath/train.txt")) {
    Write-Host "[ERROR] No preprocessed data for $Dataset" -ForegroundColor Red
    Write-Host "Run: python src/preprocessing/run_preprocessing.py --dataset $Dataset"
    exit 1
}

Write-Host "[OK] Preprocessed data found at $dataPath" -ForegroundColor Green
Write-Host ""

# ============================================================
# PHASE 1: Main Training
# ============================================================
Write-Host "============================================" -ForegroundColor Green
Write-Host " PHASE 1: MAIN TRAINING" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

foreach ($model in $Models) {
    Write-Host "-------------------------------------------" -ForegroundColor Gray
    Write-Host "Testing: $model" -ForegroundColor Cyan
    Write-Host "-------------------------------------------" -ForegroundColor Gray
    
    $startTime = Get-Date
    
    try {
        python src/main.py --model $model --dataset $Dataset --epochs $Epochs --output-dir checkpoints_test
        
        $duration = (Get-Date) - $startTime
        Write-Host "[PASS] $model completed in $($duration.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
    }
    catch {
        Write-Host "[FAIL] $model failed: $_" -ForegroundColor Red
        $failedRuns += $model
    }
    
    Clear-Vram
    Write-Host ""
}

# ============================================================
# PHASE 2: Ablation Testing
# ============================================================
if (-not $SkipAblation) {
    Write-Host "============================================" -ForegroundColor Magenta
    Write-Host " PHASE 2: ABLATION TESTING" -ForegroundColor Magenta
    Write-Host "============================================" -ForegroundColor Magenta
    
    $ablationModes = @("no_visual", "no_text")
    
    foreach ($model in $Models) {
        foreach ($ablation in $ablationModes) {
            Write-Host "-------------------------------------------" -ForegroundColor Gray
            Write-Host "Ablation: $model ($ablation)" -ForegroundColor Cyan
            Write-Host "-------------------------------------------" -ForegroundColor Gray
            
            $startTime = Get-Date
            
            try {
                python src/main.py --model $model --dataset $Dataset --epochs $Epochs --ablation $ablation --output-dir checkpoints_test
                
                $duration = (Get-Date) - $startTime
                Write-Host "[PASS] $model/$ablation completed in $($duration.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
            }
            catch {
                Write-Host "[FAIL] $model/$ablation failed: $_" -ForegroundColor Red
                $failedRuns += "$model/$ablation"
            }
            
            Clear-Vram
            Write-Host ""
        }
    }
}

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " TEST RESULTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($failedRuns.Count -eq 0) {
    Write-Host "[SUCCESS] All tests passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Ready for full run:" -ForegroundColor Yellow
    Write-Host "  .\run_pipeline.ps1"
} else {
    Write-Host "[FAILED] Some tests failed:" -ForegroundColor Red
    foreach ($run in $failedRuns) {
        Write-Host "  - $run" -ForegroundColor Red
    }
    exit 1
}
