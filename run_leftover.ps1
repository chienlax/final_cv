# Leftover Pipeline Script
# Runs only the MISSING experiments (skips already completed runs)
# - Ablation: LATTICE and DiffMM on all datasets (MICRO already done)
# - Sensitivity: All datasets × all models × all params
# - Visualization: All datasets × all models

param(
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

# ============================================================
# Helper Functions
# ============================================================

function Clear-CudaVram {
    Write-Host "[CLEANUP] Clearing CUDA VRAM..." -ForegroundColor Magenta
    python -c "import torch; torch.cuda.empty_cache(); print(f'VRAM freed. Available: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB')"
    Start-Sleep -Seconds 2
}

function Show-GpuMemory {
    python -c "import torch; used = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0])/1024**3; total = torch.cuda.mem_get_info()[1]/1024**3; print(f'GPU Memory: {used:.1f}/{total:.1f} GB')"
}

# Configuration
$Datasets = @("electronics", "beauty", "clothing")
$Models = @("lattice", "micro", "diffmm")
$AblationModels = @("lattice", "diffmm")  # MICRO already done
$ablationModes = @("no_visual", "no_text")
$failedRuns = @()

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Leftover Experiments Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script runs ONLY the missing experiments:" -ForegroundColor Yellow
Write-Host "  - Ablation: LATTICE, DiffMM (MICRO already done)" -ForegroundColor Yellow
Write-Host "  - Sensitivity: All models x all datasets" -ForegroundColor Yellow
Write-Host "  - Visualization: All models x all datasets" -ForegroundColor Yellow
Write-Host ""

# ============================================================
# PART 1: Ablation for LATTICE and DiffMM (MICRO already done)
# ============================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Magenta
Write-Host " PART 1: ABLATION (LATTICE + DiffMM only)" -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Magenta

$totalAblationRuns = $Datasets.Count * $AblationModels.Count * $ablationModes.Count
$currentAblationRun = 0

Write-Host "Total ablation runs: $totalAblationRuns (skipping MICRO - already done)"

foreach ($dataset in $Datasets) {
    foreach ($model in $AblationModels) {
        foreach ($ablation in $ablationModes) {
            $currentAblationRun++
            
            Write-Host ""
            Write-Host ">>> [$currentAblationRun/$totalAblationRuns] Ablation: $model on $dataset ($ablation)" -ForegroundColor Cyan
            Write-Host "-------------------------------------------"
            
            Show-GpuMemory
            $startTime = Get-Date
            
            try {
                python src/main.py --model $model --dataset $dataset --ablation $ablation --seed $Seed
                
                $duration = (Get-Date) - $startTime
                Write-Host "[SUCCESS] $model/$ablation on $dataset completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
            }
            catch {
                Write-Host "[ERROR] Ablation failed: $_" -ForegroundColor Red
                $failedRuns += "ablation_${model}_${ablation}/${dataset}"
            }
            
            Clear-CudaVram
        }
    }
}

Write-Host ""
Write-Host "[DONE] Ablation (LATTICE + DiffMM) complete!" -ForegroundColor Green

# ============================================================
# PART 2: Sensitivity Analysis (ALL datasets × ALL models)
# ============================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Yellow
Write-Host " PART 2: SENSITIVITY ANALYSIS" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow

Write-Host "Running sensitivity on ALL datasets with Config.EPOCHS..."
$startTime = Get-Date

try {
    python scripts/run_sensitivity.py
    
    $duration = (Get-Date) - $startTime
    Write-Host "[SUCCESS] Sensitivity analysis completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] Sensitivity analysis failed: $_" -ForegroundColor Red
    $failedRuns += "sensitivity_analysis"
}

Clear-CudaVram

# ============================================================
# PART 3: Visualization (ALL datasets × ALL models)
# ============================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Blue
Write-Host " PART 3: INDUCTIVE GAP VISUALIZATION" -ForegroundColor Blue
Write-Host "============================================" -ForegroundColor Blue

foreach ($dataset in $Datasets) {
    foreach ($model in $Models) {
        Write-Host ""
        Write-Host ">>> Generating visualization: $model on $dataset" -ForegroundColor Cyan
        
        try {
            python src/eda/vis_inductive.py $dataset $model
            Write-Host "[SUCCESS] Visualization saved to docs/images/inductive_gap_${dataset}_${model}.png" -ForegroundColor Green
        }
        catch {
            Write-Host "[ERROR] Visualization failed: $_" -ForegroundColor Red
        }
        
        Clear-CudaVram
    }
}

Write-Host ""
Write-Host "[DONE] Visualization phase complete!" -ForegroundColor Green

# ============================================================
# FINAL SUMMARY
# ============================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " LEFTOVER PIPELINE COMPLETE" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

if ($failedRuns.Count -gt 0) {
    Write-Host "Failed runs:" -ForegroundColor Red
    foreach ($run in $failedRuns) {
        Write-Host "  - $run" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "Results saved to:"
Write-Host "  - Ablation checkpoints: checkpoints_ablation/{dataset}/{model}_{ablation}/"
Write-Host "  - Sensitivity checkpoints: checkpoints_sensitivity/{dataset}/{model}_{param}/"
Write-Host "  - Sensitivity logs: logs/sensitivity/sensitivity_{timestamp}/"
Write-Host "  - Visualizations: docs/images/inductive_gap_*.png"
Write-Host ""
