# Full Pipeline Script for Multimodal Recommendation
# Runs preprocessing + training for all datasets and models
# Cleans CUDA VRAM between runs to prevent OOM
# Auto-detects existing preprocessed data and skips if complete

param(
    [switch]$SkipPreprocessing,
    [switch]$PreprocessingOnly,
    [switch]$ForcePreprocessing,  # Force preprocessing even if data exists
    [string[]]$Datasets = @("electronics", "beauty", "clothing"),
    [string[]]$Models = @("lattice", "micro", "diffmm")
)

$ErrorActionPreference = "Stop"

# Configuration
$SEED = 42
$SEED_USERS = 10000

# ============================================================
# Helper Functions
# ============================================================

function Test-ProcessedData {
    <#
    .SYNOPSIS
    Check if preprocessed data exists and is complete for a dataset.
    
    .DESCRIPTION
    Verifies that all required files exist:
    - train.txt, val.txt, test_warm.txt, test_cold.txt
    - feat_visual.npy, feat_text.npy
    - maps.json
    #>
    param([string]$Dataset)
    
    $requiredFiles = @(
        "data/processed/$Dataset/train.txt",
        "data/processed/$Dataset/val.txt",
        "data/processed/$Dataset/test_warm.txt",
        "data/processed/$Dataset/test_cold.txt",
        "data/processed/$Dataset/feat_visual.npy",
        "data/processed/$Dataset/feat_text.npy",
        "data/processed/$Dataset/maps.json"
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            return $false
        }
    }
    return $true
}

function Clear-CudaVram {
    <#
    .SYNOPSIS
    Clear CUDA VRAM to prevent OOM between runs.
    #>
    Write-Host "[CLEANUP] Clearing CUDA VRAM..." -ForegroundColor Magenta
    python -c "import torch; torch.cuda.empty_cache(); print(f'VRAM freed. Available: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB')"
    Start-Sleep -Seconds 2
}

function Show-GpuMemory {
    <#
    .SYNOPSIS
    Display current GPU memory usage.
    #>
    python -c "import torch; used = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0])/1024**3; total = torch.cuda.mem_get_info()[1]/1024**3; print(f'GPU Memory: {used:.1f}/{total:.1f} GB')"
}

# ============================================================
# Pipeline Start
# ============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Multimodal Recommendation Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Datasets: $($Datasets -join ', ')" -ForegroundColor Yellow
Write-Host "Models: $($Models -join ', ')" -ForegroundColor Yellow
Write-Host "Seed: $SEED" -ForegroundColor Yellow
Write-Host ""

# ============================================================
# PHASE 1: Preprocessing
# ============================================================
if (-not $SkipPreprocessing) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host " PHASE 1: PREPROCESSING" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    
    $allDataExists = $true
    
    foreach ($dataset in $Datasets) {
        Write-Host ""
        
        # Check if data already exists
        if ((Test-ProcessedData $dataset) -and (-not $ForcePreprocessing)) {
            Write-Host "[SKIP] $dataset - Preprocessed data already exists" -ForegroundColor Yellow
            continue
        }
        
        $allDataExists = $false
        Write-Host ">>> Preprocessing: $dataset" -ForegroundColor Cyan
        Write-Host "-------------------------------------------"
        
        $startTime = Get-Date
        
        try {
            python src/preprocessing/run_preprocessing.py --dataset $dataset --seed-users $SEED_USERS --seed $SEED
            
            $duration = (Get-Date) - $startTime
            Write-Host "[SUCCESS] $dataset completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
        }
        catch {
            Write-Host "[ERROR] $dataset preprocessing failed: $_" -ForegroundColor Red
            exit 1
        }
        
        # Clear VRAM after preprocessing
        Clear-CudaVram
    }
    
    if ($allDataExists) {
        Write-Host ""
        Write-Host "[INFO] All datasets already preprocessed. Use -ForcePreprocessing to reprocess." -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "[DONE] Preprocessing phase complete!" -ForegroundColor Green
}

if ($PreprocessingOnly) {
    Write-Host "Preprocessing only mode - exiting."
    exit 0
}

# # ============================================================
# # PHASE 2: Training
# # ============================================================
# Write-Host ""
# Write-Host "============================================" -ForegroundColor Green
# Write-Host " PHASE 2: MODEL TRAINING" -ForegroundColor Green
# Write-Host "============================================" -ForegroundColor Green

# $totalRuns = $Datasets.Count * $Models.Count
# $currentRun = 0
# $failedRuns = @()

# foreach ($dataset in $Datasets) {
#     # Verify data exists before training
#     if (-not (Test-ProcessedData $dataset)) {
#         Write-Host "[ERROR] No preprocessed data for $dataset. Run preprocessing first." -ForegroundColor Red
#         continue
#     }
    
#     foreach ($model in $Models) {
#         $currentRun++
        
#         Write-Host ""
#         Write-Host ">>> [$currentRun/$totalRuns] Training: $model on $dataset" -ForegroundColor Cyan
#         Write-Host "-------------------------------------------"
        
#         Show-GpuMemory
        
#         $startTime = Get-Date
        
#         try {
#             python src/main.py --model $model --dataset $dataset --seed $SEED
            
#             $duration = (Get-Date) - $startTime
#             Write-Host "[SUCCESS] $model/$dataset completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
#         }
#         catch {
#             Write-Host "[ERROR] $model/$dataset training failed: $_" -ForegroundColor Red
#             $failedRuns += "$model/$dataset"
#             # Continue with next run instead of stopping
#         }
        
#         # Clear VRAM after each training run
#         Clear-CudaVram
#     }
# }

# # ============================================================
# # Summary
# # ============================================================
# Write-Host ""
# Write-Host "============================================" -ForegroundColor Green
# Write-Host " PHASE 2 COMPLETE - TRAINING" -ForegroundColor Green
# Write-Host "============================================" -ForegroundColor Green
# Write-Host ""

# if ($failedRuns.Count -gt 0) {
#     Write-Host "Failed runs:" -ForegroundColor Red
#     foreach ($run in $failedRuns) {
#         Write-Host "  - $run" -ForegroundColor Red
#     }
#     Write-Host ""
# }

# # ============================================================
# # PHASE 3: Ablation Analysis
# # ============================================================
# # Tests the contribution of visual vs text modalities
# # Runs ALL models × ALL datasets × 2 ablation modes

# Write-Host ""
# Write-Host "============================================" -ForegroundColor Magenta
# Write-Host " PHASE 3: ABLATION ANALYSIS" -ForegroundColor Magenta
# Write-Host "============================================" -ForegroundColor Magenta

# $ablationModes = @("no_visual", "no_text")
# $totalAblationRuns = $Datasets.Count * $Models.Count * $ablationModes.Count
# $currentAblationRun = 0

# foreach ($dataset in $Datasets) {
#     foreach ($model in $Models) {
#         foreach ($ablation in $ablationModes) {
#             $currentAblationRun++
            
#             Write-Host ""
#             Write-Host ">>> [$currentAblationRun/$totalAblationRuns] Ablation: $model on $dataset ($ablation)" -ForegroundColor Cyan
#             Write-Host "    Using base config, only modifying ablation mode"
#             Write-Host "-------------------------------------------"
            
#             Show-GpuMemory
#             $startTime = Get-Date
            
#             try {
#                 # Use base config params, only modify ablation mode
#                 # Output auto-routes to checkpoints_ablation/
#                 python src/main.py --model $model --dataset $dataset --ablation $ablation --seed $SEED
                
#                 $duration = (Get-Date) - $startTime
#                 Write-Host "[SUCCESS] $model/$ablation on $dataset completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
#             }
#             catch {
#                 Write-Host "[ERROR] Ablation failed: $_" -ForegroundColor Red
#                 $failedRuns += "ablation_${model}_${ablation}/${dataset}"
#             }
            
#             Clear-CudaVram
#         }
#     }
# }

# Write-Host ""
# Write-Host "[DONE] Ablation analysis complete! ($totalAblationRuns runs)" -ForegroundColor Green
# Write-Host "Results saved to: checkpoints_ablation/{dataset}/{model}_{ablation}/"

# ============================================================
# PHASE 4: Visualization (Inductive Gap Analysis)
# ============================================================
# Generates t-SNE visualizations of warm vs cold item embeddings

Write-Host ""
Write-Host "============================================" -ForegroundColor Blue
Write-Host " PHASE 4: INDUCTIVE GAP VISUALIZATION" -ForegroundColor Blue
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
Write-Host " PIPELINE COMPLETE" -ForegroundColor Green
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
Write-Host "  - Preprocessed data: data/processed/{dataset}/"
Write-Host "  - Main checkpoints: checkpoints/{dataset}/{model}/"
Write-Host "  - Ablation checkpoints: checkpoints_ablation/{dataset}/{model}_{ablation}/"
Write-Host "  - Sensitivity checkpoints: checkpoints_sensitivity/{dataset}/{model}_{param}/"
Write-Host "  - Logs: logs/training/{dataset}_{model}_{timestamp}/"
Write-Host "  - Training report: docs/03_training-results.md"
Write-Host "  - Visualizations: docs/images/inductive_gap_*.png"
Write-Host ""


