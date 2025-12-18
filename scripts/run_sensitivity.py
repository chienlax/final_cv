"""
Sensitivity Analysis Automator.

Loops through critical hyperparameters for LATTICE/MICRO/DiffMM and logs Recall@20.
Results are saved to logs/sensitivity/ directory.

Usage:
    python scripts/run_sensitivity.py --dataset beauty --epochs 20
    python scripts/run_sensitivity.py --dataset clothing --epochs 20 --models lattice micro
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Define the search space
SENSITIVITY_GRIDS = {
    "lattice": {
        "lattice_k": [10, 20, 40, 60, 80],
    },
    "micro": {
        "micro_alpha": [0.01, 0.05, 0.1, 0.2, 0.5],
    },
    "diffmm": {
        "diffmm_steps": [20, 50, 100, 150, 200],
    },
}


def run_experiment(dataset: str, model: str, param_name: str, param_value, epochs: int) -> dict:
    """Run a single training experiment and capture results."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {model.upper()} on {dataset} with {param_name}={param_value}")
    print(f"{'='*60}")
    
    # Construct command
    cmd = [
        sys.executable, "src/main.py",
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs),
        f"--{param_name.replace('_', '-')}", str(param_value),
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run and capture output
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=3600,  # 1 hour timeout
            cwd=Path(__file__).parent.parent,
        )
        
        # Parse results from output
        output = result.stdout + result.stderr
        
        # Try to find Recall@20 in output
        recall_20 = None
        for line in output.split('\n'):
            if "Recall@20" in line and "Val" in line:
                # Extract the value
                try:
                    parts = line.split("Recall@20:")
                    if len(parts) > 1:
                        value_str = parts[1].split()[0].strip()
                        recall_20 = float(value_str)
                except:
                    pass
        
        return {
            "param_name": param_name,
            "param_value": param_value,
            "recall_20": recall_20,
            "success": result.returncode == 0,
            "returncode": result.returncode,
        }
        
    except subprocess.TimeoutExpired:
        print(f"⚠️  Timeout after 1 hour")
        return {
            "param_name": param_name,
            "param_value": param_value,
            "recall_20": None,
            "success": False,
            "error": "timeout",
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "param_name": param_name,
            "param_value": param_value,
            "recall_20": None,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run sensitivity analysis")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per run (short for trends)")
    parser.add_argument("--models", nargs="+", default=["lattice", "micro", "diffmm"],
                       choices=["lattice", "micro", "diffmm"], help="Models to test")
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("logs/sensitivity") / f"{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 Sensitivity Analysis")
    print(f"   Dataset: {args.dataset}")
    print(f"   Models: {args.models}")
    print(f"   Epochs per run: {args.epochs}")
    print(f"   Output: {output_dir}")
    
    all_results = {}
    
    for model in args.models:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model.upper()}")
        print(f"{'#'*60}")
        
        if model not in SENSITIVITY_GRIDS:
            print(f"⚠️  No sensitivity grid defined for {model}")
            continue
        
        model_results = {}
        
        for param_name, param_values in SENSITIVITY_GRIDS[model].items():
            param_results = []
            
            for param_value in param_values:
                result = run_experiment(
                    args.dataset, model, param_name, param_value, args.epochs
                )
                param_results.append(result)
                
                # Print progress
                if result["recall_20"]:
                    print(f"   ✓ {param_name}={param_value}: Recall@20 = {result['recall_20']:.4f}")
                else:
                    print(f"   ✗ {param_name}={param_value}: Failed")
            
            model_results[param_name] = param_results
        
        all_results[model] = model_results
    
    # Save results
    results_file = output_dir / "sensitivity_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📊 Results saved to {results_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📈 SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for model, model_results in all_results.items():
        print(f"\n{model.upper()}:")
        for param_name, param_results in model_results.items():
            recalls = [(r["param_value"], r["recall_20"]) for r in param_results if r["recall_20"]]
            if recalls:
                best = max(recalls, key=lambda x: x[1])
                print(f"  {param_name}:")
                for val, recall in recalls:
                    marker = "★" if val == best[0] else " "
                    print(f"    {marker} {val}: {recall:.4f}")


if __name__ == "__main__":
    main()
