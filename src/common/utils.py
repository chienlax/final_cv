"""
Common utilities for training and evaluation.
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    MUST be called at the very start of main() before any other operations.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: If True, prefer CUDA over CPU.
        
    Returns:
        torch.device for computation.
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    
    return device


def setup_logging(log_file: str = None, level: int = logging.INFO) -> None:
    """
    Configure logging for training.
    
    Args:
        log_file: Optional path to log file.
        level: Logging level.
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def create_run_logger(
    run_type: str,
    dataset: str = None,
    model: str = None,
    base_dir: str = "logs",
) -> tuple[logging.Logger, Path]:
    """
    Create a logger for a specific run with timestamped export folder.
    
    Creates: logs/{run_type}/{dataset}_{model}_{timestamp}/
    
    Args:
        run_type: Type of run ("preprocessing", "training", "evaluation").
        dataset: Dataset name.
        model: Model name (for training runs).
        base_dir: Base directory for logs.
        
    Returns:
        Tuple of (logger, run_dir) where run_dir is the timestamped directory.
    """
    from datetime import datetime
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model:
        run_name = f"{dataset}_{model}_{timestamp}"
    elif dataset:
        run_name = f"{dataset}_{timestamp}"
    else:
        run_name = timestamp
    
    run_dir = Path(base_dir) / run_type / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path
    log_file = run_dir / "run.log"
    
    # Configure root logger
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    
    logger = logging.getLogger(__name__)
    
    # Log run info
    logger.info("=" * 70)
    logger.info(f"RUN TYPE: {run_type.upper()}")
    if dataset:
        logger.info(f"DATASET: {dataset}")
    if model:
        logger.info(f"MODEL: {model}")
    logger.info(f"TIMESTAMP: {timestamp}")
    logger.info(f"LOG DIR: {run_dir}")
    logger.info("=" * 70)
    
    return logger, run_dir


def log_system_info(logger: logging.Logger = None) -> dict:
    """
    Log comprehensive system information.
    
    Args:
        logger: Logger to use, or None to use root logger.
        
    Returns:
        Dictionary with system info.
    """
    import platform
    import psutil
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": psutil.virtual_memory().total / 1024**3,
        "ram_available_gb": psutil.virtual_memory().available / 1024**3,
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info["cuda_version"] = torch.version.cuda
    
    logger.info("System Information:")
    for key, value in info.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return info


def log_config(config: object, logger: logging.Logger = None) -> None:
    """
    Log configuration parameters.
    
    Args:
        config: Config object (dataclass or dict).
        logger: Logger to use.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Configuration:")
    
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif hasattr(config, "__dict__"):
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    else:
        config_dict = dict(config)
    
    for key, value in sorted(config_dict.items()):
        logger.info(f"  {key}: {value}")


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = "max"):
        """
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: "max" for metrics like recall, "min" for loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric.
            
        Returns:
            True if should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False


class AverageMeter:
    """Compute and store running averages."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
