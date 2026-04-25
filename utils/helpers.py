"""
Utility helper functions for the Pothole Detection System.
"""

import os
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_output_dirs(output_base: str = "outputs") -> dict:
    """
    Create output directories if they don't exist.
    Returns dict of paths to video, graph, and report directories.
    """
    root = get_project_root()
    base_path = root / output_base
    dirs = {
        "images": base_path / "images",
        "graphs": base_path / "graphs",
        "reports": base_path / "reports"
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {path}")
    
    return {k: str(v) for k, v in dirs.items()}


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent


def get_model_path(models_dir: str = "models", run_dir: str = "runs/detect/train") -> Optional[str]:
    """
    Get the best trained model weights path.
    Checks runs/detect/train first, then models/ directory.
    """
    root = get_project_root()
    # Check runs/detect/train (Ultralytics default output)
    run_path = root / run_dir / "weights" / "best.pt"
    if run_path.exists():
        logger.info(f"Found model at: {run_path}")
        return str(run_path)
    # Check models directory
    model_path = root / models_dir / "best.pt"
    if model_path.exists():
        logger.info(f"Found model at: {model_path}")
        return str(model_path)
    logger.warning("No trained model found. Copy best.pt to project 'models/' folder or set Model Path in the UI.")
    return None


def estimate_video_distance(video_path: str, fps: float, duration_sec: float) -> float:
    """
    Estimate distance covered in video based on assumed vehicle speed.
    Used for pothole density calculation when no GPS data is available.
    
    Args:
        video_path: Path to video file
        fps: Frames per second
        duration_sec: Video duration in seconds
        assumed_speed_kmh: Assumed average speed in km/h
    
    Returns:
        Estimated distance in kilometers
    """
    # Assume 30 km/h average for road survey (conservative estimate)
    assumed_speed_kmh = 30
    hours = duration_sec / 3600
    distance_km = assumed_speed_kmh * hours
    return distance_km
