"""
Data visualization for pothole detection and road quality analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style (compatible across matplotlib versions)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def ensure_graph_dir(output_dir: str = "outputs/graphs") -> Path:
    """Ensure graph output directory exists."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_potholes_per_frame(
    potholes_per_frame: List[int],
    output_path: Optional[str] = None,
) -> str:
    """
    Plot potholes detected per frame over time.
    
    Args:
        potholes_per_frame: List of pothole counts per frame
        output_path: Save path
    
    Returns:
        Path to saved figure
    """
    graph_dir = ensure_graph_dir()
    output_path = output_path or str(graph_dir / "potholes_per_frame.png")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    frames = np.arange(len(potholes_per_frame))
    ax.plot(frames, potholes_per_frame, color='#e74c3c', alpha=0.8, linewidth=1)
    ax.fill_between(frames, potholes_per_frame, alpha=0.3, color='#e74c3c')
    
    ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylabel("Potholes Detected", fontsize=11)
    title = "Potholes Detected (Per Sample)"
    if len(potholes_per_frame) > 1:
        title = "Potholes Detected Per Frame"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path}")
    return output_path


def plot_pothole_density_chart(
    pothole_density: float,
    output_path: Optional[str] = None,
) -> str:
    """
    Bar chart showing pothole density.
    
    Args:
        pothole_density: Potholes per km
        output_path: Save path
    
    Returns:
        Path to saved figure
    """
    graph_dir = ensure_graph_dir()
    output_path = output_path or str(graph_dir / "pothole_density.png")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(["Potholes/km"], [pothole_density], color='#3498db', edgecolor='#2980b9')
    ax.set_ylabel("Potholes per Kilometer", fontsize=11)
    ax.set_title("Pothole Density", fontsize=14, fontweight='bold')
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pothole_density}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path}")
    return output_path


def plot_road_quality_classification(
    rqi: float,
    road_condition: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Gauge-style chart for road quality classification.
    
    Args:
        rqi: Road Quality Index
        road_condition: Classification string
        output_path: Save path
    
    Returns:
        Path to saved figure
    """
    graph_dir = ensure_graph_dir()
    output_path = output_path or str(graph_dir / "road_quality_classification.png")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    conditions = ["Poor", "Moderate", "Good"]
    ranges = [(0, 50), (50, 80), (80, 100)]
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    for i, (cond, (low, high)) in enumerate(zip(conditions, ranges)):
        ax.barh(0, high - low, left=low, height=0.5, color=colors[i], label=cond, edgecolor='white')
    
    ax.axvline(x=rqi, color='black', linestyle='--', linewidth=2, label=f'RQI: {rqi}')
    ax.set_xlim(0, 100)
    ax.set_xlabel("Road Quality Index", fontsize=11)
    ax.set_title(f"Road Condition: {road_condition}", fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path}")
    return output_path


def plot_speed_vs_density(
    pothole_density: float,
    estimated_speed: float,
    expected_speed: float = 60,
    output_path: Optional[str] = None,
) -> str:
    """
    Graph showing estimated speed vs pothole density.
    
    Args:
        pothole_density: Potholes per km
        estimated_speed: Calculated average speed
        expected_speed: Baseline expected speed
        output_path: Save path
    
    Returns:
        Path to saved figure
    """
    graph_dir = ensure_graph_dir()
    output_path = output_path or str(graph_dir / "speed_vs_density.png")
    
    # Simulate curve: speed decreases with density
    densities = np.linspace(0, max(20, pothole_density * 1.5), 50)
    speeds = np.maximum(0, expected_speed - densities * 2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(densities, speeds, 'b-', linewidth=2, label='Estimated Speed')
    ax.axhline(y=expected_speed, color='green', linestyle=':', alpha=0.7, label=f'Expected Speed ({expected_speed} km/h)')
    ax.scatter([pothole_density], [estimated_speed], color='red', s=100, zorder=5,
               label=f'Current: {estimated_speed} km/h @ {pothole_density} potholes/km')
    
    ax.set_xlabel("Pothole Density (potholes/km)", fontsize=11)
    ax.set_ylabel("Estimated Speed (km/h)", fontsize=11)
    ax.set_title("Estimated Vehicle Speed vs Pothole Density", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path}")
    return output_path


def generate_all_graphs(
    potholes_per_frame: List[int],
    analysis_results: Dict,
    output_dir: str = "outputs/graphs",
) -> Dict[str, str]:
    """
    Generate analysis graphs (excluding legacy potholes-per-frame plot).
    """
    paths = {}

    paths["pothole_density"] = plot_pothole_density_chart(
        analysis_results["pothole_density"],
        str(Path(output_dir) / "pothole_density.png"),
    )
    
    paths["road_quality"] = plot_road_quality_classification(
        analysis_results["rqi"],
        analysis_results["road_condition"],
        str(Path(output_dir) / "road_quality_classification.png"),
    )
    
    paths["speed_vs_density"] = plot_speed_vs_density(
        analysis_results["pothole_density"],
        analysis_results["estimated_avg_speed_kmh"],
        analysis_results["expected_speed_kmh"],
        str(Path(output_dir) / "speed_vs_density.png"),
    )
    
    logger.info(f"Generated {len(paths)} graphs")
    return paths
