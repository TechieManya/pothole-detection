"""
Road Quality Analysis - Calculates RQI, pothole density, and speed impact.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EXPECTED_SPEED_KMH = 60
SPEED_REDUCTION_FACTOR = 2
ASSUMED_SPEED_FOR_DISTANCE = 30


def calculate_pothole_density(total_potholes: int, distance_km: float) -> float:
    """Calculate potholes per kilometer."""
    if distance_km <= 0:
        logger.warning("Distance is zero or negative, cannot compute density")
        return 0.0
    return round(total_potholes / distance_km, 2)


def calculate_rqi(pothole_density: float) -> float:
    """RQI = 100 - (pothole_density x 10)"""
    rqi = 100 - (pothole_density * 10)
    return max(0, min(100, round(rqi, 2)))


def classify_road_condition(rqi: float) -> str:
    """RQI > 80 Good, 50-80 Moderate, < 50 Poor"""
    if rqi > 80:
        return "Good"
    elif rqi >= 50:
        return "Moderate"
    else:
        return "Poor"


def estimate_average_speed(
    pothole_density: float,
    expected_speed: float = EXPECTED_SPEED_KMH,
    speed_reduction_factor: float = SPEED_REDUCTION_FACTOR,
) -> float:
    """Average speed = expected_speed - (pothole_density x speed_reduction_factor)"""
    speed_reduction = pothole_density * speed_reduction_factor
    return max(0, round(expected_speed - speed_reduction, 2))


def estimate_distance_from_video(
    duration_seconds: float,
    assumed_speed_kmh: float = ASSUMED_SPEED_FOR_DISTANCE,
) -> float:
    """Estimate distance in km from video duration."""
    hours = duration_seconds / 3600
    return round(assumed_speed_kmh * hours, 4)


def run_road_analysis(
    total_potholes: int,
    video_duration_seconds: float,
    distance_km: Optional[float] = None,
) -> Dict:
    """Run complete road quality analysis."""
    if distance_km is None or distance_km <= 0:
        distance_km = estimate_distance_from_video(video_duration_seconds)
        logger.info(f"Estimated distance: {distance_km} km")

    pothole_density = calculate_pothole_density(total_potholes, distance_km)
    rqi = calculate_rqi(pothole_density)
    road_condition = classify_road_condition(rqi)
    avg_speed = estimate_average_speed(pothole_density)

    return {
        "total_potholes": total_potholes,
        "distance_km": distance_km,
        "pothole_density": pothole_density,
        "rqi": rqi,
        "road_condition": road_condition,
        "estimated_avg_speed_kmh": avg_speed,
        "expected_speed_kmh": EXPECTED_SPEED_KMH,
    }


def run_road_analysis_from_distance(
    total_potholes: int,
    distance_km: float,
) -> Dict:
    """
    Road analysis when distance is known directly (e.g., image-based sampling).
    """
    pothole_density = calculate_pothole_density(total_potholes, distance_km)
    rqi = calculate_rqi(pothole_density)
    road_condition = classify_road_condition(rqi)
    avg_speed = estimate_average_speed(pothole_density)

    return {
        "total_potholes": total_potholes,
        "distance_km": distance_km,
        "pothole_density": pothole_density,
        "rqi": rqi,
        "road_condition": road_condition,
        "estimated_avg_speed_kmh": avg_speed,
        "expected_speed_kmh": EXPECTED_SPEED_KMH,
    }


def generate_report(
    analysis_results: Dict,
    output_path: str = "outputs/reports/road_report.txt",
) -> str:
    """Generate text report from analysis results."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "=" * 50,
        "ROAD QUALITY ANALYSIS REPORT",
        "=" * 50,
        "",
        f"Total Potholes Detected: {analysis_results['total_potholes']}",
        f"Distance Analyzed: {analysis_results['distance_km']} km",
        f"Pothole Density: {analysis_results['pothole_density']} potholes/km",
        "",
        f"Road Quality Index (RQI): {analysis_results['rqi']}",
        f"Road Condition: {analysis_results['road_condition']}",
        "",
        f"Expected Vehicle Speed: {analysis_results['expected_speed_kmh']} km/h",
        f"Estimated Average Speed: {analysis_results['estimated_avg_speed_kmh']} km/h",
        "",
        "=" * 50,
    ]
    with open(output_path, 'w') as f:
        f.write("\n".join(report_lines))
    logger.info(f"Report saved to: {output_path}")
    return output_path
