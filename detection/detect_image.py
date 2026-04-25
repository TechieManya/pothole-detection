"""
Pothole detection on images using a Roboflow Workflow via inference_sdk.

Important:
- Do NOT hardcode API keys in code. Provide via env var ROBOFLOW_API_KEY or Streamlit UI input.
"""

import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from inference_sdk import InferenceHTTPClient

from utils.helpers import ensure_output_dirs, get_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    api_url: str = "https://serverless.roboflow.com"
    api_key: Optional[str] = None
    workspace_name: str = ""
    workflow_id: str = ""
    use_cache: bool = True


def _resolve_api_key(config: WorkflowConfig) -> str:
    api_key = config.api_key or os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Roboflow API key missing. Set ROBOFLOW_API_KEY env var or enter it in the UI."
        )
    return api_key


def run_workflow_on_image(
    image_path: str,
    config: WorkflowConfig,
) -> Dict[str, Any]:
    """
    Run a Roboflow Workflow on an image and return the raw JSON result.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not config.workspace_name or not config.workflow_id:
        raise ValueError("workspace_name and workflow_id are required")

    client = InferenceHTTPClient(
        api_url=config.api_url,
        api_key=_resolve_api_key(config),
    )

    result = client.run_workflow(
        workspace_name=config.workspace_name,
        workflow_id=config.workflow_id,
        images={"image": image_path},
        use_cache=config.use_cache,
    )
    return result


def _extract_predictions(raw_result: Any) -> List[Dict[str, Any]]:
    """
    Attempt to extract a flat list of predictions from common Roboflow workflow outputs.
    Returns a list of dicts containing at least x/y/width/height/confidence/class/name when available.
    """
    if raw_result is None:
        return []

    # Some workflows return a top-level list of step outputs.
    if isinstance(raw_result, list):
        preds: List[Dict[str, Any]] = []
        for item in raw_result:
            preds.extend(_extract_predictions(item))
        return preds

    if not isinstance(raw_result, dict):
        return []

    # 1) Common: result["outputs"][...]["predictions"] (list)
    outputs = raw_result.get("outputs")
    if isinstance(outputs, list):
        preds: List[Dict[str, Any]] = []
        for out in outputs:
            if isinstance(out, dict):
                preds.extend(_extract_predictions(out))
        if preds:
            return preds

    # 2) Common: result["predictions"] is already a list
    if isinstance(raw_result.get("predictions"), list):
        return raw_result["predictions"]

    # 3) Observed in your saved response:
    #    {"predictions": {"image": {...}, "predictions": [ ... ]}}
    p = raw_result.get("predictions")
    if isinstance(p, dict) and isinstance(p.get("predictions"), list):
        return p["predictions"]

    # 4) Fallback: search nested dicts for a "predictions" list
    for v in raw_result.values():
        if isinstance(v, dict):
            found = _extract_predictions(v)
            if found:
                return found

    return []


def annotate_image_with_predictions(
    image_path: str,
    predictions: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    box_color_bgr: Tuple[int, int, int] = (0, 165, 255),
) -> str:
    """
    Draw bounding boxes on an image and save it.
    Assumes prediction boxes are center-based with (x, y, width, height) in pixels.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    for p in predictions:
        x = p.get("x")
        y = p.get("y")
        w = p.get("width")
        h = p.get("height")
        conf = p.get("confidence", None)
        cls = p.get("class", p.get("class_name", p.get("name", None)))
        class_id = p.get("class_id", None)
        if cls is None and class_id is not None:
            cls = str(class_id)
        if cls in ("0", 0):
            cls = "pothole"
        if cls is None:
            cls = "pothole"

        if any(v is None for v in [x, y, w, h]):
            continue

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), box_color_bgr, 2)

        label = str(cls)
        if conf is not None:
            label = f"{label} {float(conf):.2f}"

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - lh - 10)), (x1 + lw, y1), box_color_bgr, -1)
        cv2.putText(
            img,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    ensure_output_dirs()
    out_path = Path(output_path) if output_path else (get_project_root() / "outputs" / "images" / "output_image.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return str(out_path)


def detect_potholes_image(
    image_path: str,
    config: WorkflowConfig,
    save_raw_json: bool = True,
) -> Dict[str, Any]:
    """
    Runs workflow inference, extracts predictions, saves annotated image + optional raw JSON.
    Returns a dict with:
      - annotated_image_path
      - prediction_count
      - predictions
      - raw_result_path (optional)
    """
    raw = run_workflow_on_image(image_path, config)
    preds = _extract_predictions(raw)

    annotated_path = annotate_image_with_predictions(image_path, preds)

    result: Dict[str, Any] = {
        "annotated_image_path": annotated_path,
        "prediction_count": len(preds),
        "predictions": preds,
    }

    if save_raw_json:
        ensure_output_dirs()
        raw_path = get_project_root() / "outputs" / "reports" / "raw_inference_result.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        result["raw_result_path"] = str(raw_path)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pothole detection on images via Roboflow Workflow")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace name")
    parser.add_argument("--workflow", required=True, help="Roboflow workflow id")
    parser.add_argument("--api-key", default="", help="Roboflow API key (or set ROBOFLOW_API_KEY)")
    args = parser.parse_args()

    cfg = WorkflowConfig(
        api_key=args.api_key or None,
        workspace_name=args.workspace,
        workflow_id=args.workflow,
    )
    out = detect_potholes_image(args.image, cfg)
    print(out["annotated_image_path"])
    print("Detections:", out["prediction_count"])

