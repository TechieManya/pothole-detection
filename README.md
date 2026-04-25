# Pothole Detection & Road Quality Analysis System

AI-powered road quality assessment using **Roboflow Workflow inference** (`inference_sdk`). Detects potholes from **road images**, computes Road Quality Index (RQI), and generates analysis reports.

## Project Structure

```
pothole_ai_project/
├── models/
├── detection/
│   ├── detect_image.py    # Image pothole detection via Roboflow Workflow
├── analysis/
│   └── road_analysis.py   # RQI & road quality metrics
├── visualization/
│   └── graphs.py          # Charts & graphs
├── ui/
│   └── app.py             # Streamlit web UI
├── utils/
│   └── helpers.py
├── outputs/
│   ├── images/
│   ├── graphs/
│   └── reports/
└── requirements.txt
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Roboflow API Key:** set it as an environment variable (recommended):
   - PowerShell:
     ```powershell
     setx ROBOFLOW_API_KEY "YOUR_KEY"
     ```
   Then **restart your terminal** and run the app. (Or enter it in the Streamlit UI; it is not written to disk.)

## Usage

### 1. Run Detection (CLI) on an Image (Roboflow Workflow)
```bash
python detection/detect_image.py path/to/road_image.jpg --workspace manyas-workspace-3p7by --workflow custom-workflow
```
If you didn't set `ROBOFLOW_API_KEY`, add `--api-key YOUR_KEY`.

### 2. Launch Web UI
```bash
streamlit run ui/app.py
```
Then: **Upload Image** → **Run Detection** → **Generate Analysis**

## Road Quality Index (RQI)
- **RQI = 100 - (pothole_density × 10)**
- **Good:** RQI > 80
- **Moderate:** RQI 50–80
- **Poor:** RQI < 50

## Outputs
- Annotated image: `outputs/images/output_image.jpg`
- Graphs: `outputs/graphs/`
- Report: `outputs/reports/road_report.txt`

## GPU
Inference runs on Roboflow serverless; compute is handled server-side.
