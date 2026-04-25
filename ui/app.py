"""
Streamlit UI for Pothole Detection and Road Quality Analysis System.

This version uses Roboflow Workflows (inference_sdk) to detect potholes on images.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="Pothole Detection & Road Quality Analysis",
    page_icon="🛣️",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .metric-card { 
        padding: 20px; 
        border-radius: 10px; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🛣️ Pothole Detection & Road Quality Analysis")
    st.markdown("*AI-powered road quality assessment using Roboflow Workflow inference*")

    # Initialize session state
    if "detection_done" not in st.session_state:
        st.session_state.detection_done = False
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "started" not in st.session_state:
        st.session_state.started = False

    def _get_api_key_default() -> str:
        # Prefer Streamlit secrets if present, then environment variable.
        try:
            if "ROBOFLOW_API_KEY" in st.secrets:
                return str(st.secrets["ROBOFLOW_API_KEY"])
        except Exception:
            pass
        return os.getenv("ROBOFLOW_API_KEY", "")

    # Sidebar - Workflow and options
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input(
            "Roboflow API Key",
            value=_get_api_key_default(),
            type="password",
            help="Recommended: set env var ROBOFLOW_API_KEY once (PowerShell: `setx ROBOFLOW_API_KEY \"YOUR_KEY\"`).",
        )
        workspace_name = st.text_input("Workspace Name", value="manyas-workspace-3p7by")
        workflow_id = st.text_input("Workflow ID", value="custom-workflow")
        use_cache = st.checkbox("Use Cache", value=True)

        st.divider()
        distance_km = st.number_input(
            "Distance (km) represented by this image",
            min_value=0.01,
            value=1.0,
            step=0.1,
            help="Used to compute potholes/km and RQI. Set based on your sampling method.",
        )
        st.divider()

        st.header("📁 Steps")
        st.markdown("1. Get started on the home page")
        st.markdown("2. Upload a road image")
        st.markdown("3. Run detection")
        st.markdown("4. View analysis & graphs")
        st.divider()
        view = st.radio(
            "View",
            options=["Overview", "Analysis & Graphs"],
            index=0,
            help="Switch between the main view and the detailed analysis.",
        )

    # Landing / start screen
    if not st.session_state.started:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Welcome to your road analytics cockpit")
            st.markdown(
                "- **Detect potholes** on road images using your Roboflow model.\n"
                "- **Quantify road quality** with an automatic Road Quality Index (RQI).\n"
                "- **Estimate speed impact** and generate ready-to-share reports."
            )
            st.markdown("#### How it works")
            st.markdown(
                "1. Configure your **Roboflow API key, workspace and workflow** in the sidebar.\n"
                "2. Click **Get started**.\n"
                "3. Upload a road image and run detection.\n"
                "4. Generate analysis to view RQI, graphs, and the text report."
            )
            if st.button("🚀 Get started", type="primary"):
                st.session_state.started = True
                # Compatible rerun for newer/older Streamlit versions
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()  # older versions
                    except Exception:
                        pass
        with col_right:
            st.markdown("#### Project snapshot")
            st.markdown("- **Input**: Single road image\n- **Engine**: Roboflow Workflow API\n- **Outputs**:")
            st.markdown("  - Annotated pothole image\n  - RQI & condition label\n  - Density & speed graphs\n  - Text report")
        # Do not show upload / results until started
        st.divider()
        st.caption("Configure your Roboflow settings on the left, then click **Get started**.")
        return

    # Main content - Upload (after starting)
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a road image (JPG, PNG, etc.)",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        st.success(f"Uploaded: {uploaded_file.name}")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎯 Run Detection", use_container_width=True):
                with st.spinner("Running pothole detection on the image..."):
                    try:
                        from detection.detect_image import detect_potholes_image, WorkflowConfig

                        cfg = WorkflowConfig(
                            api_key=api_key or None,
                            workspace_name=workspace_name.strip(),
                            workflow_id=workflow_id.strip(),
                            use_cache=use_cache,
                        )
                        det = detect_potholes_image(image_path, cfg, save_raw_json=True)

                        st.session_state.annotated_image_path = det["annotated_image_path"]
                        st.session_state.prediction_count = det["prediction_count"]
                        st.session_state.predictions = det["predictions"]
                        st.session_state.distance_km = float(distance_km)
                        st.session_state.detection_done = True
                        st.success("Detection complete!")
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Detection failed: {e}")

        with col2:
            if st.session_state.detection_done and st.button("📊 Generate Analysis", use_container_width=True):
                with st.spinner("Generating analysis and graphs..."):
                    try:
                        from analysis.road_analysis import run_road_analysis_from_distance, generate_report
                        from visualization.graphs import generate_all_graphs

                        total_potholes = int(st.session_state.prediction_count)
                        analysis = run_road_analysis_from_distance(
                            total_potholes=total_potholes,
                            distance_km=float(st.session_state.distance_km),
                        )
                        st.session_state.analysis = analysis

                        # Generate report
                        report_path = str(PROJECT_ROOT / "outputs" / "reports" / "road_report.txt")
                        generate_report(analysis, report_path)

                        # Generate graphs
                        graph_dir = str(PROJECT_ROOT / "outputs" / "graphs")
                        graph_paths = generate_all_graphs(
                            [total_potholes],
                            analysis,
                            graph_dir,
                        )
                        st.session_state.graph_paths = graph_paths
                        st.session_state.analysis_done = True
                        st.success("Analysis complete!")
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

        # Display results (with simple view switching)
        if st.session_state.detection_done:
            if view == "Overview":
                st.divider()
                st.header("🖼️ Annotated Output Image")
                annotated_path = st.session_state.get("annotated_image_path")
                if annotated_path and Path(annotated_path).exists():
                    st.image(annotated_path, use_container_width=True)

                st.subheader("📈 Pothole Statistics")
                total_potholes = int(st.session_state.prediction_count)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Total Potholes Detected", total_potholes)
                with m2:
                    st.metric("Distance (km)", f"{st.session_state.distance_km}")
                with m3:
                    st.metric("Workflow", f"{workspace_name}/{workflow_id}")

            elif view == "Analysis & Graphs":
                if not st.session_state.analysis_done:
                    st.info("Click **Generate Analysis** in the sidebar to see detailed graphs and RQI.")
                else:
                    st.divider()
                    st.header("📊 Road Quality Analysis")

                    analysis = st.session_state.analysis
                    rqi = analysis["rqi"]
                    condition = analysis["road_condition"]

                    # RQI with color card
                    rqi_color = "#27ae60" if rqi > 80 else "#f39c12" if rqi >= 50 else "#e74c3c"
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, {rqi_color} 0%, #2c3e50 100%);">
                        <p style="margin:0;">Road Quality Index: <span style="font-size:32px;">{rqi}</span></p>
                        <p style="margin:0;">Condition: <strong>{condition}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Pothole Density", f"{analysis['pothole_density']} /km")
                    with c2:
                        st.metric("Estimated Avg Speed", f"{analysis['estimated_avg_speed_kmh']} km/h")
                    with c3:
                        st.metric("Distance Analyzed", f"{analysis['distance_km']} km")

                    st.subheader("📉 Graphs")
                    graph_paths = st.session_state.get("graph_paths", {})
                    for name, path in graph_paths.items():
                        if Path(path).exists():
                            label = name.replace("_", " ").title()
                            st.markdown(f"**{label}**")
                            st.image(path, use_container_width=True)

                    report_path = PROJECT_ROOT / "outputs" / "reports" / "road_report.txt"
                    if report_path.exists():
                        with open(report_path) as f:
                            report_text = f.read()
                        st.subheader("📄 Text Report")
                        st.text(report_text)

    else:
        st.info("👆 Upload a road image to get started.")

    # Footer
    st.divider()
    st.markdown("---")
    st.caption("Pothole Detection System | Roboflow Workflow + Streamlit")


if __name__ == "__main__":
    main()
