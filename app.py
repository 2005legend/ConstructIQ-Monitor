import logging
import streamlit as st
from PIL import Image
import numpy as np
import json
import os
from datetime import datetime
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SiteVision")

from detection.detector import detect_objects, identify_violations, calculate_ppe_coverage
from sfm.reconstructor import reconstruct_3d
from fusion.projector import project_to_3d, overlay_violations_3d
from progress.tracker import compare_point_clouds
from session_manager import SessionManager
from app_helpers import _parse_ply_bytes, _build_reconstruction_figure

# --- Streamlit Config ---
st.set_page_config(page_title="SiteVision: Unified Pipeline", layout="wide")
st.title("🚧 SiteVision: Unified Session Pipeline")

# Initialize session state for cached views
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

tab1, tab2 = st.tabs(["🏗️ New Site Visit (Unified Analysis)", "📈 3D Progress Tracking"])

with tab1:
    st.header("New Site Visit Session")
    st.markdown("Upload overlapping photos. The system will automatically build the 3D structure and map safety violations into it using exactly the same camera poses.")
    
    files = st.file_uploader("Upload Overlapping Drone/Walkthrough Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if files and len(files) >= 2:
        if st.button("🚀 Process Unified Session"):
            with st.spinner("Creating session..."):
                file_bytes = [(f.name, f.read()) for f in files]
                session = SessionManager.create_session(file_bytes)
                st.session_state.current_session_id = session.session_id
                
            progress_bar = st.progress(0, text="Step 1: Reconstructing 3D Space (COLMAP)...")
            out_dir = os.path.join(SessionManager.get_session_dir(session.session_id), "colmap")
            
            # 1. Run COLMAP
            result = reconstruct_3d(session.image_paths, out_dir)
            session.reconstruction = result
            
            progress_bar.progress(50, text="Step 2: Running AI Object Detection (YOLO)...")
            
            # 2. Run Object Detection on the exact same session photos
            all_violations = []
            poses_dict = {os.path.basename(p.image_id): p for p in result.camera_poses} if result.success else {}
            
            for img_path in session.image_paths:
                img_name = os.path.basename(img_path)
                img_array = np.array(Image.open(img_path).convert("RGB"))
                
                detections, _ = detect_objects(img_array, conf_threshold=0.3)
                calculate_ppe_coverage(detections)
                violations = identify_violations(detections)
                
                # 3. Fuse: Project violations into 3D immediately
                for v in violations:
                    v.image_id = img_name
                    if result.success and img_name in poses_dict:
                        pose = poses_dict[img_name]
                        v.location_3d = project_to_3d(v.location_2d, pose, depth=5.0)
                all_violations.extend(violations)
                
            session.violations = all_violations
            SessionManager.save_session(session)
            
            progress_bar.progress(100, text="Done!")
            st.success(f"✅ Session Processed! {len(result.camera_poses)} cameras recovered, {len(all_violations)} violations found.")
            
    # Render active session
    if st.session_state.current_session_id:
        session = SessionManager.load_session(st.session_state.current_session_id)
        if session and session.reconstruction and session.reconstruction.success:
            st.subheader(f"Session Results: {session.session_id}")

            # Initialize to empty arrays — used even when PLY is absent (cameras-only view)
            xyz = np.empty((0, 3))
            rgb = np.empty((0, 3), dtype=np.uint8)

            if session.reconstruction.point_cloud_path and os.path.exists(session.reconstruction.point_cloud_path):
                with open(session.reconstruction.point_cloud_path, "rb") as f:
                    xyz, rgb = _parse_ply_bytes(f.read())
            else:
                st.info("⚠️ Point cloud file not found on disk. Showing camera positions only.")

            import plotly.graph_objects as go
            fig = _build_reconstruction_figure(xyz, rgb, session.reconstruction.camera_poses)

            # Add violations to 3D plot
            markers = overlay_violations_3d(None, session.violations)
            if markers["3d_markers"]:
                fig.add_trace(go.Scatter3d(
                    x=[m["position"][0] for m in markers["3d_markers"]],
                    y=[m["position"][1] for m in markers["3d_markers"]],
                    z=[m["position"][2] for m in markers["3d_markers"]],
                    mode="markers+text",
                    marker=dict(size=12, color=[m["color"] for m in markers["3d_markers"]], symbol="circle", line=dict(color="white", width=2)),
                    text=[m["type"].replace("_", " ").title() for m in markers["3d_markers"]],
                    name="Safety Violations"
                ))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📂 Upload images above and click **Process Unified Session** to start a new site visit.")

with tab2:
    st.header("3D Progress Tracking (ICP)")
    st.markdown("Select two past sessions. The system will align their 3D point clouds and highlight new structural progress.")
    
    sessions = SessionManager.list_sessions()
    if len(sessions) >= 2:
        col1, col2 = st.columns(2)
        s1_id = col1.selectbox("Past Session (Baseline)", sessions, index=len(sessions)-1)
        s2_id = col2.selectbox("Recent Session (Current)", sessions, index=0)

        threshold = st.slider("Distance threshold (meters)", 0.05, 1.0, 0.1, 0.05)
        
        if st.button("Compare 3D Progress"):
            if s1_id == s2_id:
                st.warning("Please select two different sessions to compare.")
            else:
                s1 = SessionManager.load_session(s1_id)
                s2 = SessionManager.load_session(s2_id)
                
                if s1.reconstruction and s2.reconstruction and s1.reconstruction.success and s2.reconstruction.success:
                    with st.spinner("Aligning point clouds..."):
                        with open(s1.reconstruction.point_cloud_path, "rb") as f:
                            xyz1, rgb1 = _parse_ply_bytes(f.read())
                        with open(s2.reconstruction.point_cloud_path, "rb") as f:
                            xyz2, rgb2 = _parse_ply_bytes(f.read())
                            
                        aligned_xyz2, added_points, pct_change = compare_point_clouds(xyz1, xyz2, threshold=threshold)
                        
                        st.metric("Volumetric Progress / Structural Change", f"{pct_change}%")
                        
                        fig = _build_reconstruction_figure(xyz1, rgb1, None, added_points=added_points)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Both sessions must have successful 3D reconstructions to compare.")
    else:
        st.info("You need at least 2 processed sessions to track progress.")
