import streamlit as st
from PIL import Image
import numpy as np
import cv2
import zipfile
import json
from datetime import datetime
from io import BytesIO
from detect import run_inference
from progress_tracker import detect_progress_change

def export_report(counts, zones, hazards):
    report = {
        "timestamp": datetime.now().isoformat(),
        "detections": counts,
        "hazards": hazards,
        "progress_zones": zones,
        "overall_change_pct": sum(zones.values()) / len(zones) if zones else 0
    }
    return json.dumps(report, indent=2)

st.set_page_config(page_title="SiteVision", layout="wide")

st.title("🚧 SiteVision: Construction Site Monitor")
st.markdown("Monitor construction site safety and structural progress using computer vision.")

tab1, tab2 = st.tabs(["Object & Safety Detection", "Progress Tracking (Frame-Diff)"])

with tab1:
    st.header("👷 Object & Safety Detection")
    st.markdown("Upload a site image to detect workers, machinery, and PPE compliance.")
    
    uploaded_files = st.file_uploader("Choose up to 5 images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="detect_uploader")
    
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning("You uploaded more than 5 images. Only the first 5 will be processed.")
            uploaded_files = uploaded_files[:5]
            
        if "results_cache" not in st.session_state:
            st.session_state.results_cache = {}
            
        run_inference_btn = st.button("Run Detection")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### {uploaded_file.name}")
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
            if run_inference_btn:
                with st.spinner(f"Running Inference on {uploaded_file.name}..."):
                    output_image, counts, hazards = run_inference(img_array)
                    st.session_state.results_cache[uploaded_file.name] = (output_image, counts, hazards)
                    
            if uploaded_file.name in st.session_state.results_cache:
                output_image, counts, hazards = st.session_state.results_cache[uploaded_file.name]
                with col2:
                    st.image(output_image, caption="Detection Results", use_container_width=True, channels="BGR")
                    
                    is_success, buffer = cv2.imencode(".png", output_image)
                    if is_success:
                        st.download_button(
                            label="⬇️ Download Result",
                            data=buffer.tobytes(),
                            file_name=f"result_{uploaded_file.name}",
                            mime="image/png",
                            key=f"dl_{idx}_{uploaded_file.name}"
                        )
                
                if counts:
                    metric_cols = st.columns(len(counts))
                    for m_idx, (cls_name, count) in enumerate(counts.items()):
                        metric_cols[m_idx % len(metric_cols)].metric(label=cls_name, value=count)
                else:
                    st.info("No objects detected.")
                    
                if hazards:
                    st.error("🚨 Safety Hazards Detected:")
                    for h in hazards:
                        st.write(h)
                
                report_json = export_report(counts, {}, hazards)
                st.download_button(
                    label="📄 Download JSON Report",
                    data=report_json,
                    file_name=f"report_{uploaded_file.name}.json",
                    mime="application/json",
                    key=f"rep_{idx}_{uploaded_file.name}"
                )
            
            st.divider()
            
        if any(f.name in st.session_state.results_cache for f in uploaded_files):
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in uploaded_files:
                    if f.name in st.session_state.results_cache:
                        output_image, _, _ = st.session_state.results_cache[f.name]
                        is_success, img_buf = cv2.imencode(".png", output_image)
                        if is_success:
                            zf.writestr(f"result_{f.name}.png", img_buf.tobytes())
                            
            st.download_button(
                label="📦 Download All Results (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="all_detection_results.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary"
            )

with tab2:
    st.header("📈 Progress Tracking")
    st.markdown("Upload two timestamped images of the same site to detect structural changes.")
    
    col1, col2 = st.columns(2)
    with col1:
        img1_file = st.file_uploader("Upload Image 1 (Before)", type=["jpg", "jpeg", "png"])
    with col2:
        img2_file = st.file_uploader("Upload Image 2 (After)", type=["jpg", "jpeg", "png"])
        
    if img1_file and img2_file:
        img1 = Image.open(img1_file)
        img2 = Image.open(img2_file)
        
        col1.image(img1, caption="Before", use_container_width=True)
        col2.image(img2, caption="After", use_container_width=True)
        
        if "diff_cache" not in st.session_state:
            st.session_state.diff_cache = {}
            
        if st.button("Compare Frames"):
            with st.spinner("Computing structural differences..."):
                img1_array = np.array(img1)
                img2_array = np.array(img2)
                
                diff_image, change_percent, zones = detect_progress_change(img1_array, img2_array)
                st.session_state.diff_cache['last_run'] = (diff_image, change_percent, zones)
                
        if 'last_run' in st.session_state.diff_cache:
            diff_image, change_percent, zones = st.session_state.diff_cache['last_run']
            
            st.subheader("Progress Analysis")
            st.metric("Site Area Changed", f"{change_percent:.2f}%")
            
            st.write("### Spatial Change Zones")
            zone_cols = st.columns(3)
            for idx, (zone, pct) in enumerate(zones.items()):
                zone_cols[idx % 3].metric(zone, f"{pct}%")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(diff_image, caption="Highlighted Changes (Red)", use_container_width=True)
                
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(diff_image, cv2.COLOR_RGB2BGR))
                if is_success:
                    st.download_button(
                        label="⬇️ Download Diff Image",
                        data=buffer.tobytes(),
                        file_name="progress_diff.png",
                        mime="image/png"
                    )
            
            with col_res2:
                report_json = export_report({}, zones, [])
                st.download_button(
                    label="📄 Download Progress JSON Report",
                    data=report_json,
                    file_name="progress_report.json",
                    mime="application/json"
                )
