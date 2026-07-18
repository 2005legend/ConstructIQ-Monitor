import numpy as np
import re
import struct
import os
import logging

logger = logging.getLogger("SiteVision")
def _parse_ply_bytes(raw: bytes, max_points: int = 60_000):
    """Parse PLY from raw bytes. Returns (xyz ndarray, rgb ndarray)."""
    if not raw:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    try:
        header_end = raw.find(b"end_header")
        if header_end == -1:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

        header = raw[:header_end].decode("ascii", errors="replace")
        body_offset = header_end + len("end_header") + 1

        m = re.search(r"element vertex\s+(\d+)", header)
        if not m:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
        n_verts = int(m.group(1))
        if n_verts == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

        prop_types = re.findall(r"property\s+(\S+)\s+\S+", header)
        props      = re.findall(r"property\s+\S+\s+(\S+)", header)
        has_rgb    = "red" in props or "r" in props

        type_sizes = {
            "float": 4, "float32": 4, "double": 8, "float64": 8,
            "uchar": 1, "uint8": 1,  "char": 1,   "int8": 1,
            "short": 2, "int16": 2,  "ushort": 2, "uint16": 2,
            "int":   4, "int32": 4,  "uint":   4, "uint32": 4,
        }
        vertex_size = sum(type_sizes.get(t, 4) for t in prop_types)
        endian = "<" if "binary_little_endian" in header else ">"

        step = max(1, n_verts // max_points)
        xyz_list, rgb_list = [], []

        for i in range(0, n_verts, step):
            offset = body_offset + i * vertex_size
            if offset + vertex_size > len(raw):
                break
            chunk    = raw[offset: offset + vertex_size]
            byte_pos = 0
            row_vals = []
            for t in prop_types:
                sz = type_sizes.get(t, 4)
                if t in ("float", "float32"):
                    v = struct.unpack_from(endian + "f", chunk, byte_pos)[0]
                elif t in ("double", "float64"):
                    v = struct.unpack_from(endian + "d", chunk, byte_pos)[0]
                elif t in ("uchar", "uint8"):
                    v = struct.unpack_from("B", chunk, byte_pos)[0]
                else:
                    v = struct.unpack_from(endian + "i", chunk, byte_pos)[0]
                row_vals.append(v)
                byte_pos += sz
            xyz_list.append(row_vals[:3])
            if has_rgb and len(row_vals) >= 6:
                rgb_list.append([int(row_vals[3]), int(row_vals[4]), int(row_vals[5])])
            else:
                rgb_list.append([180, 180, 180])

        if not xyz_list:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

        xyz = np.array(xyz_list, dtype=float)
        rgb = np.array(rgb_list, dtype=np.uint8)

        # Drop NaN / Inf immediately
        finite = np.all(np.isfinite(xyz), axis=1)
        return xyz[finite], rgb[finite]

    except Exception as e:
        logger.warning("PLY byte-parse failed: %s", e)
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)



def _build_reconstruction_figure(xyz, rgb, camera_poses, added_points=None):
    """Build a rich Plotly 3D figure with point cloud + camera frustums."""
    import plotly.graph_objects as go

    traces = []

    # --- Point cloud ---
    if len(xyz) > 0:
        # Remove statistical outliers (points > 3σ from median)
        median = np.median(xyz, axis=0)
        std = np.std(xyz, axis=0) + 1e-9
        mask = np.all(np.abs(xyz - median) < 3 * std, axis=1)
        xyz_clean = xyz[mask]
        rgb_clean = rgb[mask]

        if len(xyz_clean) > 0:
            colors = [f"rgb({r},{g},{b})" for r, g, b in rgb_clean]
            traces.append(go.Scatter3d(
                x=xyz_clean[:, 0],
                y=xyz_clean[:, 1],
                z=xyz_clean[:, 2],
                mode="markers",
                marker=dict(
                    size=1.5,
                    color=colors,
                    opacity=0.7,
                ),
                name=f"Baseline Point Cloud ({len(xyz_clean):,} pts)",
                hoverinfo="skip",
            ))
            
    # --- Added points ---
    if added_points is not None and len(added_points) > 0:
        traces.append(go.Scatter3d(
            x=added_points[:, 0],
            y=added_points[:, 1],
            z=added_points[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color="lime",
                opacity=0.9,
            ),
            name=f"New Structure ({len(added_points):,} pts)",
            hoverinfo="skip",
        ))

    # --- Camera positions ---
    if camera_poses:
        cx = [p.position[0] for p in camera_poses]
        cy = [p.position[1] for p in camera_poses]
        cz = [p.position[2] for p in camera_poses]
        labels = [p.image_id for p in camera_poses]

        traces.append(go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode="markers+text",
            marker=dict(
                size=6,
                color="red",
                symbol="diamond",
                line=dict(color="darkred", width=1),
            ),
            text=[os.path.basename(lbl) for lbl in labels],
            textposition="top center",
            textfont=dict(size=9, color="red"),
            name="Camera Positions",
        ))

        # Draw lines connecting cameras in order (trajectory)
        traces.append(go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode="lines",
            line=dict(color="rgba(255,80,80,0.5)", width=2),
            name="Camera Trajectory",
            hoverinfo="skip",
        ))

    fig = go.Figure(data=traces)
    n_cameras = len(camera_poses) if camera_poses else 0
    fig.update_layout(
        title=dict(
            text=f"3D Site Reconstruction — {len(xyz):,} points, {n_cameras} cameras",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(title="X (m)", backgroundcolor="rgb(240,240,255)", showgrid=True),
            yaxis=dict(title="Y (m)", backgroundcolor="rgb(240,255,240)", showgrid=True),
            zaxis=dict(title="Z (m)", backgroundcolor="rgb(255,245,240)", showgrid=True),
            aspectmode="data",  # keeps real-world proportions
            bgcolor="rgb(20,20,30)",
        ),
        paper_bgcolor="rgb(20,20,30)",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(30,30,40,0.8)",
            bordercolor="gray",
            font=dict(color="white"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=650,
    )
    return fig

