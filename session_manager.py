import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

from models import Session, ReconstructionResult, CameraPose, Detection, Violation

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")

class SessionManager:
    @staticmethod
    def _ensure_dir():
        if not os.path.exists(SESSIONS_DIR):
            os.makedirs(SESSIONS_DIR)

    @staticmethod
    def create_session(image_bytes_list: List[Tuple[str, bytes]]) -> Session:
        SessionManager._ensure_dir()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
        session_dir = os.path.join(SESSIONS_DIR, session_id)
        os.makedirs(session_dir)
        
        img_dir = os.path.join(session_dir, "images")
        os.makedirs(img_dir)
        
        image_paths = []
        for name, b in image_bytes_list:
            path = os.path.join(img_dir, name)
            with open(path, "wb") as f:
                f.write(b)
            image_paths.append(path)
            
        return Session(
            session_id=session_id,
            timestamp=datetime.now(),
            image_paths=image_paths
        )
        
    @staticmethod
    def get_session_dir(session_id: str) -> str:
        return os.path.join(SESSIONS_DIR, session_id)
        
    @staticmethod
    def save_session(session: Session):
        SessionManager._ensure_dir()
        session_dir = os.path.join(SESSIONS_DIR, session.session_id)
        
        data = {
            "session_id": session.session_id,
            "timestamp": session.timestamp.isoformat(),
            "image_paths": session.image_paths,
            "has_reconstruction": session.reconstruction is not None and session.reconstruction.success,
            "point_cloud_path": session.reconstruction.point_cloud_path if session.reconstruction else None,
        }
        
        # Save camera poses
        if session.reconstruction and session.reconstruction.camera_poses:
            poses = []
            for p in session.reconstruction.camera_poses:
                poses.append({
                    "image_id": p.image_id,
                    "position": p.position,
                    "rotation": p.rotation,
                    "confidence": p.confidence,
                    "intrinsics": p.intrinsics.tolist() if p.intrinsics is not None else None
                })
            data["camera_poses"] = poses
            
        # Save violations
        if session.violations:
            viols = []
            for v in session.violations:
                viols.append({
                    "id": v.id,
                    "type": v.type,
                    "worker_id": v.worker_id,
                    "severity": v.severity,
                    "location_2d": v.location_2d,
                    "location_3d": v.location_3d,
                    "confidence": v.confidence,
                    "timestamp": v.timestamp.isoformat(),
                    "image_id": v.image_id,
                })
            data["violations"] = viols
            
        with open(os.path.join(session_dir, "meta.json"), "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def list_sessions() -> List[str]:
        SessionManager._ensure_dir()
        return sorted([d for d in os.listdir(SESSIONS_DIR) if os.path.isdir(os.path.join(SESSIONS_DIR, d))], reverse=True)

    @staticmethod
    def load_session(session_id: str) -> Session:
        session_dir = os.path.join(SESSIONS_DIR, session_id)
        meta_path = os.path.join(session_dir, "meta.json")
        
        if not os.path.exists(meta_path):
            return None
            
        with open(meta_path, "r") as f:
            data = json.load(f)
            
        session = Session(
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            image_paths=data["image_paths"]
        )
        
        if data.get("has_reconstruction"):
            poses = []
            for p in data.get("camera_poses", []):
                poses.append(CameraPose(
                    image_id=p["image_id"],
                    position=tuple(p["position"]),
                    rotation=tuple(p["rotation"]),
                    confidence=p["confidence"],
                    intrinsics=np.array(p["intrinsics"]) if p["intrinsics"] else None
                ))
            session.reconstruction = ReconstructionResult(
                success=True,
                camera_poses=poses,
                point_cloud_path=data.get("point_cloud_path"),
                quality_score=1.0,
                error_message=None,
                processing_time=0.0
            )
            
        if "violations" in data:
            viols = []
            for v in data["violations"]:
                viol = Violation(
                    id=v["id"],
                    type=v["type"],
                    worker_id=v["worker_id"],
                    severity=v["severity"],
                    location_2d=tuple(v["location_2d"]),
                    location_3d=tuple(v["location_3d"]) if v["location_3d"] else None,
                    confidence=v["confidence"],
                    timestamp=datetime.fromisoformat(v["timestamp"]),
                    image_id=v.get("image_id"),
                )
                viols.append(viol)
            session.violations = viols
            
        return session
