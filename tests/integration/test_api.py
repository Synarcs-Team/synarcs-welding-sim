import pytest
from fastapi.testclient import TestClient
import json
from pathlib import Path
from welding_simulator.api.main import app, DATA_DIR, ROOT

client = TestClient(app)

def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "has_scan" in data
    assert "has_merged" in data
    assert "has_seams" in data
    assert "scan_count" in data

def test_configure_endpoint_and_read_config():
    payload = {
        "joint_type": "butt",
        "w": 0.2,
        "l": 0.3,
        "t": 0.05,
        "gap": 0.01,
        "sim_engine": "pybullet"
    }
    response = client.post("/api/configure", json=payload)
    assert response.status_code == 200
    
    config_path = ROOT / "config" / "joint_config.json"
    assert config_path.exists()
    
    with open(config_path, "r") as f:
        saved_cfg = json.load(f)
        
    assert saved_cfg["joint_type"] == "butt"
    assert saved_cfg["sim_engine"] == "pybullet"

def test_scan_video_404():
    # If file doesn't exist or we just cleaned up
    video_path = DATA_DIR / "scan_video.mp4"
    if video_path.exists():
        video_path.unlink()
        
    response = client.get("/api/scan-video")
    assert response.status_code == 404
