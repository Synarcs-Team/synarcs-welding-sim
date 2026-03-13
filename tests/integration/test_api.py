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

def test_seam_results_error_fallback():
    """Verify that if the detector leaves an error JSON, the API returns it correctly."""
    error_data = {"error": "Seam mathematical computation failed: mock error"}
    results_path = DATA_DIR / "seam_results.json"
    
    # Ensure dir exists and write the fake error file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(error_data, f)
        
    try:
        response = client.get("/api/seam-results")
        assert response.status_code == 200
        assert "error" in response.json()
        assert "mock error" in response.json()["error"]
    finally:
        results_path.unlink(missing_ok=True)
