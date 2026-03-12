import pytest
import os
import json
import shutil
from pathlib import Path

from welding_simulator.api.main import DATA_DIR

def setup_module(module):
    # Ensure a clean slate for E2E tests
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def test_e2e_scan():
    from welding_simulator.sim.engines.pybullet.scanner import run_scan
    
    cfg = {
        "joint_type": "tee",
        "bw": 0.15, "bl": 0.15, "bt": 0.025,
        "sh": 0.15, "st": 0.025,
        "rotation": 0, "tilt": 0, "flip": False
    }
    
    run_scan(cfg=cfg, log_cb=lambda msg: None) # Mock log_cb to ignore output
    
    # Assert artifacts were created
    assert (DATA_DIR / "scan_0.pcd").exists()
    assert (DATA_DIR / "cam_pos_0.npy").exists()
    assert (DATA_DIR / "rgb_0.jpg").exists()
    
    # Check that video was encoded
    assert (DATA_DIR / "scan_video.mp4").exists()

def test_e2e_weld():
    from welding_simulator.sim.engines.pybullet.welder import run_weld
    
    # Mock the seams.json that should normally be created by the perception step
    seams = {
        "seam1": {
            "start": [0.75, 0.075, 1.025],
            "end": [0.75, -0.075, 1.025],
        },
        "seam2": {
            "start": [0.775, -0.075, 1.025],
            "end": [0.775, 0.075, 1.025],
        }
    }
    
    cfg = {
        "joint_type": "tee",
        "bw": 0.15, "bl": 0.15, "bt": 0.025,
        "sh": 0.15, "st": 0.025,
    }
    
    # write seams.json to simulate the perception step
    with open(DATA_DIR / "seams.json", "w") as f:
        json.dump(seams, f)
        
    run_weld(cfg=cfg, seams=seams, log_cb=None)
    
    # Assert weld video is created
    assert (DATA_DIR / "weld_video.mp4").exists()
