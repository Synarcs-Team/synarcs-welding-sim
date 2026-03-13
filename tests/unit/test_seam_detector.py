import os
import json
import numpy as np
import pytest
from unittest.mock import patch

from welding_simulator.perception.seam_detector import run_seam_detection

class DummyPlaneModel:
    def __init__(self):
        self.n = np.array([0.0, 0.0, 1.0])
        self.d = 0.0
        self.inliers = np.arange(10)

def test_seam_detection_fallback_on_math_error(tmp_path):
    """
    Test that if the mathematical intersection of planes throws an error,
    run_seam_detection catches it and cleanly writes a JSON {"error": ...} response
    instead of crashing the server/subprocess completely.
    """
    # 1) Redirect DATA_DIR so it reads/writes to a safe temp dir
    with patch("welding_simulator.perception.seam_detector.DATA_DIR", str(tmp_path)):
        
        # Make a dummy pointcloud to bypass the missing file check
        dummy_points = np.random.rand(50, 6)
        np.save(str(tmp_path / "merged_xyzrgb.npy"), dummy_points)
        
        # 2) Mock the initial plane RANSAC fitting step to inherently "succeed"
        with patch("welding_simulator.perception.seam_detector.ransac_plane_deterministic", return_value=DummyPlaneModel()):
            
            # 3) Mock the final intersection step to explicitly throw a math error
            with patch("welding_simulator.perception.seam_detector.seam_segment_from_planes", side_effect=RuntimeError("Simulated degenerate matrix")):
                
                # Execute the main function; if it lacks a try/except, this test will fail immediately.
                run_seam_detection()
                
                # Verify that it caught the exception and correctly serialized the JSON error block.
                res_path = tmp_path / "seam_results.json"
                assert res_path.exists(), "The fallback JSON was never written."
                
                with open(res_path, "r") as f:
                    res_json = json.load(f)
                    
                assert "error" in res_json
                assert "Simulated degenerate matrix" in res_json["error"]

