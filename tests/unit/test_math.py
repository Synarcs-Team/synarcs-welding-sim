import numpy as np
import pytest
from welding_simulator.core.joint_factory import _euler_to_quat, _qmul

def test_euler_to_quat_zero():
    q = _euler_to_quat(0, 0, 0)
    np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-7)

def test_euler_to_quat_90_pitch():
    q = _euler_to_quat(0, np.pi/2, 0)
    # expect w=0.707, y=0.707
    expected = [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]
    np.testing.assert_allclose(q, expected, atol=1e-7)
    
def test_qmul_identity():
    q1 = [1.0, 0.0, 0.0, 0.0]
    q2 = [0.5, 0.5, 0.5, 0.5]
    res = _qmul(q1, q2)
    np.testing.assert_allclose(res, q2, atol=1e-7)

def test_qmul_rotations():
    # 90 deg around X
    q_x = [np.cos(np.pi/4), np.sin(np.pi/4), 0, 0]
    # 90 deg around Y
    q_y = [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]
    
    # Check that it multiplies without crashing and maintains unit length
    res = _qmul(q_x, q_y)
    assert np.isclose(np.linalg.norm(res), 1.0)
