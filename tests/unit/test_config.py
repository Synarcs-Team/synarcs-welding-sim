import pytest
from welding_simulator.core.joint_factory import create_joint_from_config
from unittest.mock import patch

@patch('welding_simulator.core.joint_factory.create_prim')
@patch('welding_simulator.core.joint_factory.VisualCuboid')
@patch('welding_simulator.core.joint_factory.FixedCuboid')
@patch('welding_simulator.core.joint_factory.RigidPrim')
@patch('welding_simulator.core.joint_factory.XFormPrim')
def test_create_tee_joint_default_config(mock_xform, mock_rigid, mock_fixed, mock_vis, mock_create):
    from welding_simulator.core.joint_factory import create_joint_from_config
    # Empty config should use defaults
    cfg = {}
    parent, colls, bbox = create_joint_from_config(cfg, name="test_joint")
    
    # Assert bbox handles tee default
    assert bbox.shape == (3,)
    assert bbox[0] == 0.15 # default bw
    assert bbox[1] == 0.15 # default bl
    assert bbox[2] == 0.025 + 0.15 # default bt + sh

@patch('welding_simulator.core.joint_factory.create_prim')
@patch('welding_simulator.core.joint_factory.VisualCuboid')
@patch('welding_simulator.core.joint_factory.FixedCuboid')
@patch('welding_simulator.core.joint_factory.RigidPrim')
@patch('welding_simulator.core.joint_factory.XFormPrim')
def test_create_butt_joint_custom_config(mock_xform, mock_rigid, mock_fixed, mock_vis, mock_create):
    from welding_simulator.core.joint_factory import create_joint_from_config
    cfg = {
        "joint_type": "butt",
        "w": 0.2,
        "l": 0.3,
        "t": 0.05,
        "gap": 0.01
    }
    parent, colls, bbox = create_joint_from_config(cfg)
    
    # Check bbox for butt joint
    assert bbox[0] == 0.2
    assert bbox[1] == 0.3 * 2 + 0.01
    assert bbox[2] == 0.05
