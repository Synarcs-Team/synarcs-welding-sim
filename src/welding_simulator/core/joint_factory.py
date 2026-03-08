import numpy as np
from isaacsim.core.api.objects import VisualCuboid, FixedCuboid
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.prims import RigidPrim, XFormPrim


def _euler_to_quat(roll, pitch, yaw):
    """Convert roll/pitch/yaw (radians) to quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def create_joint_from_config(cfg: dict, position=(0, 0, 0), name="weld_joint"):
    """
    Parses the joint_config and creates the appropriate 3D primitives in Isaac Sim.
    Returns: (parent_prim, list_of_collision_cuboids, bounding_box_dimensions)
    """
    jtype = cfg.get("joint_type", "tee")

    # ── Pose from explicit UI parameters ──────────────────────────────────────
    rotation_deg = float(cfg.get("rotation", 0))   # rotation around Z (table plane)
    tilt_deg     = float(cfg.get("tilt",     0))   # tilt forward (around X)
    flip         = bool(cfg.get("flip",      False))

    flip_rad  = np.pi if flip else 0.0
    tilt_rad  = np.deg2rad(tilt_deg)
    rot_rad   = np.deg2rad(rotation_deg)

    # Compose: first flip (Y-axis = upside down), then tilt (X-axis), then rotate (Z-axis/yaw)
    # This matches the JS preview convention exactly
    q_flip = _euler_to_quat(0, flip_rad, 0)   # Y-axis flip (upside-down)
    q_tilt = _euler_to_quat(tilt_rad, 0, 0)   # X-axis tilt (forward lean)
    q_rot  = _euler_to_quat(0, 0, rot_rad)    # Z-axis rotation (spin on table)

    def _qmul(a, b):
        aw,ax,ay,az = a; bw,bx,by,bz = b
        return np.array([aw*bw-ax*bx-ay*by-az*bz,
                         aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx,
                         aw*bz+ax*by-ay*bx+az*bw])

    ori = _qmul(q_rot, _qmul(q_tilt, q_flip))

    # 1. Create parent XForm and apply explicit pose
    create_prim(f"/World/{name}", "Xform")
    RigidPrim(f"/World/{name}")
    parent_prim = XFormPrim(f"/World/{name}")
    parent_prim.set_local_poses(np.array(position).reshape((1, 3)), ori.reshape((1, 4)))

    colls = []
    color = np.array([0.2, 0.2, 0.2])
    bbox = np.array([0.0, 0.0, 0.0]) # approximate (w, l, h) for camera planning

    def _add_plate(sub_name, dims, pos):
        create_prim(f"/World/{name}/{sub_name}", "Xform")
        VisualCuboid(prim_path=f"/World/{name}/{sub_name}/visual", name="visual",
                     translation=pos, scale=dims, color=color)
        c = FixedCuboid(prim_path=f"/World/{name}/{sub_name}/collision", name="collision",
                        translation=pos, scale=dims, visible=False)
        colls.append(c)

    # ── TEE JOINT ──
    if jtype == "tee":
        bw, bl, bt = cfg.get("bw", 0.15), cfg.get("bl", 0.15), cfg.get("bt", 0.025)
        sh, st = cfg.get("sh", 0.15), cfg.get("st", 0.025)

        base_dims = np.array([bw, bl, bt])
        base_pos  = base_dims / 2

        stem_dims = np.array([bw, st, sh])
        # Stem centered at the midpoint of base length (bl/2), sitting on top of base (z = bt)
        stem_pos  = np.array([bw/2, bl/2, sh/2 + bt])

        _add_plate("base", base_dims, base_pos)
        _add_plate("stem", stem_dims, stem_pos)
        bbox = np.array([bw, bl, bt + sh])


    # ── BUTT JOINT ──
    elif jtype == "butt":
        w, l, t = cfg.get("w", 0.15), cfg.get("l", 0.15), cfg.get("t", 0.025)
        gap = cfg.get("gap", 0.005)

        p1_dims = np.array([w, l, t])
        p1_pos = p1_dims / 2

        p2_dims = np.array([w, l, t])
        p2_pos = np.array([w/2, l/2 + l + gap, t/2])

        _add_plate("plate1", p1_dims, p1_pos)
        _add_plate("plate2", p2_dims, p2_pos)
        bbox = np.array([w, l * 2 + gap, t])

    # ── LAP JOINT ──
    elif jtype == "lap":
        w, l, t = cfg.get("w", 0.15), cfg.get("l", 0.15), cfg.get("t", 0.025)
        overlap = cfg.get("overlap", 0.05)
        
        # Bottom plate
        p1_dims = np.array([w, l, t])
        p1_pos = p1_dims / 2
        
        # Top plate (shifted over and up)
        p2_dims = np.array([w, l, t])
        p2_pos = np.array([w/2, l/2 + l - overlap, t/2 + t])

        _add_plate("plate_bottom", p1_dims, p1_pos)
        _add_plate("plate_top", p2_dims, p2_pos)
        bbox = np.array([w, l * 2 - overlap, t * 2])

    # ── CORNER JOINT ──
    elif jtype == "corner":
        w, l, t = cfg.get("w", 0.15), cfg.get("l", 0.15), cfg.get("t", 0.025)
        is_open = cfg.get("type", 0) == 1
        
        # Horizontal plate
        p1_dims = np.array([w, l, t])
        p1_pos = p1_dims / 2
        
        # Vertical plate
        p2_dims = np.array([w, t, l]) # Taller than it is thick
        
        if is_open:
            # Shifted completely off the edge
            p2_pos = np.array([w/2, t/2 + l, l/2 + t])
            bbox = np.array([w, l + t, t + l])
        else:
            # Flush with the edge
            p2_pos = np.array([w/2, t/2 + l - t, l/2 + t])
            bbox = np.array([w, l, t + l])

        _add_plate("plate_horiz", p1_dims, p1_pos)
        _add_plate("plate_vert", p2_dims, p2_pos)

    # ── EDGE JOINT ──
    elif jtype == "edge":
        w, l, t = cfg.get("w", 0.15), cfg.get("l", 0.15), cfg.get("t", 0.025)
        gap = cfg.get("gap", 0.002)

        p1_dims = np.array([w, t, l]) # standing upright
        p1_pos = p1_dims / 2

        p2_dims = np.array([w, t, l])
        p2_pos = np.array([w/2, t/2 + t + gap, l/2])

        _add_plate("plate1", p1_dims, p1_pos)
        _add_plate("plate2", p2_dims, p2_pos)
        bbox = np.array([w, t * 2 + gap, l])

    else:
        print(f"[WARN] Unknown joint type: {jtype}")

    return parent_prim, colls, bbox
