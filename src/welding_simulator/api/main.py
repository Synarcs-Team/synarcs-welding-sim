"""
webapp/main.py — FastAPI backend for the Welding Simulation Web App
"""
import asyncio
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import datetime
# main.py is in src/welding_simulator/api/
ROOT     = Path(__file__).parent.parent.parent.parent   # simulator/
DATA_DIR = ROOT / "data" / "latest"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SIM_PY   = ROOT / ".venv" / "bin" / "python"

app = FastAPI(title="Welding Simulation API")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Active subprocess (one at a time)
_proc: subprocess.Popen | None = None

async def stream_subprocess_with_logging(ws: WebSocket, cmd: list[str], log_prefix: str, env: dict | None = None):
    """Run a subprocess, stream stdout to websocket, and log to a timestamped file."""
    global _proc
    await ws.accept()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{log_prefix}_{timestamp}.log"
    
    try:
        if log_prefix in ["scan", "weld"]:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            
        _proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        loop = asyncio.get_event_loop()
        
        with open(log_path, "w") as log_file:
            log_file.write(f"--- Started: {timestamp} ---\n")
            log_file.write(f"--- Command: {' '.join(cmd)} ---\n\n")
            log_file.flush()
            
            while True:
                line = await loop.run_in_executor(None, _proc.stdout.readline)
                if not line:
                    break
                # Log to the file
                log_file.write(line)
                log_file.flush()
                # Stream to websocket
                await ws.send_text(line.rstrip())
                
            _proc.wait()
            exit_msg = f"[EXIT] code={_proc.returncode}"
            log_file.write(f"\n{exit_msg}\n")
            await ws.send_text(exit_msg)
            
    except WebSocketDisconnect:
        if _proc:
            _proc.terminate()
            with open(log_path, "a") as log_file:
                log_file.write("\n[EXIT] WebSocket disconnected, process terminated.\n")
    finally:
        _proc = None


@app.get("/")
async def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


# ── Step 1: Configure ─────────────────────────────────────────────────────────
@app.post("/api/configure")
async def configure(params: dict):
    """Save T-joint parameters and prepare a fresh session folder."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Clear previous session data safely
    import shutil
    for f in DATA_DIR.iterdir():
        if f.is_file() or f.is_symlink():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)
    config_path = ROOT / "config" / "joint_config.json"
    with open(config_path, "w") as f:
        json.dump(params, f, indent=4)
    return {"status": "ok", "saved": str(config_path)}


# ── Step 2: Scan (WebSocket streams log) ─────────────────────────────────────
@app.websocket("/ws/scan")
async def ws_scan(ws: WebSocket):
    # Run scanner via module so it can import welding_simulator.core
    cmd = [str(SIM_PY), "-m", "welding_simulator.sim.engines.isaac_sim.scanner"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    await stream_subprocess_with_logging(ws, cmd, "scan", env=env)


# ── Step 2b: Scan Video/Images ────────────────────────────────────────────────
@app.get("/api/scan-video")
async def scan_video():
    """Return the generated scan video."""
    video_path = DATA_DIR / "scan_video.mp4"
    if not video_path.exists():
        return JSONResponse({"error": "Video not found"}, status_code=404)
    return FileResponse(str(video_path), media_type="video/mp4")

@app.get("/api/scan-images")
async def scan_images():
    """Return RGB scan images as base64 strings."""
    images = []
    for i in range(10):
        path = DATA_DIR / f"rgb_{i}.jpg"
        if not path.exists():
            break
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        images.append({"index": i, "data": f"data:image/jpeg;base64,{b64}"})
    return JSONResponse({"images": images})


# ── Step 3: Process (WebSocket streams log) ───────────────────────────────────
@app.websocket("/ws/process")
async def ws_process(ws: WebSocket):
    cmd = [str(SIM_PY), "-m", "welding_simulator.perception.pipeline"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    await stream_subprocess_with_logging(ws, cmd, "process", env=env)


# ── Step 3b: Point cloud data ─────────────────────────────────────────────────
@app.get("/api/pointcloud")
async def pointcloud(max_points: int = 0):
    """
    Return merged point cloud as JSON for Plotly.
    max_points=0 → return all.  max_points=N → subsample to N.
    """
    xyz_path   = DATA_DIR / "merged_xyz.npy"
    seams_path = DATA_DIR / "seams.json"
    if not xyz_path.exists():
        return JSONResponse({"error": "No point cloud found. Run process first."}, status_code=404)

    pts = np.load(str(xyz_path))
    if max_points > 0 and len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    seams = {}
    if seams_path.exists():
        try:
            with open(seams_path) as f:
                seams = json.load(f)
        except Exception:
            pass

    return JSONResponse({
        "points": {
            "x": pts[:, 0].tolist(),
            "y": pts[:, 1].tolist(),
            "z": pts[:, 2].tolist(),
        },
        "total_points": int(len(np.load(str(xyz_path)))),
        "shown_points": int(len(pts)),
        "seams": seams,
    })


@app.get("/api/download-pcd")
async def download_pcd():
    """Download the raw merged point cloud."""
    pcd_path = DATA_DIR / "merged.pcd"
    if not pcd_path.exists():
        return JSONResponse({"error": "No point cloud file found. Run process first."}, status_code=404)
    return FileResponse(str(pcd_path), media_type="application/octet-stream", filename="merged.pcd")



# ── Step 4: Weld (WebSocket streams log) ──────────────────────────────────────
@app.websocket("/ws/weld")
async def ws_weld(ws: WebSocket):
    cmd = [str(SIM_PY), "-m", "welding_simulator.sim.engines.isaac_sim.welder"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    await stream_subprocess_with_logging(ws, cmd, "weld", env=env)

# ── Step 4b: Weld Video ───────────────────────────────────────────────────────
@app.get("/api/weld-video")
async def weld_video():
    """Return the generated weld video."""
    video_path = DATA_DIR / "weld_video.mp4"
    if not video_path.exists():
        return JSONResponse({"error": "Video not found"}, status_code=404)
    return FileResponse(str(video_path), media_type="video/mp4")

# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    session_files = list(DATA_DIR.glob("*")) if DATA_DIR.exists() else []
    return {
        "has_scan":   any(f.name.startswith("scan_") for f in session_files),
        "has_merged": (DATA_DIR / "merged_xyz.npy").exists(),
        "has_seams":  (DATA_DIR / "seams.json").exists(),
        "scan_count": sum(1 for f in session_files if f.name.startswith("scan_")),
    }
