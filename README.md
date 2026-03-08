# Synarcs Welding Simulator

**[Synarcs](https://synarcs.com/)** bridges the virtual and real worlds with hybrid simulation environments to accelerate industrial robot deployment, training, and optimization.

This repository contains the **Synarcs Welding Simulator**, a browser-based visualization and testing workspace integrating interactive UI configurations with **fast, serverless PyBullet simulation** and **high-fidelity NVIDIA Isaac Sim** point cloud generation. It serves as a foundational tool for developing and validating robotic welding autonomy before physical deployment.

## Cloud vs Local Usage

The simulator can be accessed online via our **[Cloud Deployment (Link coming soon)](#)** for lightweight CPU simulations using the PyBullet engine. 

If you want to leverage the **NVIDIA Isaac Sim** engine for GPU-accelerated high-fidelity robotics pipelines, you must run this project **locally** on your own machine. Please follow the installation instructions below to download and set up the local workspace.

## Features

- **Dynamic Joint Configuration**: Live 3D parameterization of 5 standard AWS welding joints (Tee, Butt, Lap, Corner, Edge).
- **Dual Simulation Engines**: 
  - **PyBullet**: Lightning-fast, lightweight CPU physics and rendering engine perfect for web deployments and standard laptops.
  - **NVIDIA Isaac Sim**: High-fidelity, GPU-accelerated robotics simulation platform for advanced testing.
- **Automated Scanning**: Headless orchestration of the robotic arm to navigate and capture multi-angle depth point clouds of the parameterized joint.
- **Point Cloud Processing**: Automatic merging and visualization of the scanned point cloud data natively in your browser.
- **Modular Pipeline**: Built using a modern Python package structure (`src/welding_simulator`) designed to support pluggable simulation engines and computer vision seam detection algorithms.

## System Requirements

- Ubuntu 22.04 or 24.04
- Python 3.11
- (Optional) NVIDIA GPU (RTX series recommended) with latest drivers for Isaac Sim. *Not required for PyBullet.*
- (Optional) NVIDIA Isaac Sim (v4+ or equivalent `isaacsim` pip package)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Synarcs-Team/synarcs-welding-sim.git
   cd synarcs-welding-sim
   ```

2. **Create and Activate a Virtual Environment** (requires Python 3.11 for Isaac Sim compatibility):
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install fastapi uvicorn websockets open3d numpy pillow opencv-python "pydantic>=2.9.2"
   ```

4. **Install NVIDIA Isaac Sim**:
   *Note: This package requires access to the NVIDIA package index.*
   ```bash
   pip install isaacsim --extra-index-url https://pypi.nvidia.com
   ```

## Usage

Start the simulator web interface by running the launcher script:

```bash
bash scripts/run_webapp.sh
```

Navigate to `http://localhost:8000` in your web browser.

### The Simulation Pipeline

The interface guides you through three distinct simulation stages:

#### 1. Configure
Select your joint type, compute engine, and use the parametric sliders to adjust dimensions interactively. View the live 3D preview and click **Save Configuration**.

![Configure Step View](docs/images/configure_step.png)

#### 2. Scan
Click **Start Scanning**. This triggers a headless simulation backend orchestrator (PyBullet or Isaac Sim). A robotic arm is spawned in the engine, your custom joint geometry is dynamically mapped to the table, and the camera uses path-planning to navigate to 5 optimal viewpoints around the weld. Wait for the live progress bar to reach 100% and view the captured image results.

![Scan Step Complete](docs/images/scan_step.png)

#### 3. Process
Click **Merge Point Clouds**. The backend processes the 5 distinct raw captures output by the previous step and merges them into a solitary 3D reconstruction using Open3D. You can inspect the dense point cloud directly in the browser viewport.

![Process Step Point Cloud Visualization](docs/images/process_step.png)

## Developer Documentation & Architecture

The codebase has been refactored from standalone shell scripts into a modular, production-ready Python package hosted in `src/welding_simulator/`.

### Directory Structure

```text
simulator/
├── config/                  # Joint and algorithm configurations
├── data/latest/             # Automatically generated simulation outputs
├── scripts/                 # Entrypoint bash scripts
├── src/
│   └── welding_simulator/   # Main Python package
│       ├── api/             # FastAPI backend implementation
│       ├── core/            # Business logic and geometry generation (joint_factory.py)
│       ├── perception/      # Standalone Computer Vision processing and seam algorithms
│       └── sim/             # Abstact simulation logic and engines (Isaac Sim implementation)
└── docs/                    # Static assets used in documentation
```

### Extending the Simulator

- **Simulation Engines**: The codebase abstracts simulation commands under `src/welding_simulator/sim/base.py`. Developers can inherit this base class to create wrappers for new engines (e.g., Gazebo or PyBullet) alongside the existing `isaac_sim` implementation.
- **Seam Finding Algorithms**: Base algorithm pipelines live in `src/welding_simulator/perception/`. Custom algorithms can be plugged in to replace or supplement standard planar heuristics and point cloud analysis.
