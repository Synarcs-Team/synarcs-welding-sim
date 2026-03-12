# T-Joint Seam Detection Algorithm

This document provides a technical overview of the seam detection algorithm used in the Synarcs Welding Simulator. The module (Stage 4) identifies weld seams on T-joint models directly from processed point clouds.

## Overview

The algorithm uses a **constrained RANSAC (Random Sample Consensus)** approach to fit geometric primitives (planes) to the point cloud. By identifying the base plate and the vertical stem faces, the algorithm can calculate the precise intersection lines where the weld seams are located.

---

## 1. Input Data
- **Source**: `merged_xyzrgb.npy` (generated in the Process step).
- **Format**: An $N \times 6$ array representing spatial coordinates ($X, Y, Z$) and color information ($R, G, B$).
- **Preprocessing**: Voxel downsampling may be applied to reduce noise and improve performance while preserving geometric features.

## 2. Plane Fitting Hierarchy

To ensure robustness, the algorithm fits planes in a specific order with geometric constraints:

### A. Base Plane Fitting
- **Goal**: Identify the surface of the flat table or base plate.
- **Constraint**: The plane normal must be within $25^\circ$ of the global **World Up** vector $[0, 0, 1]$.
- **Method**: 
  1. Sample 3 points to define a candidate plane.
  2. Verify if the normal satisfies the "Near Up" constraint.
  3. Count inliers (points within $5\text{mm}$ of the plane).
  4. Refine the best candidate using **Total Least Squares (TLS)** on all inliers.

### B. Stem Face Fitting (Face 1 & 2)
- **Goal**: Identify the two vertical sides of the T-joint stem.
- **Constraint**: The plane normal must be **perpendicular** (within $20^\circ$) to the already detected Base Plane normal.
- **Method**:
  1. Remove Base Plane inliers from the cloud.
  2. Perform RANSAC on the remaining points.
  3. Verify the perpendicular constraint.
  4. Repeat for the second face after removing Face 1 inliers.

---

## 3. Seam Line Computation

For each stem face, the algorithm computes the intersection line with the base plane:

The intersection of two planes defined by normals $n_1, n_2$ and distances $d_1, d_2$ is a line $L(t) = p_0 + t \cdot v$:
- **Direction ($v$)**: $n_1 \times n_2$ (normalized).
- **Point ($p_0$)**: A point satisfying both plane equations, found via linear algebra.

---

## 4. Robust Trimming & Filtering

The intersection line is theoretically infinite. To find the physical extents of the seam, several filters are applied:

1.  **Dual-Plane Filter**: Only points within $10\text{mm}$ of *both* the Base and Stem planes are considered "candidate seam points."
2.  **Tube Filter**: Points must be within a $20\text{mm}$ radius of the computed intersection line.
3.  **Percentile Trimming**: To avoid edge noise, the algorithm projects the candidate points onto the line and takes the **2nd to 98th percentile** as the final start and end points.

---

## 5. Toolpath Offsets

For robotic welding, the algorithm generates parallel toolpaths:
- **Travel Direction**: The unit vector $v$ along the seam.
- **Side Direction**: Computed as $n_{base} \times v$.
- **Offset Paths**: Shifts the center seam line by $\pm 3\text{mm}$ along the side direction to provide "Left" and "Right" target lines for the welder.

---

## 6. Output (seam_results.json)

The results are exported as a structured JSON object containing:
- Start/End coordinates for the center seam and offset paths.
- Unit vectors for Travel, Side, and Bisector (for torch orientation).
- Statistics about the fitted planes (inlier counts, normals).
