# These_Justin
Simplification of Photogrammetry-Derived Digital Surface Models (DSM) for Radio Wave Propagation Applications

Author: OUATTARA Gninlnafanlan Justin

################################################################
 Author: OUATTARA Gninlnafanlan Justin
 Date: 2025-10-05

 Title: Simplification of Photogrammetric DSM for Radio Wave Propagation

 Description:
 This script simplifies a photogrammetry-derived Digital Surface Model (DSM)
 while preserving above-ground structures relevant for radio wave propagation analysis.

 The algorithm:
   1. Loads a 3D mesh (OBJ format)
   2. Fits a ground plane using RANSAC regression
   3. Extracts above-ground points (buildings, vegetation, etc.)
   4. Recreates a simplified, smoothed mesh including a flat ground plane
   5. Aligns the final mesh so that the minimum altitude equals zero

 Requirements:
   - Python 3.9+
   - pyvista
   - numpy
   - scikit-learn

 License: MIT (feel free to modify and share with attribution)
################################################################
