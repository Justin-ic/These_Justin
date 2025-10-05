################################################################
# Author: OUATTARA Gninlnafanlan Justin
# Date: 2025-10-05
#
# Title: Simplification of Photogrammetric DSM for Radio Wave Propagation
#
# Description:
# This script simplifies a photogrammetry-derived Digital Surface Model (DSM)
# while preserving above-ground structures relevant for radio wave propagation analysis.
#
# The algorithm:
#   1. Loads a 3D mesh (OBJ format)
#   2. Fits a ground plane using RANSAC regression
#   3. Extracts above-ground points (buildings, vegetation, etc.)
#   4. Recreates a simplified, smoothed mesh including a flat ground plane
#   5. Aligns the final mesh so that the minimum altitude equals zero
#
# Requirements:
#   - Python 3.9+
#   - pyvista
#   - numpy
#   - scikit-learn
#
# License: MIT (feel free to modify and share with attribution)
################################################################

import pyvista as pv
import numpy as np
from sklearn.linear_model import RANSACRegressor


# 1) Load the 3D mesh
mesh = (
    pv.read("Carto_cite_H40_3D_Map_simplified_3d_mesh.obj")
    .extract_surface()
    .clean()
    .triangulate()
)

# 2) Count the number of triangular faces
triangle_faces = mesh.faces.reshape((-1, 4))  # [3, id1, id2, id3] per triangle
n_triangles = np.sum(triangle_faces[:, 0] == 3)
print(f"Number of triangles: {n_triangles}")

# 3) Extract XYZ coordinates
points = mesh.points
X = points[:, :2]  # (X, Y)
Z = points[:, 2]   # (Z)

# 4) Estimate ground plane using RANSAC regression
model = RANSACRegressor()
model.fit(X, Z)
Z_pred = model.predict(X)

# 5) Compute distances between points and estimated ground plane
distance_to_plane = Z - Z_pred
threshold = 0.3  # in meters (tune as needed)

# 6) Keep only points significantly above the ground
above_ground_indices = np.where(distance_to_plane > threshold)[0]
mesh_above = mesh.extract_points(above_ground_indices, adjacent_cells=True)

# 7) Compute bounds to define the replacement ground plane
xmin, xmax, ymin, ymax = mesh.bounds[0], mesh.bounds[1], mesh.bounds[2], mesh.bounds[3]

# 8) Compute max absolute extent (useful for centering)
abs_x_max = int(max(abs(xmin), abs(xmax)))
abs_y_max = int(max(abs(ymin), abs(ymax)))

# 9) Generate a centered 2x2 grid (ground plane extent)
grid_x, grid_y = np.meshgrid([-abs_x_max, abs_x_max], [-abs_y_max, abs_y_max])
grid_xy = np.c_[grid_x.ravel(), grid_y.ravel()]
grid_z = model.predict(grid_xy)

# 10) Create a 3D mesh from the ground plane points
vertices = np.c_[grid_xy, grid_z]
faces = [[4, 0, 1, 3, 2]]  # rectangle with 4 vertices (correct ordering)
ground_plane = pv.PolyData(vertices, faces)

# 11) Clean and smooth the above-ground mesh
mesh_above_ = mesh_above.extract_surface().clean().triangulate()
mesh_smooth = mesh_above_.smooth(
    n_iter=20, relaxation_factor=0.5, edge_angle=20, feature_angle=45
)

# 12) Merge the ground plane and above-ground mesh
combined = pv.PolyData()
combined = combined.merge(mesh_above_)
combined = combined.merge(ground_plane)

# 13) Find minimum altitude (z_min)
z_min = combined.bounds[4]  # bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

# 14) Translate the mesh so that z_min = 0
mesh_aligned = combined.translate((0, 0, -z_min))

# 15) Export final mesh (uncomment to save)
# mesh_aligned.save("mesh_simplified_aligned.obj")

# 16) Visualization
plotter = pv.Plotter()
plotter.add_mesh(mesh_aligned, scalars=mesh_aligned.points[:, 2], cmap="viridis")
plotter.show()
