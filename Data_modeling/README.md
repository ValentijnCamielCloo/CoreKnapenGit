


#### 3.1  Pointcloud processing
   - **Translation & Orientation:** Aligning the point cloud with a reference model using `pcd.translate_orientate()`.
   - **Ground Removal:** Filtering out unwanted points (e.g., ground) based on color thresholds using `pcd.filter_colors()`.
   - **Registration:** Merging multiple scans to create a single unified point cloud using `pcd.registration()`.
   - **Downsampling:** Reducing the number of points using voxel downsampling (`pcd.voxel_downsample()`).
   - **Clustering:** Grouping similar points together using K-means clustering (`pcd.cluster_kmeans_normals()`), removing outliers, and preparing for the final model.
#### 3.2 Comparing calculated results
   - Comparing the processed point cloud to a reference mesh using `ComparePCDMesh()`.
   - Writing final results to CSV files for documentation and analysis using `compare.write_results()`.


![Workflow Overview](img/maintask.png)
