"""
Dense Multi-View Stereo (MVS) Reconstruction
Produces high-quality dense point clouds from multiple views
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
import subprocess
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DenseReconstruction:
    """Dense reconstruction using Multi-View Stereo"""

    def __init__(
        self,
        frames: List[np.ndarray],
        poses: List[np.ndarray],
        intrinsics: np.ndarray,
        output_dir: str = "./dense_output"
    ):
        """
        Initialize dense reconstruction

        Args:
            frames: List of input frames
            poses: Camera poses (num_frames, 4, 4)
            intrinsics: Camera intrinsic matrix (3, 3)
            output_dir: Output directory for results
        """
        self.frames = frames
        self.poses = poses
        self.intrinsics = intrinsics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.point_cloud = None
        self.mesh = None
        
        logger.info(f"Initialized dense reconstruction with {len(frames)} views")

    def create_depth_maps(self) -> List[np.ndarray]:
        """
        Estimate depth maps for all frames using block matching

        Returns:
            List of depth maps
        """
        logger.info("📊 Computing depth maps...")
        
        depth_maps = []
        
        # Create stereo matcher
        stereo = cv2.StereoBM_create(
            numDisparities=16 * 5,
            blockSize=15
        )
        
        for i in tqdm(range(len(self.frames) - 1), desc="Depth estimation"):
            frame1 = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(self.frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Compute disparity
            disparity = stereo.compute(frame1, frame2)
            disparity = np.float32(disparity) / 16.0
            
            # Convert to depth
            focal_length = self.intrinsics[0, 0]
            baseline = np.linalg.norm(self.poses[i+1, :3, 3] - self.poses[i, :3, 3])
            
            depth = (focal_length * baseline) / (disparity + 1e-3)
            depth[depth < 0.1] = 0
            depth[depth > 1000] = 0
            
            depth_maps.append(depth)
        
        logger.info(f"✓ Computed {len(depth_maps)} depth maps")
        return depth_maps

    def depth_to_point_cloud(
        self,
        depth_maps: List[np.ndarray],
        use_colors: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert depth maps to 3D point cloud

        Args:
            depth_maps: List of depth maps
            use_colors: Include color information

        Returns:
            Point cloud, color array
        """
        logger.info("🎯 Converting depth maps to point cloud...")
        
        points = []
        colors = []
        
        K_inv = np.linalg.inv(self.intrinsics)
        
        for frame_idx in tqdm(range(len(depth_maps)), desc="Point cloud generation"):
            depth = depth_maps[frame_idx]
            pose = self.poses[frame_idx]
            frame = self.frames[frame_idx]
            
            h, w = depth.shape
            
            # Create pixel coordinates
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            
            # Get valid depth pixels
            valid = depth > 0
            
            # Unproject to camera coordinates
            xy1 = np.stack([x[valid], y[valid], np.ones_like(x[valid])], axis=-1)
            xyz_cam = (K_inv @ xy1.T).T * depth[valid, np.newaxis]
            
            # Transform to world coordinates
            R = pose[:3, :3]
            t = pose[:3, 3]
            xyz_world = (R.T @ (xyz_cam.T - t[:, np.newaxis])).T
            
            points.append(xyz_world)
            
            if use_colors:
                frame_colors = frame[valid] / 255.0
                colors.append(frame_colors)
        
        # Concatenate all points
        point_cloud = np.vstack(points)
        color_cloud = np.vstack(colors) if use_colors else None
        
        logger.info(f"✓ Generated point cloud with {len(point_cloud)} points")
        
        self.point_cloud = point_cloud
        return point_cloud, color_cloud

    def filter_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        statistical_nb_neighbors: int = 20,
        statistical_std_ratio: float = 2.0,
        voxel_size: float = 0.01
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Filter point cloud using statistical and spatial filters

        Args:
            points: Input point cloud
            colors: Point colors
            statistical_nb_neighbors: Neighbors for statistical outlier removal
            statistical_std_ratio: Standard deviation ratio
            voxel_size: Voxel size for downsampling

        Returns:
            Filtered point cloud, filtered colors
        """
        logger.info("🧹 Filtering point cloud...")
        
        # Statistical outlier removal
        valid_indices = self._statistical_outlier_removal(
            points,
            nb_neighbors=statistical_nb_neighbors,
            std_ratio=statistical_std_ratio
        )
        
        filtered_points = points[valid_indices]
        filtered_colors = colors[valid_indices] if colors is not None else None
        
        num_removed = len(points) - len(filtered_points)
        logger.info(f"   Removed {num_removed} statistical outliers")
        
        # Voxel downsampling
        filtered_points, keep_indices = self._voxel_downsample(
            filtered_points,
            voxel_size=voxel_size
        )
        
        if filtered_colors is not None:
            filtered_colors = filtered_colors[keep_indices]
        
        logger.info(f"   Downsampled to {len(filtered_points)} points")
        
        return filtered_points, filtered_colors

    @staticmethod
    def _statistical_outlier_removal(
        points: np.ndarray,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> np.ndarray:
        """Remove statistical outliers"""
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=nb_neighbors+1)
        distances = distances[:, 1:]  # Exclude self
        
        mean_dist = np.mean(distances, axis=1)
        std_dist = np.std(distances, axis=1)
        
        threshold = mean_dist + std_ratio * std_dist
        valid = np.mean(distances, axis=1) < threshold
        
        return np.where(valid)[0]

    @staticmethod
    def _voxel_downsample(
        points: np.ndarray,
        voxel_size: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample point cloud using voxel grid"""
        points_normalized = np.floor(points / voxel_size).astype(int)
        
        _, unique_indices = np.unique(
            points_normalized,
            axis=0,
            return_index=True
        )
        
        return points[unique_indices], unique_indices

    def estimate_normals(
        self,
        points: np.ndarray,
        k_neighbors: int = 30
    ) -> np.ndarray:
        """
        Estimate surface normals for point cloud

        Args:
            points: Point cloud
            k_neighbors: Number of neighbors for estimation

        Returns:
            Normal vectors
        """
        logger.info("📐 Estimating surface normals...")
        
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points)
        _, indices = tree.query(points, k=k_neighbors+1)
        
        normals = []
        
        for idx_list in tqdm(indices, desc="Normal estimation"):
            neighbors = points[idx_list]
            
            # Compute covariance
            center = neighbors.mean(axis=0)
            neighbors_centered = neighbors - center
            cov = neighbors_centered.T @ neighbors_centered
            
            # Get smallest eigenvector (normal)
            _, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            
            normals.append(normal)
        
        return np.array(normals)

    def reconstruct_mesh_poisson(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        depth: int = 9,
        samples_per_node: float = 1.5
    ) -> str:
        """
        Reconstruct mesh using Poisson surface reconstruction

        Args:
            points: Input point cloud
            colors: Point colors
            depth: Octree depth
            samples_per_node: Samples per node

        Returns:
            Path to output mesh
        """
        logger.info("🔨 Reconstructing surface using Poisson method...")
        
        # Save point cloud to PLY
        temp_ply = self.output_dir / "temp_points.ply"
        self._write_ply_with_normals(temp_ply, points, colors)
        
        # Run Poisson reconstruction if available
        try:
            output_mesh = self.output_dir / "mesh_poisson.ply"
            cmd = [
                "PoissonRecon",
                f"--in={temp_ply}",
                f"--out={output_mesh}",
                f"--depth={depth}",
                f"--samplesPerNode={samples_per_node}"
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✓ Mesh reconstructed: {output_mesh}")
            return str(output_mesh)
        
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("PoissonRecon not available, skipping mesh reconstruction")
            return str(temp_ply)

    @staticmethod
    def _write_ply_with_normals(
        path: Path,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None
    ):
        """Write PLY file with optional colors and normals"""
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if normals is not None:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
            
            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            for i in range(len(points)):
                p = points[i]
                f.write(f"{p[0]} {p[1]} {p[2]}")
                
                if normals is not None:
                    n = normals[i]
                    f.write(f" {n[0]} {n[1]} {n[2]}")
                
                if colors is not None:
                    c = colors[i]
                    f.write(f" {int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}")
                
                f.write("\n")

    def run_full_pipeline(
        self,
        use_poisson: bool = False
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Run complete dense reconstruction pipeline

        Args:
            use_poisson: Attempt Poisson mesh reconstruction

        Returns:
            Point cloud, mesh path (if successful)
        """
        # Create depth maps
        depth_maps = self.create_depth_maps()
        
        # Convert to point cloud
        points, colors = self.depth_to_point_cloud(depth_maps)
        
        # Filter
        points, colors = self.filter_point_cloud(points, colors)
        
        # Estimate normals
        normals = self.estimate_normals(points)
        
        # Save point cloud
        output_ply = self.output_dir / "dense_reconstruction.ply"
        self._write_ply_with_normals(output_ply, points, colors, normals)
        logger.info(f"💾 Saved point cloud to {output_ply}")
        
        # Mesh reconstruction
        mesh_path = None
        if use_poisson:
            mesh_path = self.reconstruct_mesh_poisson(points, colors)
        
        self.point_cloud = points
        return points, mesh_path


class OpenMVSWrapper:
    """Wrapper for OpenMVS dense reconstruction"""

    def __init__(self, output_dir: str = "./mvs_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_openmvs(
        self,
        scene_mvs: str,
        resolution_level: int = 1,
        max_views: int = 10
    ) -> Optional[str]:
        """
        Run OpenMVS reconstruction

        Args:
            scene_mvs: Path to scene MVS file
            resolution_level: Resolution level (1=full resolution)
            max_views: Maximum views per depth map

        Returns:
            Path to output dense point cloud
        """
        logger.info("🚀 Running OpenMVS reconstruction...")
        
        # Densify
        output_dense = self.output_dir / "scene_dense.mvs"
        cmd = [
            "DensifyPointCloud",
            f"--input-file={scene_mvs}",
            f"--output-file={output_dense}",
            f"--resolution-level={resolution_level}",
            f"--max-views={max_views}"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
            logger.info("✓ Densification completed")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.error(f"DensifyPointCloud failed: {e}")
            return None
        
        # Reconstruct mesh
        output_mesh = self.output_dir / "scene.ply"
        cmd = [
            "ReconstructMesh",
            f"--input-file={output_dense}",
            f"--output-file={output_mesh}"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
            logger.info(f"✓ Mesh reconstructed: {output_mesh}")
            return str(output_mesh)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.error(f"ReconstructMesh failed: {e}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    frames = []  # Load your frames
    poses = np.eye(4)[np.newaxis].repeat(len(frames), axis=0)  # Dummy poses
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    
    mvs = DenseReconstruction(frames, poses, K)
    points, colors = mvs.run_full_pipeline(use_poisson=False)
