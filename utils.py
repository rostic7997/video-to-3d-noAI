"""
Utility functions for 3D reconstruction pipeline
"""

import numpy as np
import cv2
import torch
import logging
from typing import Tuple, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CameraUtils:
    """Camera-related utility functions"""

    @staticmethod
    def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        q = quat / np.linalg.norm(quat)
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        return R

    @staticmethod
    def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])

    @staticmethod
    def estimate_focal_length(image_shape: Tuple[int, int]) -> float:
        """Estimate focal length from image shape"""
        h, w = image_shape
        return max(w, h) * 1.2

    @staticmethod
    def project_points(
        points_3d: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """Project 3D points to image plane"""
        # Transform to camera coordinates
        points_cam = R @ points_3d.T + t.reshape(3, 1)
        
        # Project
        points_proj = K @ points_cam
        points_2d = points_proj[:2] / points_proj[2]
        
        return points_2d.T

    @staticmethod
    def triangulate_points(
        K: np.ndarray,
        R1: np.ndarray,
        t1: np.ndarray,
        R2: np.ndarray,
        t2: np.ndarray,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> np.ndarray:
        """Triangulate 3D points from two views"""
        # Create projection matrices
        P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
        
        # Normalize coordinates
        points1_norm = np.linalg.inv(K) @ np.vstack([points1.T, np.ones(len(points1))])[:2]
        points2_norm = np.linalg.inv(K) @ np.vstack([points2.T, np.ones(len(points2))])[:2]
        
        points_3d = []
        for i in range(len(points1)):
            # Build linear system
            A = np.array([
                points1_norm[0, i] * P1[2] - P1[0],
                points1_norm[1, i] * P1[2] - P1[1],
                points2_norm[0, i] * P2[2] - P2[0],
                points2_norm[1, i] * P2[2] - P2[1]
            ])
            
            # Solve with SVD
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            
            # Normalize
            X = X / X[3]
            points_3d.append(X[:3])
        
        return np.array(points_3d)


class PointCloudUtils:
    """Point cloud processing utilities"""

    @staticmethod
    def remove_outliers_statistical(
        points: np.ndarray,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> np.ndarray:
        """Remove statistical outliers"""
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=nb_neighbors+1)
        distances = distances[:, 1:]
        
        mean_dist = np.mean(distances, axis=1)
        std_dist = np.std(distances, axis=1)
        
        threshold = mean_dist + std_ratio * std_dist
        valid = np.mean(distances, axis=1) < threshold
        
        return points[valid]

    @staticmethod
    def voxel_downsample(
        points: np.ndarray,
        voxel_size: float = 0.01
    ) -> np.ndarray:
        """Downsample point cloud"""
        points_normalized = np.floor(points / voxel_size).astype(int)
        _, indices = np.unique(points_normalized, axis=0, return_index=True)
        return points[indices]

    @staticmethod
    def estimate_normals(
        points: np.ndarray,
        k_neighbors: int = 30
    ) -> np.ndarray:
        """Estimate surface normals"""
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points)
        _, indices = tree.query(points, k=k_neighbors+1)
        
        normals = []
        for idx_list in indices:
            neighbors = points[idx_list]
            center = neighbors.mean(axis=0)
            neighbors_centered = neighbors - center
            cov = neighbors_centered.T @ neighbors_centered
            _, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            normals.append(normal)
        
        return np.array(normals)

    @staticmethod
    def compute_point_colors(
        points: np.ndarray,
        frames: List[np.ndarray],
        poses: np.ndarray,
        K: np.ndarray
    ) -> np.ndarray:
        """Sample colors from frames for 3D points"""
        colors = np.zeros((len(points), 3))
        counts = np.zeros(len(points))
        
        for frame_idx, frame in enumerate(frames):
            pose = poses[frame_idx]
            R, t = pose[:3, :3], pose[:3, 3]
            
            # Project points
            points_cam = R @ points.T + t.reshape(3, 1)
            points_proj = K @ points_cam
            points_2d = points_proj[:2] / points_proj[2]
            
            h, w = frame.shape[:2]
            for i, (x, y) in enumerate(points_2d.T):
                x, y = int(x), int(y)
                if 0 <= x < w and 0 <= y < h:
                    colors[i] += frame[y, x]
                    counts[i] += 1
        
        # Average colors
        valid = counts > 0
        colors[valid] /= counts[valid, np.newaxis]
        
        return colors / 255.0


class ImageUtils:
    """Image processing utilities"""

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def compute_image_sharpness(image: np.ndarray) -> float:
        """Compute image sharpness using Laplacian"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def compute_image_motion(image1: np.ndarray, image2: np.ndarray) -> float:
        """Compute optical flow magnitude"""
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        return np.mean(magnitude)

    @staticmethod
    def create_grid_image(images: List[np.ndarray], rows: int = 2) -> np.ndarray:
        """Create grid of images for visualization"""
        h, w = images[0].shape[:2]
        cols = (len(images) + rows - 1) // rows
        
        grid = np.zeros((h*rows, w*cols, 3), dtype=images[0].dtype)
        
        for i, img in enumerate(images):
            r = i // cols
            c = i % cols
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
        
        return grid


class TorchUtils:
    """PyTorch utility functions"""

    @staticmethod
    def numpy_to_torch(arr: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Convert numpy array to torch tensor"""
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        return torch.from_numpy(arr).to(device)

    @staticmethod
    def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array"""
        return tensor.cpu().detach().numpy()

    @staticmethod
    def get_device() -> str:
        """Get available device"""
        return "cuda" if torch.cuda.is_available() else "cpu"


class FileUtils:
    """File I/O utilities"""

    @staticmethod
    def write_ply(
        path: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None
    ):
        """Write PLY file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
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

    @staticmethod
    def read_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Read PLY file"""
        points = []
        colors = []
        has_color = False
        
        with open(path, 'r') as f:
            # Parse header
            while True:
                line = f.readline().strip()
                if line.startswith("property uchar red"):
                    has_color = True
                if line == "end_header":
                    break
            
            # Parse data
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                points.append([x, y, z])
                
                if has_color and len(parts) >= 6:
                    r = int(parts[3]) / 255.0
                    g = int(parts[4]) / 255.0
                    b = int(parts[5]) / 255.0
                    colors.append([r, g, b])
        
        return np.array(points), np.array(colors) if has_color else None
