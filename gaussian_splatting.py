"""
Differentiable Rendering for 3D Gaussian Splatting (3DGS)
Implements adaptive Gaussian optimization with PyTorch
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class GaussianSplatting(nn.Module):
    """3D Gaussian Splatting model for view synthesis"""

    def __init__(
        self,
        point_cloud: np.ndarray,
        num_views: int = 10,
        image_size: Tuple[int, int] = (512, 512),
        learning_rate: float = 0.0016,
        sh_degree: int = 3,
        device: Optional[str] = None,
    ):
        """
        Initialize 3D Gaussian Splatting

        Args:
            point_cloud: Initial point cloud (N, 3)
            num_views: Number of training views
            image_size: Training image size
            learning_rate: Initial learning rate
            sh_degree: Spherical Harmonics degree (0-3)
            device: Device to use
        """
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_views = num_views
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.sh_degree = sh_degree
        
        # Initialize Gaussians from point cloud
        self._initialize_gaussians(point_cloud)
        
        # Optimizer
        self.optimizer = optim.Adam(
            [
                {"params": [self.means], "lr": learning_rate},
                {"params": [self.log_scales], "lr": learning_rate * 10},
                {"params": [self.quats], "lr": learning_rate},
                {"params": [self.shs], "lr": learning_rate}
            ],
            eps=1e-15
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.9999
        )
        
        logger.info(f"🚀 Initialized {self.num_gaussians} Gaussians on {self.device}")

    def _initialize_gaussians(self, point_cloud: np.ndarray):
        """Initialize Gaussian parameters from point cloud"""
        num_points = len(point_cloud)
        self.num_gaussians = num_points
        
        # Position (means)
        means = torch.from_numpy(point_cloud).float().to(self.device)
        self.means = nn.Parameter(means)
        
        # Scale (log-space for stability)
        scales = np.ones((num_points, 3)) * 0.1
        log_scales = torch.from_numpy(np.log(scales)).float().to(self.device)
        self.log_scales = nn.Parameter(log_scales)
        
        # Rotation (unit quaternions)
        quats = np.tile([1, 0, 0, 0], (num_points, 1))
        quats = torch.from_numpy(quats).float().to(self.device)
        self.quats = nn.Parameter(quats / torch.norm(quats, dim=1, keepdim=True))
        
        # Colors (Spherical Harmonics)
        num_sh_coeffs = (self.sh_degree + 1) ** 2
        shs = np.random.randn(num_points, 3, num_sh_coeffs) * 0.1
        shs = torch.from_numpy(shs).float().to(self.device)
        self.shs = nn.Parameter(shs)
        
        # Opacity
        opcities = np.ones((num_points, 1)) * 0.5
        opcities = torch.from_numpy(opcities).float().to(self.device)
        self.opcities = nn.Parameter(opcities)
        
        # Tracking for adaptive strategy
        self.num_new = 0
        self.num_pruned = 0

    def forward(
        self,
        poses: np.ndarray,
        intrinsics: np.ndarray,
    ) -> List[torch.Tensor]:
        """
        Render multiple views

        Args:
            poses: Camera poses (num_views, 4, 4)
            intrinsics: Camera intrinsics (3, 3)

        Returns:
            Rendered images
        """
        renders = []
        
        for i in range(min(self.num_views, len(poses))):
            pose = torch.from_numpy(poses[i]).float().to(self.device)
            K = torch.from_numpy(intrinsics).float().to(self.device)
            
            # Project Gaussians to image space
            render = self._render_view(pose, K)
            renders.append(render)
        
        return renders

    def _render_view(
        self,
        pose: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """Render single view"""
        h, w = self.image_size
        
        # Transform points to camera space
        R = pose[:3, :3]
        t = pose[:3, 3:4]
        
        points_cam = (R @ self.means.T + t).T  # (N, 3)
        
        # Project to image
        points_proj = (intrinsics @ points_cam.T).T
        uv = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-8)
        
        # Initialize render
        image = torch.zeros(h, w, 3, device=self.device)
        
        # Get scales
        scales = torch.exp(self.log_scales)
        
        # Render each Gaussian
        for i in range(self.num_gaussians):
            # Skip if behind camera
            if points_cam[i, 2] < 0.1:
                continue
            
            # Get Gaussian position in image
            u, v = int(uv[i, 0]), int(uv[i, 1])
            
            # Check bounds
            if not (0 <= u < w and 0 <= v < h):
                continue
            
            # Compute covariance in image space
            scale = scales[i]
            quat = self.quats[i]
            
            # Rotation matrix from quaternion
            R_gauss = self._quat_to_mat(quat)
            
            # Covariance
            cov = R_gauss @ torch.diag(scale) @ R_gauss.T
            
            # Gaussian splat
            y_coords, x_coords = torch.meshgrid(
                torch.arange(max(0, v - 20), min(h, v + 20), device=self.device),
                torch.arange(max(0, u - 20), min(w, u + 20), device=self.device),
                indexing='ij'
            )
            
            pos = torch.stack([x_coords, y_coords], dim=-1).float() - uv[i]
            
            # Gaussian kernel
            try:
                cov_inv = torch.inverse(cov[:2, :2])
                exponent = -(pos @ cov_inv * pos).sum(dim=-1) / 2
                gaussian = torch.exp(torch.clamp(exponent, -10, 0))
            except:
                continue
            
            # Get color
            color = self.shs[i, :, 0]  # Use first SH coefficient
            color = torch.sigmoid(color)
            
            # Opacity
            opacity = torch.sigmoid(self.opcities[i])
            
            # Splat
            yi = y_coords - max(0, v - 20)
            xi = x_coords - max(0, u - 20)
            
            for yy, xx, g in zip(yi.flat, xi.flat, gaussian.flat):
                if g > 1e-3:
                    yy_idx = yy.long() + max(0, v - 20)
                    xx_idx = xx.long() + max(0, u - 20)
                    if 0 <= yy_idx < h and 0 <= xx_idx < w:
                        image[yy_idx, xx_idx] += g * opacity * color
        
        return image

    @staticmethod
    def _quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        mat = torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ], device=quat.device, dtype=quat.dtype)
        
        return mat

    def optimize(
        self,
        images: List[np.ndarray],
        poses: np.ndarray,
        intrinsics: np.ndarray,
        iterations: int = 7000,
        densification_interval: int = 100,
        pruning_interval: int = 500,
        pruning_threshold: float = 0.005,
    ):
        """
        Optimize Gaussians

        Args:
            images: Training images
            poses: Camera poses
            intrinsics: Camera intrinsics
            iterations: Training iterations
            densification_interval: Add/remove Gaussians every N iterations
            pruning_interval: Prune low opacity Gaussians every N iterations
            pruning_threshold: Opacity threshold for pruning
        """
        logger.info(f"🎯 Optimizing {self.num_gaussians} Gaussians for {iterations} iterations")
        
        # Convert to tensors
        images_t = [torch.from_numpy(img).float().to(self.device) for img in images]
        poses_t = torch.from_numpy(poses).float().to(self.device)
        K_t = torch.from_numpy(intrinsics).float().to(self.device)
        
        pbar = tqdm(total=iterations, desc="Optimizing")
        
        for iteration in range(iterations):
            # Render
            renders = self.forward(poses, intrinsics)
            
            # Loss computation
            loss = 0.0
            for i, render in enumerate(renders):
                target = images_t[i % len(images_t)]
                if render.shape != target.shape:
                    continue
                
                # L1 + L2 loss
                l1_loss = torch.abs(render - target).mean()
                l2_loss = torch.pow(render - target, 2).mean()
                loss += 0.8 * l1_loss + 0.2 * l2_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Adaptive densification
            if iteration % densification_interval == 0 and iteration > 500:
                self._adaptively_add_gaussians()
            
            # Pruning
            if iteration % pruning_interval == 0 and iteration > 1000:
                self._prune_gaussians(pruning_threshold)
            
            # Learning rate scheduling
            if iteration % 1000 == 0:
                self.scheduler.step()
            
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "gaussians": self.num_gaussians})
        
        pbar.close()
        logger.info(f"✓ Optimization complete. Final Gaussians: {self.num_gaussians}")

    def _adaptively_add_gaussians(self):
        """Add new Gaussians in high-gradient regions"""
        # Simplified version - add copy of high-gradient Gaussians
        if len(self.means.grad) == 0:
            return
        
        grad_norm = torch.norm(self.means.grad, dim=1)
        top_indices = torch.topk(grad_norm, min(100, len(grad_norm)//10)).indices
        
        for idx in top_indices:
            # Add perturbed copy
            new_mean = self.means[idx] + torch.randn(3, device=self.device) * 0.01
            self.means.data = torch.cat([self.means.data, new_mean.unsqueeze(0)])
            
            self.log_scales.data = torch.cat([
                self.log_scales.data,
                self.log_scales[idx].unsqueeze(0) + torch.randn(3, device=self.device) * 0.1
            ])
            
            self.quats.data = torch.cat([
                self.quats.data,
                self.quats[idx].unsqueeze(0)
            ])
            
            self.shs.data = torch.cat([
                self.shs.data,
                self.shs[idx].unsqueeze(0)
            ])
            
            self.opcities.data = torch.cat([
                self.opcities.data,
                self.opcities[idx].unsqueeze(0)
            ])
        
        self.num_gaussians = len(self.means)
        self.num_new += len(top_indices)

    def _prune_gaussians(self, threshold: float = 0.005):
        """Remove low-opacity Gaussians"""
        opcities_np = torch.sigmoid(self.opcities).squeeze(-1).cpu().detach().numpy()
        valid_mask = opcities_np > threshold
        
        num_pruned = (~valid_mask).sum()
        
        if num_pruned > 0:
            self.means.data = self.means.data[valid_mask]
            self.log_scales.data = self.log_scales.data[valid_mask]
            self.quats.data = self.quats.data[valid_mask]
            self.shs.data = self.shs.data[valid_mask]
            self.opcities.data = self.opcities.data[valid_mask]
            
            self.num_gaussians = len(self.means)
            self.num_pruned += num_pruned
            
            logger.info(f"   Pruned {num_pruned} Gaussians (opacity < {threshold})")

    def export_ply(self, output_path: str):
        """Export as PLY point cloud"""
        logger.info(f"💾 Exporting to {output_path}")
        
        means = self.means.cpu().detach().numpy()
        colors = torch.sigmoid(self.shs[:, :, 0]).cpu().detach().numpy() * 255
        opcities = torch.sigmoid(self.opcities).cpu().detach().numpy() * 255
        
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(means)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property uchar alpha\n")
            f.write("end_header\n")
            
            for i in range(len(means)):
                x, y, z = means[i]
                r, g, b = colors[i]
                a = opcities[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)} {int(a)}\n")
        
        logger.info(f"✓ Exported {len(means)} Gaussians")

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            "means": self.means.cpu().detach().numpy(),
            "log_scales": self.log_scales.cpu().detach().numpy(),
            "quats": self.quats.cpu().detach().numpy(),
            "shs": self.shs.cpu().detach().numpy(),
            "opcities": self.opcities.cpu().detach().numpy(),
            "num_gaussians": self.num_gaussians,
        }
        np.savez(path, **checkpoint)
        logger.info(f"✓ Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        data = np.load(path)
        
        means = torch.from_numpy(data["means"]).float().to(self.device)
        self.means = nn.Parameter(means)
        
        self.log_scales = nn.Parameter(
            torch.from_numpy(data["log_scales"]).float().to(self.device)
        )
        self.quats = nn.Parameter(
            torch.from_numpy(data["quats"]).float().to(self.device)
        )
        self.shs = nn.Parameter(
            torch.from_numpy(data["shs"]).float().to(self.device)
        )
        self.opcities = nn.Parameter(
            torch.from_numpy(data["opcities"]).float().to(self.device)
        )
        self.num_gaussians = int(data["num_gaussians"])
        
        logger.info(f"✓ Loaded checkpoint with {self.num_gaussians} Gaussians")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    points = np.random.randn(1000, 3)
    gs = GaussianSplatting(points, num_views=10)
    
    # Optimize (requires images and poses)
    # gs.optimize(images, poses, intrinsics, iterations=7000)
    
    # Export
    gs.export_ply("output.ply")
