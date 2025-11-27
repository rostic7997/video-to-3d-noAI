"""
COLMAP Integration for Structure-from-Motion (SfM) Reconstruction
Handles camera calibration, feature detection, matching, and 3D reconstruction
"""

import os
import json
import logging
import subprocess
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)


class COLMAPPipeline:
    """COLMAP-based 3D reconstruction pipeline"""

    def __init__(
        self,
        video_path: str,
        output_dir: str = "./reconstruction",
        max_frames: int = 300,
        skip_frames: int = 2,
        resize_width: int = 1280,
        quality_check: bool = True,
        colmap_config: Optional[Dict] = None,
    ):
        """
        Initialize COLMAP pipeline

        Args:
            video_path: Path to input video
            output_dir: Output directory for reconstruction
            max_frames: Maximum frames to extract
            skip_frames: Extract every N-th frame
            resize_width: Resize width for frames
            quality_check: Check frame quality (blur, motion)
            colmap_config: Custom COLMAP configuration
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames_dir = self.output_dir / "frames"
        self.database_path = self.output_dir / "database.db"
        self.sparse_dir = self.output_dir / "sparse"
        self.dense_dir = self.output_dir / "dense"
        
        self.frames_dir.mkdir(exist_ok=True)
        self.sparse_dir.mkdir(exist_ok=True)
        self.dense_dir.mkdir(exist_ok=True)
        
        self.max_frames = max_frames
        self.skip_frames = skip_frames
        self.resize_width = resize_width
        self.quality_check = quality_check
        
        self.colmap_config = colmap_config or self._default_config()
        
        self.frames = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_poses = []
        self.point_cloud = None
        
        logger.info(f"Initialized COLMAP pipeline at {self.output_dir}")

    def _default_config(self) -> Dict:
        """Default COLMAP configuration"""
        return {
            "matcher_type": "sequential",  # sequential, exhaustive, spatial
            "triangulation_type": "auto",
            "mapper_min_track_length": 2,
            "mapper_min_num_matches": 15,
            "mapper_ba_refine_focal_length": True,
            "mapper_ba_refine_principal_point": True,
            "mapper_ba_refine_extra_params": True,
        }

    def extract_frames(self) -> List[np.ndarray]:
        """Extract frames from video with quality checks"""
        logger.info(f"📹 Extracting frames from {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"   Total frames: {total_frames}, FPS: {fps:.1f}")
        
        frame_count = 0
        extracted = 0
        prev_frame = None
        
        pbar = tqdm(total=self.max_frames, desc="Extracting frames")
        
        while cap.isOpened() and extracted < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.skip_frames != 0:
                frame_count += 1
                continue
            
            # Resize
            h, w = frame.shape[:2]
            if w > self.resize_width:
                ratio = self.resize_width / w
                frame = cv2.resize(frame, (self.resize_width, int(h * ratio)))
            
            # Quality check
            if self.quality_check:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if laplacian_var < 50:  # Too blurry
                    frame_count += 1
                    continue
                
                if prev_frame is not None:
                    diff = cv2.absdiff(frame, prev_frame)
                    motion = np.mean(diff)
                    
                    if motion < 5:  # Too similar
                        frame_count += 1
                        continue
            
            # Enhance contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Save frame
            frame_path = self.frames_dir / f"{extracted:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self.frames.append(frame)
            prev_frame = frame.copy()
            extracted += 1
            
            pbar.update(1)
            frame_count += 1
        
        cap.release()
        pbar.close()
        
        logger.info(f"✓ Extracted {len(self.frames)} frames")
        return self.frames

    def calibrate_camera(self) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate camera intrinsics from first frame"""
        if len(self.frames) == 0:
            raise ValueError("No frames extracted")
        
        frame = self.frames[0]
        h, w = frame.shape[:2]
        
        # Estimate focal length (typical for smartphone/camera)
        focal_length = max(w, h) * 1.2
        center = (w / 2, h / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=float)
        
        self.dist_coeffs = np.zeros((5, 1))
        
        logger.info(f"📷 Camera calibration:")
        logger.info(f"   Focal length: {focal_length:.1f}")
        logger.info(f"   Principal point: {center}")
        
        return self.camera_matrix, self.dist_coeffs

    def create_colmap_database(self):
        """Create COLMAP database from extracted frames"""
        logger.info("📊 Creating COLMAP database...")
        
        # Remove existing database
        if self.database_path.exists():
            self.database_path.unlink()
        
        # Create camera model
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.frames_dir),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1",
            "--SiftExtraction.max_image_size", "2048"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
            logger.info("✓ Features extracted")
        except subprocess.CalledProcessError as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
        
        # Feature matching
        cmd = [
            "colmap", f"{self.colmap_config['matcher_type']}_matcher",
            "--database_path", str(self.database_path),
            "--SiftMatching.use_gpu", "1"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
            logger.info("✓ Features matched")
        except subprocess.CalledProcessError as e:
            logger.error(f"Feature matching failed: {e}")
            raise

    def run_incremental_mapper(self):
        """Run COLMAP incremental mapper for SfM"""
        logger.info("🗺️  Running COLMAP incremental mapper...")
        
        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.frames_dir),
            "--output_path", str(self.sparse_dir),
            "--Mapper.num_threads", "4",
            "--Mapper.min_model_type", "2",
            "--Mapper.min_num_matches", str(self.colmap_config["mapper_min_num_matches"]),
            "--Mapper.init_min_tri_angle", "4",
            "--Mapper.multiple_models", "0",
            "--Mapper.extract_colors", "1"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
            logger.info("✓ SfM reconstruction completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Mapper failed: {e}")
            raise

    def load_reconstruction(self) -> Dict:
        """Load COLMAP reconstruction"""
        logger.info("📂 Loading COLMAP reconstruction...")
        
        sparse_model_dir = list(self.sparse_dir.glob("*"))[0]
        
        # Parse camera.bin
        cameras = self._read_cameras_binary(sparse_model_dir / "cameras.bin")
        
        # Parse images.bin
        images = self._read_images_binary(sparse_model_dir / "images.bin")
        
        # Parse points3D.bin
        points3d = self._read_points3d_binary(sparse_model_dir / "points3D.bin")
        
        logger.info(f"   Cameras: {len(cameras)}")
        logger.info(f"   Images: {len(images)}")
        logger.info(f"   3D Points: {len(points3d)}")
        
        return {
            "cameras": cameras,
            "images": images,
            "points3d": points3d
        }

    @staticmethod
    def _read_cameras_binary(path) -> Dict:
        """Read COLMAP cameras.bin"""
        cameras = {}
        with open(path, "rb") as fid:
            num_cameras = np.frombuffer(fid.read(8), dtype=np.uint64)[0]
            for _ in range(num_cameras):
                camera_id = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
                model_id = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
                width = np.frombuffer(fid.read(8), dtype=np.uint64)[0]
                height = np.frombuffer(fid.read(8), dtype=np.uint64)[0]
                params = np.frombuffer(fid.read(24), dtype=np.float64)
                
                cameras[camera_id] = {
                    "model_id": model_id,
                    "width": width,
                    "height": height,
                    "params": params
                }
        return cameras

    @staticmethod
    def _read_images_binary(path) -> Dict:
        """Read COLMAP images.bin"""
        images = {}
        with open(path, "rb") as fid:
            num_images = np.frombuffer(fid.read(8), dtype=np.uint64)[0]
            for _ in range(num_images):
                image_id = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
                qvec = np.frombuffer(fid.read(32), dtype=np.float64)
                tvec = np.frombuffer(fid.read(24), dtype=np.float64)
                
                camera_id = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
                name_len = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
                name = fid.read(name_len).decode("utf-8")
                
                num_points2d = np.frombuffer(fid.read(8), dtype=np.uint64)[0]
                points2d = np.frombuffer(fid.read(num_points2d * 16), dtype=np.float64)
                points2d = points2d.reshape(-1, 2)
                point3d_ids = np.frombuffer(fid.read(num_points2d * 8), dtype=np.int64)
                
                images[image_id] = {
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name,
                    "points2d": points2d,
                    "point3d_ids": point3d_ids
                }
        return images

    @staticmethod
    def _read_points3d_binary(path) -> Dict:
        """Read COLMAP points3D.bin"""
        points3d = {}
        with open(path, "rb") as fid:
            num_points = np.frombuffer(fid.read(8), dtype=np.uint64)[0]
            for _ in range(num_points):
                point_id = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
                xyz = np.frombuffer(fid.read(24), dtype=np.float64)
                rgb = np.frombuffer(fid.read(3), dtype=np.uint8)
                
                num_tracks = np.frombuffer(fid.read(8), dtype=np.uint64)[0]
                track_data = np.frombuffer(fid.read(num_tracks * 8), dtype=np.uint32)
                
                points3d[point_id] = {
                    "xyz": xyz,
                    "rgb": rgb,
                    "num_tracks": num_tracks
                }
        return points3d

    def export_sparse_reconstruction(self, output_ply: str):
        """Export sparse reconstruction as PLY"""
        logger.info(f"💾 Exporting sparse reconstruction to {output_ply}")
        
        reconstruction = self.load_reconstruction()
        points3d = reconstruction["points3d"]
        
        points = []
        colors = []
        
        for point_id, point in points3d.items():
            points.append(point["xyz"])
            colors.append(point["rgb"])
        
        points = np.array(points)
        colors = np.array(colors)
        
        # Write PLY
        self._write_ply(output_ply, points, colors)
        logger.info(f"✓ Exported {len(points)} points")

    @staticmethod
    def _write_ply(path: str, points: np.ndarray, colors: np.ndarray):
        """Write PLY file"""
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i in range(len(points)):
                p = points[i]
                c = colors[i]
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    def run_colmap_sfm(self) -> Tuple[List[np.ndarray], List[Dict], np.ndarray]:
        """
        Run complete COLMAP SfM pipeline

        Returns:
            frames: List of extracted frames
            poses: List of camera poses [R, t]
            intrinsics: Camera intrinsic matrix
        """
        # Extract frames
        self.extract_frames()
        
        # Calibrate camera
        self.calibrate_camera()
        
        # Create COLMAP database
        self.create_colmap_database()
        
        # Run mapper
        self.run_incremental_mapper()
        
        # Load reconstruction
        reconstruction = self.load_reconstruction()
        
        return self.frames, reconstruction["images"], self.camera_matrix

    def run_dense_mvs(self):
        """Run COLMAP dense reconstruction"""
        logger.info("🔍 Running COLMAP dense reconstruction...")
        
        sparse_model = list(self.sparse_dir.glob("*"))[0]
        
        # Undistortion
        cmd = [
            "colmap", "image_undistorter",
            "--image_path", str(self.frames_dir),
            "--input_path", str(sparse_model),
            "--output_path", str(self.dense_dir),
            "--output_type", "COLMAP"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
            logger.info("✓ Images undistorted")
        except subprocess.CalledProcessError as e:
            logger.error(f"Undistortion failed: {e}")
            return None
        
        # Stereo matching
        cmd = [
            "colmap", "patch_match_stereo",
            "--workspace_path", str(self.dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "1"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
            logger.info("✓ Stereo matching completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Stereo matching failed: {e}")
            return None
        
        # Fusion
        cmd = [
            "colmap", "stereo_fusion",
            "--workspace_path", str(self.dense_dir),
            "--workspace_format", "COLMAP",
            "--output_path", str(self.dense_dir / "fused.ply")
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
            logger.info("✓ Point cloud fusion completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Fusion failed: {e}")
            return None
        
        return self.dense_dir / "fused.ply"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    pipeline = COLMAPPipeline(
        video_path="video.mp4",
        output_dir="./reconstruction",
        max_frames=300,
        skip_frames=2
    )
    
    frames, images, K = pipeline.run_colmap_sfm()
    pipeline.export_sparse_reconstruction("sparse.ply")
    
    ply_path = pipeline.run_dense_mvs()
    if ply_path:
        print(f"Dense reconstruction saved to {ply_path}")
