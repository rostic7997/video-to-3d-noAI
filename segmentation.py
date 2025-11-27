"""
Segment Anything Integration for automatic foreground/background segmentation
Improves reconstruction quality by masking irrelevant regions
"""

import logging
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SegmentAnything:
    """Segment Anything model wrapper for video segmentation"""

    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Segment Anything

        Args:
            model_size: Model size (tiny, small, base, large)
            device: Device to use (cuda, cpu). Auto-detects if None
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🤖 Loading Segment Anything ({model_size}) on {self.device}")
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            model_name = f"vit_{model_size}"
            self.sam = sam_model_registry[model_name](pretrained=True)
            self.sam = self.sam.to(self.device)
            
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
            logger.info("✓ Segment Anything loaded")
            
        except ImportError as e:
            logger.error(f"Failed to import Segment Anything: {e}")
            logger.warning("Install with: pip install segment-anything")
            self.mask_generator = None

    def segment_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Segment a single frame

        Args:
            frame: Input frame (BGR)

        Returns:
            mask: Binary mask (foreground)
            masks: List of detected masks with metadata
        """
        if self.mask_generator is None:
            logger.warning("Segment Anything not loaded, returning empty mask")
            return np.ones_like(frame[:, :, 0], dtype=np.uint8), []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        masks = self.mask_generator.generate(rgb_frame)
        
        # Create combined foreground mask
        h, w = frame.shape[:2]
        foreground_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Sort masks by area (largest first)
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Use largest masks for foreground
        for i, mask_info in enumerate(sorted_masks[:5]):
            mask = mask_info['segmentation']
            foreground_mask[mask] = 255
        
        return foreground_mask, masks

    def segment_video(
        self,
        frames: List[np.ndarray],
        return_masks: bool = False,
        iou_threshold: float = 0.5
    ) -> List[np.ndarray]:
        """
        Segment video frames

        Args:
            frames: List of input frames
            return_masks: Return detailed mask info
            iou_threshold: IoU threshold for mask filtering

        Returns:
            List of foreground masks
        """
        logger.info(f"🎬 Segmenting {len(frames)} frames with Segment Anything...")
        
        masks = []
        detailed_masks = []
        
        for frame in tqdm(frames, desc="Segmenting"):
            fg_mask, mask_info = self.segment_frame(frame)
            masks.append(fg_mask)
            detailed_masks.append(mask_info)
        
        logger.info(f"✓ Segmented {len(masks)} frames")
        
        if return_masks:
            return masks, detailed_masks
        return masks

    def refine_point_cloud(
        self,
        points: np.ndarray,
        projections: List[Tuple[int, int, int]],
        masks: List[np.ndarray],
        frames: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter point cloud using segmentation masks

        Args:
            points: 3D point cloud
            projections: List of (frame_idx, x, y) for each point
            masks: List of segmentation masks
            frames: Input frames

        Returns:
            filtered_points: Points inside foreground mask
            valid_indices: Indices of kept points
        """
        logger.info("🔍 Filtering point cloud with segmentation masks...")
        
        valid_indices = []
        
        for i, (frame_idx, x, y) in enumerate(projections):
            if frame_idx >= len(masks):
                continue
            
            mask = masks[frame_idx]
            h, w = mask.shape
            
            # Check if point is within frame bounds
            if 0 <= int(x) < w and 0 <= int(y) < h:
                # Check if point is in foreground mask
                if mask[int(y), int(x)] > 0:
                    valid_indices.append(i)
        
        filtered_points = points[valid_indices]
        logger.info(f"   Kept {len(filtered_points)}/{len(points)} points ({len(filtered_points)/len(points)*100:.1f}%)")
        
        return filtered_points, np.array(valid_indices)

    def fill_holes(
        self,
        mask: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Fill holes in segmentation mask

        Args:
            mask: Input binary mask
            kernel_size: Morphological kernel size

        Returns:
            Filled mask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Close operation (dilation then erosion)
        filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Fill holes using flood fill
        h, w = filled.shape
        temp = np.zeros((h + 2, w + 2), dtype=np.uint8)
        temp[1:-1, 1:-1] = filled
        
        cv2.floodFill(temp, None, (0, 0), 0)
        filled = temp[1:-1, 1:-1]
        
        return 255 - filled

    def get_bounding_box(
        self,
        mask: np.ndarray,
        padding: float = 0.05
    ) -> Tuple[int, int, int, int]:
        """
        Get bounding box from segmentation mask

        Args:
            mask: Input binary mask
            padding: Padding ratio (0.0-1.0)

        Returns:
            x_min, y_min, x_max, y_max
        """
        coords = np.argwhere(mask > 0)
        
        if len(coords) == 0:
            h, w = mask.shape
            return 0, 0, w, h
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        h, w = mask.shape
        pad_x = int((x_max - x_min) * padding)
        pad_y = int((y_max - y_min) * padding)
        
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)
        
        return x_min, y_min, x_max, y_max

    def get_camera_frustum_mask(
        self,
        frame_shape: Tuple[int, int],
        camera_pose: np.ndarray,
        intrinsics: np.ndarray,
        point_cloud: np.ndarray,
        depth_range: Tuple[float, float] = (0.1, 1000.0)
    ) -> np.ndarray:
        """
        Create mask of 3D points visible in camera frustum

        Args:
            frame_shape: (height, width)
            camera_pose: Camera extrinsics [R | t]
            intrinsics: Camera intrinsic matrix
            point_cloud: 3D point cloud
            depth_range: Valid depth range

        Returns:
            Visibility mask
        """
        h, w = frame_shape
        R, t = camera_pose[:3, :3], camera_pose[:3, 3:4]
        
        # Project to camera
        points_cam = R @ point_cloud.T + t
        
        # Check depth
        depth_valid = (points_cam[2] > depth_range[0]) & (points_cam[2] < depth_range[1])
        
        # Project to image
        points_proj = intrinsics @ points_cam
        points_2d = points_proj[:2] / points_proj[2]
        
        # Check image bounds
        in_image = (
            (points_2d[0] >= 0) & (points_2d[0] < w) &
            (points_2d[1] >= 0) & (points_2d[1] < h)
        )
        
        visibility = depth_valid & in_image.astype(bool)
        
        return visibility


class VideoSegmentationPipeline:
    """Complete video segmentation and filtering pipeline"""

    def __init__(self, model_size: str = "base"):
        self.sam = SegmentAnything(model_size=model_size)

    def process_video(
        self,
        frames: List[np.ndarray],
        point_cloud: Optional[np.ndarray] = None,
        point_projections: Optional[List] = None
    ) -> dict:
        """
        Process video for segmentation

        Args:
            frames: Input frames
            point_cloud: Optional 3D point cloud to filter
            point_projections: Optional point projections

        Returns:
            Dictionary with masks, filtered point cloud, etc.
        """
        # Segment all frames
        masks = self.sam.segment_video(frames)
        
        # Process results
        results = {
            "masks": masks,
            "filled_masks": [],
            "bboxes": []
        }
        
        for mask in masks:
            # Fill holes
            filled = self.sam.fill_holes(mask)
            results["filled_masks"].append(filled)
            
            # Get bounding box
            bbox = self.sam.get_bounding_box(filled)
            results["bboxes"].append(bbox)
        
        # Filter point cloud if provided
        if point_cloud is not None and point_projections is not None:
            filtered_cloud, valid_idx = self.sam.refine_point_cloud(
                point_cloud,
                point_projections,
                masks,
                frames
            )
            results["filtered_point_cloud"] = filtered_cloud
            results["valid_indices"] = valid_idx
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    sam = SegmentAnything(model_size="base")
    
    # Segment a single frame
    frame = cv2.imread("frame.jpg")
    mask, masks = sam.segment_frame(frame)
    
    print(f"Generated {len(masks)} masks")
    print(f"Foreground mask shape: {mask.shape}")
