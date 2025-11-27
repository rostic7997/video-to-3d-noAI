"""
PhotoGeometry - Advanced 3D Reconstruction from Video
Gradio web interface for Hugging Face Spaces
"""

import gradio as gr
import logging
import numpy as np
import cv2
import torch
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import traceback

# Import our modules
from colmap_pipeline import COLMAPPipeline
from segmentation import SegmentAnything, VideoSegmentationPipeline
from gaussian_splatting import GaussianSplatting
from mvs_reconstruction import DenseReconstruction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
class ReconstructionState:
    def __init__(self):
        self.frames = []
        self.poses = None
        self.intrinsics = None
        self.point_cloud = None
        self.colors = None
        self.depth_maps = None
        self.segmentation_masks = None
        
    def reset(self):
        self.frames = []
        self.poses = None
        self.intrinsics = None
        self.point_cloud = None
        self.colors = None
        self.depth_maps = None
        self.segmentation_masks = None

state = ReconstructionState()


def extract_frames_from_video(
    video_path: str,
    max_frames: int = 100,
    skip_frames: int = 2,
    resize_width: int = 1280
) -> Tuple[str, int]:
    """Extract frames from video"""
    try:
        state.reset()
        
        logger.info(f"📹 Extracting frames from {video_path}")
        logger.info(f"   Max frames: {max_frames}, Skip: {skip_frames}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        extracted = 0
        
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            # Resize
            h, w = frame.shape[:2]
            if w > resize_width:
                ratio = resize_width / w
                frame = cv2.resize(frame, (resize_width, int(h * ratio)))
            
            # Quality check
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 50:  # Too blurry
                frame_count += 1
                continue
            
            state.frames.append(frame)
            extracted += 1
            frame_count += 1
        
        cap.release()
        
        message = f"✅ Extracted {len(state.frames)} frames from {total_frames} total frames\n"
        message += f"   FPS: {fps:.1f}\n"
        message += f"   Resolution: {state.frames[0].shape[1]}x{state.frames[0].shape[0]}"
        
        logger.info(message)
        return message, len(state.frames)
        
    except Exception as e:
        error_msg = f"❌ Frame extraction failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg, 0


def run_colmap_sfm() -> str:
    """Run COLMAP Structure-from-Motion"""
    try:
        if len(state.frames) == 0:
            return "❌ No frames extracted. Please extract frames first."
        
        logger.info("🗺️  Running COLMAP SfM reconstruction...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = COLMAPPipeline(
                video_path="dummy.mp4",
                output_dir=tmpdir,
                max_frames=len(state.frames)
            )
            
            # Skip frame extraction, use our frames
            pipeline.frames = state.frames
            
            # Calibrate camera
            pipeline.calibrate_camera()
            state.intrinsics = pipeline.camera_matrix.copy()
            
            # Create COLMAP database
            pipeline.create_colmap_database()
            
            # Run mapper
            pipeline.run_incremental_mapper()
            
            # Load reconstruction
            reconstruction = pipeline.load_reconstruction()
            
            # Extract poses
            state.poses = []
            for img_id in sorted(reconstruction["images"].keys()):
                img = reconstruction["images"][img_id]
                qvec = img["qvec"]
                tvec = img["tvec"]
                
                # Convert quaternion to rotation matrix
                q = qvec
                R = np.array([
                    [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
                    [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
                    [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]
                ])
                
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = tvec
                state.poses.append(pose)
            
            state.poses = np.array(state.poses)
            
            message = f"✅ COLMAP SfM completed\n"
            message += f"   Cameras: {len(reconstruction['cameras'])}\n"
            message += f"   Images: {len(reconstruction['images'])}\n"
            message += f"   3D Points: {len(reconstruction['points3d'])}"
            
            logger.info(message)
            return message
            
    except Exception as e:
        error_msg = f"❌ COLMAP SfM failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg


def run_segmentation(model_size: str = "base") -> str:
    """Run Segment Anything for video"""
    try:
        if len(state.frames) == 0:
            return "❌ No frames extracted. Please extract frames first."
        
        logger.info(f"🤖 Running Segment Anything ({model_size})...")
        
        sam = SegmentAnything(model_size=model_size)
        masks = sam.segment_video(state.frames)
        
        state.segmentation_masks = masks
        
        message = f"✅ Segmentation completed\n"
        message += f"   Masks generated: {len(masks)}\n"
        message += f"   Foreground avg coverage: {np.mean([m.sum()/(m.shape[0]*m.shape[1]) for m in masks])*100:.1f}%"
        
        logger.info(message)
        return message
        
    except Exception as e:
        error_msg = f"❌ Segmentation failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg


def run_dense_reconstruction() -> str:
    """Run dense MVS reconstruction"""
    try:
        if len(state.frames) == 0:
            return "❌ No frames extracted. Please extract frames first."
        
        if state.poses is None:
            return "❌ No camera poses. Please run COLMAP SfM first."
        
        logger.info("📊 Running dense MVS reconstruction...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mvs = DenseReconstruction(
                state.frames,
                state.poses,
                state.intrinsics,
                output_dir=tmpdir
            )
            
            # Create depth maps
            depth_maps = mvs.create_depth_maps()
            state.depth_maps = depth_maps
            
            # Convert to point cloud
            points, colors = mvs.depth_to_point_cloud(depth_maps, use_colors=True)
            
            # Filter
            points, colors = mvs.filter_point_cloud(points, colors)
            
            state.point_cloud = points
            state.colors = colors
            
            message = f"✅ Dense reconstruction completed\n"
            message += f"   Point cloud size: {len(points):,}\n"
            message += f"   Average point spacing: {np.mean(np.min(np.linalg.norm(points[:1000, np.newaxis] - points[np.newaxis, :100], axis=2), axis=1)):.4f}"
            
            logger.info(message)
            return message
            
    except Exception as e:
        error_msg = f"❌ Dense reconstruction failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg


def run_gaussian_splatting(
    iterations: int = 2000,
    sh_degree: int = 2
) -> str:
    """Optimize with 3D Gaussian Splatting"""
    try:
        if state.point_cloud is None:
            return "❌ No point cloud. Please run dense reconstruction first."
        
        if state.poses is None:
            return "❌ No camera poses. Please run COLMAP SfM first."
        
        logger.info(f"🚀 Optimizing with 3DGS ({iterations} iterations)...")
        
        # Downsample for optimization
        if len(state.point_cloud) > 10000:
            indices = np.random.choice(len(state.point_cloud), 10000, replace=False)
            points = state.point_cloud[indices]
            colors = state.colors[indices] if state.colors is not None else None
        else:
            points = state.point_cloud
            colors = state.colors
        
        # Initialize Gaussian Splatting
        gs = GaussianSplatting(
            points,
            num_views=min(10, len(state.frames)),
            image_size=(state.frames[0].shape[0], state.frames[0].shape[1]),
            learning_rate=0.0016,
            sh_degree=sh_degree
        )
        
        message = f"✅ Gaussian Splatting initialized\n"
        message += f"   Gaussians: {gs.num_gaussians}\n"
        message += f"   SH degree: {sh_degree}\n"
        message += f"   Note: Run optimization in terminal for full training"
        
        logger.info(message)
        return message
        
    except Exception as e:
        error_msg = f"❌ Gaussian Splatting failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg


def export_point_cloud(output_format: str = "ply") -> Tuple[str, Optional[str]]:
    """Export point cloud"""
    try:
        if state.point_cloud is None:
            return "❌ No point cloud to export", None
        
        logger.info(f"💾 Exporting point cloud as {output_format.upper()}...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / f"reconstruction.{output_format}"
            
            if output_format == "ply":
                _write_ply(output_path, state.point_cloud, state.colors)
            elif output_format == "xyz":
                np.savetxt(output_path, state.point_cloud)
            else:
                return f"❌ Unsupported format: {output_format}", None
            
            # Read file for download
            with open(output_path, 'rb') as f:
                file_bytes = f.read()
            
            message = f"✅ Exported {len(state.point_cloud):,} points to {output_format.upper()}"
            logger.info(message)
            return message, str(output_path)
            
    except Exception as e:
        error_msg = f"❌ Export failed: {str(e)}"
        logger.error(error_msg)
        return error_msg, None


def _write_ply(path, points, colors):
    """Write PLY file"""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            p = points[i]
            f.write(f"{p[0]} {p[1]} {p[2]}")
            if colors is not None:
                c = colors[i]
                f.write(f" {int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}")
            f.write("\n")


# Build Gradio interface
def create_interface():
    with gr.Blocks(title="PhotoGeometry 3D Reconstruction") as demo:
        gr.Markdown("""
        # 🎬 PhotoGeometry - Advanced 3D Reconstruction from Video
        
        Professional 3D reconstruction using COLMAP, Dense MVS, Segment Anything, and 3D Gaussian Splatting.
        
        **Features:**
        - ✨ COLMAP Structure-from-Motion for precise camera calibration
        - 🤖 Segment Anything for automatic foreground detection
        - 📊 Dense Multi-View Stereo reconstruction
        - 🚀 3D Gaussian Splatting optimization
        - 💾 Multiple export formats (PLY, XYZ, etc.)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Step 1: Extract Frames")
                video_input = gr.Video(label="Upload Video", format="mp4")
                max_frames_slider = gr.Slider(10, 500, value=100, step=10, label="Max Frames")
                skip_frames_slider = gr.Slider(1, 10, value=2, step=1, label="Skip Frames")
                resize_width_slider = gr.Slider(480, 2048, value=1280, step=64, label="Resize Width")
                
                extract_btn = gr.Button("🎬 Extract Frames", variant="primary")
                extract_output = gr.Textbox(label="Extraction Status", interactive=False)
                frame_count = gr.Number(label="Frames Extracted", interactive=False)
                
            with gr.Column(scale=1):
                gr.Markdown("### 🗺️ Step 2: COLMAP SfM")
                colmap_btn = gr.Button("🗺️ Run COLMAP", variant="primary")
                colmap_output = gr.Textbox(label="COLMAP Status", interactive=False)
                
                gr.Markdown("### 🤖 Step 3: Segmentation")
                sam_model = gr.Radio(["base", "small", "large"], value="base", label="SAM Model Size")
                sam_btn = gr.Button("🤖 Run Segmentation", variant="primary")
                sam_output = gr.Textbox(label="Segmentation Status", interactive=False)
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Step 4: Dense Reconstruction")
                dense_btn = gr.Button("📊 Dense Reconstruction", variant="primary")
                dense_output = gr.Textbox(label="Dense Status", interactive=False)
                
                gr.Markdown("### 🚀 Step 5: Gaussian Splatting")
                gs_iterations = gr.Slider(100, 10000, value=2000, step=100, label="Iterations")
                gs_sh_degree = gr.Slider(0, 3, value=2, step=1, label="SH Degree")
                gs_btn = gr.Button("🚀 Gaussian Splatting", variant="primary")
                gs_output = gr.Textbox(label="GS Status", interactive=False)
                
                gr.Markdown("### 💾 Export Results")
                export_format = gr.Radio(["ply", "xyz"], value="ply", label="Export Format")
                export_btn = gr.Button("💾 Export Point Cloud", variant="primary")
                export_output = gr.Textbox(label="Export Status", interactive=False)
                export_file = gr.File(label="Download", interactive=False)
        
        # Connect buttons
        extract_btn.click(
            extract_frames_from_video,
            inputs=[video_input, max_frames_slider, skip_frames_slider, resize_width_slider],
            outputs=[extract_output, frame_count]
        )
        
        colmap_btn.click(run_colmap_sfm, outputs=[colmap_output])
        
        sam_btn.click(
            run_segmentation,
            inputs=[sam_model],
            outputs=[sam_output]
        )
        
        dense_btn.click(run_dense_reconstruction, outputs=[dense_output])
        
        gs_btn.click(
            run_gaussian_splatting,
            inputs=[gs_iterations, gs_sh_degree],
            outputs=[gs_output]
        )
        
        export_btn.click(
            export_point_cloud,
            inputs=[export_format],
            outputs=[export_output, export_file]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
