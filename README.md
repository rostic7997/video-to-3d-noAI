# 🎬 PhotoGeometry 3D Reconstruction

Advanced 3D reconstruction from video with COLMAP, Dense MVS, Segment Anything, and Gaussian Splatting optimization.

## ✨ Features

- **COLMAP Integration** - Precise camera calibration and Structure-from-Motion (SfM)
- **Dense MVS Reconstruction** - High-quality dense point clouds (replaces sparse triangulation)
- **Segment Anything** - Automatic foreground/background segmentation for improved quality
- **Differentiable Rendering** - PyTorch-based 3D Gaussian Splatting optimization (3DGS)
- **Adaptive Gaussian Strategy** - Smart addition/deletion of Gaussians for optimal detail
- **Neural Geometry Refinement** - Zero-shot neural methods for geometry enhancement
- **Multi-View Optimization** - Bundle adjustment and consistency across views

## 📋 Requirements

- Python >= 3.8
- GPU with CUDA support (for best performance)
- COLMAP pre-installed on system

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rostic7997/video-to-3d-noAI.git
cd video-to-3d-noAI
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install COLMAP (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y colmap
```

For other OS, see: https://colmap.github.io/install.html

### Usage

#### Via Gradio Web Interface
```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

#### Via Command Line
```python
from colmap_pipeline import COLMAPPipeline
from gaussian_splatting import GaussianSplatting
from segmentation import SegmentAnything

# Step 1: Extract frames and run COLMAP
pipeline = COLMAPPipeline(
    video_path="video.mp4",
    output_dir="./reconstruction",
    max_frames=300,
    skip_frames=2
)
frames, poses, intrinsics = pipeline.run_colmap_sfm()

# Step 2: Apply segmentation
sam = SegmentAnything()
segmentation_masks = sam.segment_video(frames)

# Step 3: Dense reconstruction
mvs_pointcloud = pipeline.run_dense_mvs()

# Step 4: Optimize with Gaussian Splatting
gs_optimizer = GaussianSplatting(
    point_cloud=mvs_pointcloud,
    images=frames,
    poses=poses,
    intrinsics=intrinsics,
    learning_rate=0.0016
)
optimized_model = gs_optimizer.optimize(iterations=7000)

# Export results
gs_optimizer.export_ply("output.ply")
```

## 📊 Processing Pipeline

```
Video Input
    ↓
Frame Extraction & Quality Check
    ↓
COLMAP Feature Detection & Matching
    ↓
SfM Reconstruction (Camera Poses)
    ↓
Segment Anything (Foreground Detection)
    ↓
Dense MVS Reconstruction
    ↓
Point Cloud Filtering
    ↓
3D Gaussian Splatting Optimization
    ↓
Adaptive Gaussian Strategy
    ↓
Neural Geometry Refinement
    ↓
Export (PLY, USD, ply, etc.)
```

## 📁 Project Structure

```
.
├── app.py                      # Gradio web interface
├── colmap_pipeline.py          # COLMAP integration & SfM
├── gaussian_splatting.py       # Differentiable rendering & optimization
├── segmentation.py             # Segment Anything integration
├── mvs_reconstruction.py       # Dense point cloud reconstruction
├── neural_refinement.py        # Zero-shot geometry enhancement
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎯 Configuration

### Key Parameters

**Frame Extraction:**
- `max_frames`: Maximum frames to process (higher = better quality, slower)
- `skip_frames`: Process every N-th frame (reduce computation)
- `resize_width`: Resize frames to this width (balance quality/speed)

**COLMAP:**
- `colmap_matcher`: Feature matcher type (sequential, exhaustive, spatial)
- `colmap_triangulator`: Triangulation method (delaunay, auto)

**Gaussian Splatting:**
- `learning_rate`: Initial learning rate (default: 0.0016)
- `sh_degree`: Spherical Harmonics degree (0-3, higher = more colors)
- `position_lr_init`: Position learning rate (default: 0.00016)
- `position_lr_final`: Final position learning rate

**Segmentation:**
- `sam_model_size`: tiny/small/base/large (larger = better quality, slower)
- `sam_iou_threshold`: IoU threshold for object detection (0.0-1.0)

## 🎬 Input Requirements

For best results:

1. **Video Quality:**
   - Resolution: 1080p+ recommended
   - Duration: 30-120 seconds
   - FPS: 24-60 recommended

2. **Camera Motion:**
   - Smooth, slow camera movements
   - Circular or figure-8 patterns around object
   - Avoid rapid movements or motion blur

3. **Lighting:**
   - Well-lit scene (natural light or studio lighting)
   - Consistent lighting across frames
   - Avoid harsh shadows

4. **Object:**
   - Textured surface (avoid white/plain objects)
   - Rigid objects work best
   - Avoid thin/transparent structures

## 📤 Export Formats

- **PLY** - Point cloud with colors and normals
- **USD** - Universal Scene Description (for 3D viewers)
- **OBJ** - Mesh format
- **GLTF** - glTF 2.0 format
- **XYZ** - Simple point coordinate format

## 🔧 Advanced Usage

### Custom COLMAP Configuration

```python
from colmap_pipeline import COLMAPPipeline

pipeline = COLMAPPipeline(
    video_path="video.mp4",
    output_dir="./reconstruction",
    colmap_config={
        'matcher_type': 'exhaustive',
        'triangulation_type': 'auto',
        'mapper_min_track_length': 2,
        'mapper_min_num_matches': 15
    }
)
```

### Fine-tuning Gaussian Splatting

```python
from gaussian_splatting import GaussianSplatting

gs = GaussianSplatting(
    point_cloud=points,
    images=frames,
    poses=poses,
    intrinsics=intrinsics,
    learning_rate=0.001,
    sh_degree=3,
    densification_interval=100,
    pruning_interval=500
)

# Resume from checkpoint
gs.load_checkpoint("checkpoint.pth")
```

### Segmentation Filtering

```python
from segmentation import SegmentAnything

sam = SegmentAnything(model_size='large')

# Get foreground masks
masks = sam.segment_video(frames)

# Apply masks to point cloud
filtered_cloud = mvs.filter_by_mask(mvs_pointcloud, masks)
```

## 🐛 Troubleshooting

### COLMAP not found
```bash
# Install COLMAP
sudo apt-get install colmap

# Or build from source
git clone https://github.com/colmap/colmap.git
cd colmap && mkdir build && cd build
cmake .. && make -j4 && sudo make install
```

### Out of Memory (OOM)
- Reduce `max_frames`
- Increase `skip_frames`
- Reduce `resize_width`
- Use smaller SAM model size

### Poor reconstruction quality
- Use better lighting
- Record with higher FPS
- Move camera more slowly
- Process more frames
- Increase Gaussian Splatting iterations

### COLMAP reconstruction fails
- Ensure clear, textured objects
- Check if camera moved during recording
- Verify video codec compatibility
- Try different COLMAP matcher type

## 📊 Performance

**Typical processing times (HD video, 100 frames):**
- Frame extraction: 10-20s
- COLMAP SfM: 2-5 min
- Dense MVS: 1-3 min
- Gaussian Splatting (7000 iter): 5-10 min
- **Total: ~15-25 minutes**

## 🔬 Technical Details

### COLMAP Integration
- Automatic feature detection (SIFT)
- Robust feature matching with RANSAC
- Incremental SfM for camera localization
- Bundle adjustment for pose refinement

### Dense MVS
- OpenMVS-based depth map estimation
- Multi-view consistency
- Poisson surface reconstruction
- Mesh generation from point clouds

### Gaussian Splatting (3DGS)
- Spherical Harmonics for view-dependent colors
- Covariance matrix optimization
- Adaptive Gaussian density control
- Differentiable volume rendering with PyTorch

### Segment Anything
- Foundation model for image segmentation
- Zero-shot object detection
- Foreground/background separation
- Quality improvement via mask-based filtering

## 📚 References

- [COLMAP](https://colmap.github.io/)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Segment Anything](https://segment-anything.com/)
- [OpenMVS](https://github.com/cdcseacave/openMVS)
- [PyTorch](https://pytorch.org/)

## 📜 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## 📧 Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions

## 🙏 Acknowledgments

Built on top of:
- COLMAP (Schönberger et al.)
- 3D Gaussian Splatting (Kerbl et al.)
- Segment Anything (Kirillov et al.)
- OpenMVS community

---

**Made with ❤️ for 3D reconstruction enthusiasts**

