# **4D Gaussian Splatting Pipeline**  

https://github.com/user-attachments/assets/9256fc36-219f-425c-aac1-40de6f2cc510
## **Overview**  
This project implements a **4D Gaussian Splatting (4DGS) pipeline** for reconstructing dynamic 3D scenes from multi-view images. The system processes frames captured from a **camera array setup** around an animated object and optimizes **3D Gaussians** to represent the scene efficiently.  

The project includes:  

- **Dataset Generation**: Created in **Blender**, where an array of cameras surrounds an animated object, capturing images from multiple viewpoints.  
- **Gaussian Optimization Pipeline**: Processes **COLMAP-formatted** camera data, initializes **3D Gaussians**, and optimizes them using PyTorch.  
- **Output Processing**: Converts Gaussian data into `.ply`, `.fbx`, or rendered images using additional scripts.  

---

## **Dataset Structure**  
The dataset consists of **frames captured using a camera array**, stored in a structured format:  

```
C:\Users\91910\Downloads\heart_4dgs> tree /f
Folder PATH listing for volume Windows
Volume serial number is 349D-C389
C:.
├───Frame0001
│       cameras.txt
│       images.txt
│       points3D.txt
│       MinAngle26_ArrayCam_IN.001.png
│       MinAngle26_ArrayCam_IN.002.png
│       ...
│       MinAngle26_ArrayCam_IN.026.png
│
├───Frame0002
│       cameras.txt
│       images.txt
│       points3D.txt
│       MinAngle26_ArrayCam_IN.001.png
│       ...
```

Each frame contains:  
- `cameras.txt`: Camera intrinsics from COLMAP.  
- `images.txt`: Image-to-camera mappings.  
- `points3D.txt`: 3D points reconstructed by COLMAP.  
- PNG images: Multi-view images captured from the **camera array** in Blender.  
![gaussian_visualization](https://github.com/user-attachments/assets/0afa4ebb-f741-40c2-8e63-2e066b5537cc)

---

## **Pipeline Structure**  

### **Step 1: Gaussian Splatting Optimization**  
The `gs_pipeline.py` script processes the dataset and optimizes Gaussian parameters:  

#### **Usage**  
```bash
python gs_pipeline.py --data_dir <COLMAP_DATA_PATH> --output_dir <OUTPUT_PATH> --frame Frame0001 --iterations 20000 --lr 0.001
```

#### **Arguments**  
- `--data_dir`: Path to dataset (`Frame0001`, `Frame0002` folders).  
- `--output_dir`: Where to save Gaussian model checkpoints.  
- `--frame`: (Optional) Specific frame to process.  
- `--iterations`: Training iterations (default: 20,000).  
- `--lr`: Learning rate (default: 0.001).  

### **Step 2: Model Checkpoints (Intermediate Output)**  
The optimization produces **Gaussian splatting parameters** stored in `.pt` format:  

```
C:\Users\91910\Downloads\heart_gs_project\gs_output\checkpoint_2000> tree /f
Folder PATH listing for volume Windows
Volume serial number is 349D-C389
C:.
    colors.pt
    opacities.pt
    positions.pt
    rotations.pt
    scales.pt
```

Each file represents a tensor:  
- `positions.pt`: 3D coordinates of the Gaussians.  
- `colors.pt`: RGB values for each Gaussian.  
- `scales.pt`: Size of each Gaussian in 3D space.  
- `rotations.pt`: Quaternion rotations.  
- `opacities.pt`: Transparency values.  

---

## **Step 3: Exporting & Visualization**  
To export the Gaussians into standard 3D formats (`.ply`, `.fbx`) or visualize them as images, additional scripts are used:  

| **Script** | **Function** |
|------------|-------------|
| `gs_file_to_ply.ipynb` | Converts Gaussian `.pt` files to `.ply` for MeshLab/Blender. |
| `gs_file_to_image.ipynb` | Renders the Gaussian model into images. |

#### **Export Example**  
To convert Gaussians to `.ply`:  
```bash
python gs_file_to_ply.py --input_dir <OUTPUT_PATH> --output_file model.ply
```

---

## **Future Improvements**  
- **Enhance rendering quality** with more efficient splatting.  
- **Support real-time rendering** via CUDA acceleration.  
- **Extend to true 4D** (dynamic splatting with time-based motion).  

## **License**  
MIT License.  



This README now fully details your **pipeline, dataset, processing flow, and output structure.** Let me know if you'd like any refinements! 🚀
 
