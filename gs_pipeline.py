import os
import numpy as np
import torch
from PIL import Image
import json
import argparse
import shutil

class GaussianSplattingPipeline:
    def __init__(self, data_dir, output_dir, frame=None):
        """
        Initialize the Gaussian Splatting pipeline.
        
        Args:
            data_dir: Base directory containing Frame folders
            output_dir: Directory to save outputs
            frame: Specific frame to process (e.g., "Frame0001"). If None, processes first frame.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Find the frame to process
        if frame is None:
            # Find the first frame folder
            for item in os.listdir(data_dir):
                if item.startswith("Frame") and os.path.isdir(os.path.join(data_dir, item)):
                    frame = item
                    break
        
        self.frame_dir = os.path.join(data_dir, frame)
        print(f"Processing frame: {frame}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize parameters
        self.gaussians = None
        self.cameras = []
        
        # Hyperparameters
        self.num_iterations = 5000
        self.learning_rate = 0.001
        
    def load_colmap_cameras(self):
        """Load camera parameters from COLMAP data"""
        print("Loading COLMAP camera data...")
        
        # Path to cameras.txt and images.txt in your frame folder
        cameras_file = os.path.join(self.frame_dir, "cameras.txt")
        images_file = os.path.join(self.frame_dir, "images.txt")
        
        # Parse cameras.txt for intrinsics
        camera_intrinsics = {}
        if os.path.exists(cameras_file):
            with open(cameras_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("#") or line.strip() == "":
                        continue
                    parts = line.strip().split()
                    if len(parts) > 3:
                        camera_id = int(parts[0])
                        camera_model = parts[1]
                        width = int(parts[2])
                        height = int(parts[3])
                        
                        # For OPENCV camera model
                        if camera_model == "OPENCV":
                            fx = float(parts[4])
                            fy = float(parts[5])
                            cx = float(parts[6])
                            cy = float(parts[7])
                            
                            camera_intrinsics[camera_id] = {
                                "width": width,
                                "height": height,
                                "fx": fx,
                                "fy": fy,
                                "cx": cx,
                                "cy": cy
                            }
                        # Handle other camera models like SIMPLE_PINHOLE or PINHOLE
                        elif camera_model in ["SIMPLE_PINHOLE", "PINHOLE"]:
                            fx = float(parts[4])
                            fy = float(parts[5]) if camera_model == "PINHOLE" else fx
                            cx = float(parts[6]) if camera_model == "PINHOLE" else float(parts[5])
                            cy = float(parts[7]) if camera_model == "PINHOLE" else float(parts[6])
                            
                            camera_intrinsics[camera_id] = {
                                "width": width,
                                "height": height,
                                "fx": fx,
                                "fy": fy,
                                "cx": cx,
                                "cy": cy
                            }
        else:
            print(f"Warning: {cameras_file} not found. Using default camera parameters.")
        
        # Parse images.txt for extrinsics
        if os.path.exists(images_file):
            with open(images_file, 'r') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith("#") or line.strip() == "":
                        i += 1
                        continue
                        
                    parts = line.strip().split()
                    if len(parts) > 7:
                        image_id = int(parts[0])
                        qw, qx, qy , qz = map(float, parts[1:5])  # Quaternion
                        tx, ty, tz = map(float, parts[5:8])  # Translation
                        camera_id = int(parts[8])
                        image_name = parts[9]
                        
                        # Find corresponding intrinsics
                        if camera_id in camera_intrinsics:
                            intrinsics = camera_intrinsics[camera_id]
                            
                            # Convert quaternion to rotation matrix
                            r = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
                            t = np.array([tx, ty, tz]).reshape(3, 1)
                            
                            # Create camera object
                            camera = {
                                "image_name": image_name,
                                "rotation": r,
                                "translation": t,
                                "intrinsics": intrinsics
                            }
                            
                            self.cameras.append(camera)
                    
                    # Skip the next line with points
                    i += 2
        else:
            print(f"Warning: {images_file} not found. Camera extrinsics are missing.")
        
        print(f"Loaded {len(self.cameras)} cameras.")
        return len(self.cameras) > 0
    
    def _quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        """Convert quaternion to rotation matrix"""
        R = np.zeros((3, 3))
        
        R[0, 0] = 1 - 2 * qy**2 - 2 * qz**2
        R[0, 1] = 2 * qx * qy - 2 * qz * qw
        R[0, 2] = 2 * qx * qz + 2 * qy * qw
        
        R[1, 0] = 2 * qx * qy + 2 * qz * qw
        R[1, 1] = 1 - 2 * qx**2 - 2 * qz**2
        R[1, 2] = 2 * qy * qz - 2 * qx * qw
        
        R[2, 0] = 2 * qx * qz - 2 * qy * qw
        R[2, 1] = 2 * qy * qz + 2 * qx * qw
        R[2, 2] = 1 - 2 * qx**2 - 2 * qy**2
        
        return R
    
    def load_images(self):
        """Load images from the frame directory"""
        print("Loading images...")
        
        images = []
        for camera in self.cameras:
            image_path = os.path.join(self.frame_dir, camera["image_name"])
            if os.path.exists(image_path):
                img = Image.open(image_path)
                img = np.array(img) / 255.0  # Normalize to [0, 1]
                images.append(torch.tensor(img, device=self.device, dtype=torch.float32).requires_grad_())
            else:
                print(f"Warning: Image {image_path} not found.")
                images.append(None)
        
        print(f"Loaded {sum(1 for img in images if img is not None)} images.")
        return images
    
    def load_points3D(self):
        """Load 3D points from COLMAP reconstruction for initialization"""
        points3D_file = os.path.join(self.frame_dir, "points3D.txt")
        points = []
        
        if os.path.exists(points3D_file):
            with open(points3D_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("#") or line.strip() == "":
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        x, y, z = map(float, parts[1:4])
                        r, g, b = map(int, parts[4:7])
                        points.append([x, y, z, r/255.0, g/255.0, b/255.0])
                        
            points = np.array(points)
            print(f"Loaded {len(points)} points from points3D.txt")
            return points
        else:
            print("points3D.txt not found. Will use random initialization.")
            return None
    
    def initialize_gaussians(self, num_gaussians=5000):
        """Initialize Gaussian parameters using 3D points or randomly"""
        print(f"Initializing Gaussians...")
        
        # Try to load points3D for better initialization
        points = self.load_points3D()
        
        if points is not None and len(points) > 0:
            # Use points from COLMAP reconstruction
            num_points = min(num_gaussians, len(points))
            indices = np.random.choice(len(points), num_points, replace=False)
            selected_points= points[indices]
            
            positions = torch.tensor(selected_points[:, :3], device=self.device, dtype=torch.float32).requires_grad_()
            colors = torch.tensor(selected_points[:, 3:6], device=self.device, dtype=torch.float32).requires_grad_()
            
            num_gaussians = num_points
            print(f"Initialized {num_gaussians} Gaussians from COLMAP points.")
        else:
            # Get scene bounds from camera positions
            camera_positions = np.array([cam["translation"].flatten() for cam in self.cameras])
            min_bounds = np.min(camera_positions, axis=0) - 1.0
            max_bounds = np.max(camera_positions, axis=0) + 1.0
            
            # Randomly initialize positions within the scene bounds
            positions = torch.rand(num_gaussians, 3, device=self.device) * \
                        torch.tensor(max_bounds - min_bounds, device=self.device) + \
                        torch.tensor(min_bounds, device=self.device)
            
            # Random colors
            colors = torch.rand(num_gaussians, 3, device=self.device)  # RGB
            print(f"Initialized {num_gaussians} Gaussians randomly.")
        
        # Initialize scales, rotations, and opacities
        scales = torch.ones(num_gaussians, 3, device=self.device) * 0.01
        rotations = torch.zeros(num_gaussians, 4, device=self.device)  # Quaternion (w, x, y, z)
        rotations[:, 0] = 1.0  # Identity rotation
        opacities = torch.ones(num_gaussians, 1, device=self.device) * 0.5
        
        self.gaussians = {
            "positions": positions,
            "scales": scales,
            "rotations": rotations,
            "colors": colors,
            "opacities": opacities
        }
        
        return self.gaussians
    
    def train(self, images):
        """Train the Gaussian Splatting model"""
        print(f"Training for {self.num_iterations} iterations...")
        
        # Initialize optimization parameters
        for param_name, param in self.gaussians.items():
            param.requires_grad_(True)
        
        # Create optimizer
        optimizer = torch.optim.Adam([
            {"params": self.gaussians["positions"], "lr": self.learning_rate},
            {"params": self.gaussians["scales"], "lr": self.learning_rate * 0.1},
            {"params": self.gaussians["rotations"], "lr": self.learning_rate * 0.01},
            {"params": self.gaussians["colors"], "lr": self.learning_rate},
            {"params": self.gaussians["opacities"], "lr": self.learning_rate}
        ])
        
        # Training loop
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Randomly select a camera/image for this iteration
            camera_idx = np.random.randint(0, len(self.cameras))
            camera = self.cameras[camera_idx]
            target_image = images[camera_idx]
            
            if target_image is None:
                continue
                
            # Render the image from this camera view
            rendered_image = self.render(camera)
            
            # Ensure the target image has the same number of channels as rendered image
            if target_image.shape[2] != rendered_image.shape[2]:
                target_image = target_image[:, :, :3]  # Convert RGBA to RGB if necessary
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(rendered_image, target_image)
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            
            # Log progress
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.num_iterations}, Loss: {loss.item():.6f}")
                
                # Save checkpoint
                if (iteration + 1) % 1000 == 0:
                    self.save_checkpoint(iteration + 1)
        
        print("Training completed.")
    
    def render(self, camera):
        """Render an image from a specific camera view using Gaussian splatting"""
        # This is a simplified rendering function
        # In a real implementation, this would perform the actual Gaussian splatting
        
        # Get camera parameters
        R = torch.tensor(camera["rotation"], device=self.device, dtype=torch.float32)
        t = torch.tensor(camera["translation"], device=self.device, dtype=torch.float32)
        intrinsics = camera["intrinsics"]
        
        width = intrinsics["width"]
        height = intrinsics["height"]
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        
        # Create a mock rendered image (placeholder for actual rendering)
        rendered_image = torch.zeros((height, width, 3), device=self.device, requires_grad=True)
        
        # Note: This is where the actual rendering algorithm would go
        # For a complete implementation, refer to the 3D Gaussian Splatting papers
        
        return rendered_image
    
    def save_checkpoint(self, iteration):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save Gaussian parameters
        for param_name, param in self.gaussians.items():
            torch.save(param.detach().cpu(), os.path.join(checkpoint_dir, f"{param_name}.pt"))
        
        print(f"Saved checkpoint at iteration {iteration}")
    
    def export_model(self):
        """Export the trained model in a standard format"""
        export_path = os.path.join(self.output_dir, "exported_model")
        os.makedirs(export_path, exist_ok=True)
        
        # Save Gaussian parameters in a format that can be loaded by viewers
        gaussians_data = {}
        for param_name, param in self.gaussians.items():
            gaussians_data[param_name] = param.detach().cpu().numpy()
        
        np.savez(os.path.join(export_path, "gaussians.npz"), **gaussians_data)
        
        # Save metadata
        metadata = {
            "num_gaussians": self.gaussians["positions"].shape[0],
            "creation_date": "simplified_pipeline"
        }
        
        with open(os.path.join(export_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Exported model to {export_path}")
        
    def run_pipeline(self):
        """Run the complete Gaussian Splatting pipeline"""
        print("Starting Gaussian Splatting pipeline...")
        
        # Step 1: Load camera data
        if not self.load_colmap_cameras():
            print("Failed to load camera data. Exiting.")
            return False
        
        # Step 2: Load images
        images = self.load_images()
        
        # Step 3: Initialize Gaussians
        self.initialize_gaussians()
        
        # Step 4: Train the model
        self.train(images)
        
        # Step 5: Export the final model
        self.export_model()
        
        print("Pipeline completed successfully!")
        return True


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified Gaussian Splatting Pipeline for Heart 4DGS")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing Frame folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--frame", type=str, help="Specific frame to process (e.g., 'Frame0001')")
    parser.add_argument("--iterations", type=int, default=20000, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create and run the pipeline
    pipeline = GaussianSplattingPipeline(args.data_dir, args.output_dir, args.frame)
    pipeline.num_iterations = args.iterations
    pipeline.learning_rate = args.lr
    pipeline.run_pipeline()