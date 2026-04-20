import os 
import sys
import cv2
import glob
import torch
import argparse
import numpy as np
from collections import deque
from pathlib import Path
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setup paths properly
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import HRNet components
from lib.hrnet.lib.config import cfg, update_config
from lib.hrnet.lib.models.pose_hrnet import get_pose_net
from lib.preprocess import h36m_coco_format, revise_kpts


# Import common utilities (reuse from original vis.py)
from common.utils import normalize_screen_coordinates
from common.camera import camera_to_world
from model.mixste.hot_mixste import Model

# Reuse visualization functions from vis.py
def show2Dpose(kps, img):
    colors = [(138, 201, 38),
              (25, 130, 196),
              (255, 202, 58)] 

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j]-1], thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=colors[LR[j]-1], radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=colors[LR[j]-1], radius=3)

    return img

# Add this function for 3D visualization
def show3Dpose_plt(pose_3d, ax):
    """Show 3D pose in a matplotlib 3D plot"""
    colors = ['red', 'blue', 'green']
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

    # Clear previous frame
    ax.clear()

    # Get X, Y, Z coordinates
    X = pose_3d[:, 0]
    Y = pose_3d[:, 1]
    Z = pose_3d[:, 2]

    for i in range(len(I)):
        x, y, z = [np.array([pose_3d[I[i], j], pose_3d[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=colors[LR[i]])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose')
    
    # Set equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    # Set view angle
    ax.view_init(elev=15, azim=45)

class HRNetWrapper:
    def __init__(self, model_path, hrnet_cfg_path, use_gpu=True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Initialize HRNet configuration
        cfg_args = argparse.Namespace()
        cfg_args.cfg = hrnet_cfg_path
        cfg_args.opts = []
        
        update_config(cfg, cfg_args)  # Updated to use cfg
        
        # Create HRNet model
        self.model = get_pose_net(cfg, pretrained="", is_train=False)  # Updated to use cfg
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Set image sizes from config
        self.target_height = cfg.MODEL.IMAGE_SIZE[0]  # Updated to use cfg
        self.target_width = cfg.MODEL.IMAGE_SIZE[1]   # Updated to use cfg
        
        # Setup image transformation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_height, self.target_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def detect_poses(self, frame):
        with torch.no_grad():
            input_tensor = self.preprocess_frame(frame)
            output = self.model(input_tensor)
            
            # Convert heatmaps to coordinates
            coords, scores = self._get_max_preds(output.cpu().numpy())
            
            # Scale coordinates to original image size
            h, w = frame.shape[:2]
            
            # Get first person's keypoints and scores
            coords = coords[0]  # First person only
            scores = scores[0]
            
            if np.mean(scores) > 0.3:  # Only if detection is confident
                # Scale coordinates from heatmap size to target size
                coords[:, 0] = coords[:, 0] * self.target_width / output.shape[3]
                coords[:, 1] = coords[:, 1] * self.target_height / output.shape[2]
                
                # Scale from target size to original image size
                coords[:, 0] = coords[:, 0] * w / self.target_width
                coords[:, 1] = coords[:, 1] * h / self.target_height
                
                # Calculate bounding box from valid keypoints
                valid_coords = coords[scores[:, 0] > 0.3]
                if len(valid_coords) > 0:
                    x_min = np.min(valid_coords[:, 0])
                    x_max = np.max(valid_coords[:, 0])
                    y_min = np.min(valid_coords[:, 1])
                    y_max = np.max(valid_coords[:, 1])
                    
                    # Add padding to bounding box
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    padding = max(bbox_width, bbox_height) * 0.2
                    
                    # Ensure bbox stays within image bounds
                    x_min = max(0, x_min - padding)
                    x_max = min(w, x_max + padding)
                    y_min = max(0, y_min - padding)
                    y_max = min(h, y_max + padding)
                    
                    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                    return coords, scores[:, 0], bbox
            
        return None, None, None

    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(frame_rgb)
        return input_tensor.unsqueeze(0).to(self.device)

    def _get_max_preds(self, heatmaps):
        # Reused from HRNet's get_max_preds function
        N, K, H, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, axis=2)
        maxvals = np.max(heatmaps_reshaped, axis=2)

        maxvals = maxvals.reshape((N, K, 1))
        idx = idx.reshape((N, K, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        pred_mask = np.tile(maxvals, (1, 1, 2)) > 0.0
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals

def load_pose_model(args):
    """
    Flexible model loading function that handles different model types
    """
    try:
        # Initialize 3D pose model
        model_3d = Model(args).to(args.device)
        
        model_path = None
        if args.checkpoint_3d is not None and os.path.isfile(args.checkpoint_3d):
            model_path = args.checkpoint_3d
        elif args.checkpoint_dir_3d is not None and os.path.isdir(args.checkpoint_dir_3d):
            checkpoints = glob.glob(os.path.join(args.checkpoint_dir_3d, '*.pth'))
            if not checkpoints:
                raise FileNotFoundError(f"No .pth files found in {args.checkpoint_dir_3d}")
            model_path = sorted(checkpoints)[0]
        else:
            raise FileNotFoundError("No valid model checkpoint path provided")

        # Load state dict
        print(f"Loading 3D pose model from: {model_path}")
        state_dict = torch.load(model_path, map_location=args.device)
        
        # Handle different state dict formats
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        
        # Try loading with strict=False first
        try:
            model_3d.load_state_dict(state_dict, strict=True) # Change to strict=True
            # print("Model loaded with strict=False")
        except Exception as e:
            print(f"Warning: {str(e)}")
            print("Attempting to fix state dict keys...")
            
            # Try to fix state dict keys
            fixed_state_dict = {}
            for k, v in state_dict.items():
                # Remove common prefixes if present
                k = k.replace('module.', '')
                k = k.replace('model.', '')
                fixed_state_dict[k] = v
            
            try:
                model_3d.load_state_dict(fixed_state_dict, strict=True) # Change to strict=True
                print("Model loaded successfully after fixing keys")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
        
        model_3d.eval()
        return model_3d
        
    except Exception as e:
        print(f"Error initializing 3D pose model: {str(e)}")
        raise

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_index', type=int, default=0)
    parser.add_argument('--checkpoint_hrnet', type=str, default='demo/lib/checkpoint/pose_hrnet_w48_384x288.pth')
    parser.add_argument('--checkpoint_dir_3d', type=str, help='Directory containing 3D model checkpoint')
    parser.add_argument('--checkpoint_3d', type=str, help='Direct path to 3D model checkpoint')
    parser.add_argument('--hrnet-cfg', type=str, default='demo/lib/hrnet/lib/config/w48_384x288_adam_lr1e-3.yaml')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--frames', type=int, default=243)
    
    # Model architecture parameters
    parser.add_argument('--model_type', type=str, default='hot_mixste', help='Type of 3D pose model')
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--channel', type=int, default=512)
    parser.add_argument('--d_hid', type=int, default=1024)
    parser.add_argument('--token_num', type=int, default=81)
    parser.add_argument('--layer_index', type=int, default=3)
    
    args = parser.parse_args()
    
    # Setup device
    args.use_gpu = args.gpu_id.lower() != 'cpu'
    args.device = torch.device('cuda' if args.use_gpu else 'cpu')
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Additional parameters
    args.pad = (args.frames - 1) // 2
    args.n_joints = args.out_joints = 17
    
    return args

# Update realtime_pose_estimation function
def realtime_pose_estimation(args):
    # Initialize HRNet
    hrnet_model = HRNetWrapper(
        model_path=args.checkpoint_hrnet,
        hrnet_cfg_path=args.hrnet_cfg,
        use_gpu=args.use_gpu
    )

    # Load 3D pose model with robust error handling
    model_3d = load_pose_model(args)

    # Print 3D model information
    # print("\n--- 3D Pose Model Information ---")
    # print(f"Model Type: {args.model_type}")
    # total_params = sum(p.numel() for p in model_3d.parameters())
    # trainable_params = sum(p.numel() for p in model_3d.parameters() if p.requires_grad)
    # print(f"Total Parameters: {total_params:,}")
    # print(f"Trainable Parameters: {trainable_params:,}")
    # # Assuming parameters are float32 (4 bytes)
    # model_size_mb = total_params * 4 / (1024**2)
    # print(f"Estimated Model Size: {model_size_mb:.2f} MB")
    # print(f"Layers: {args.layers}")
    # print(f"Channel: {args.channel}")
    # print(f"Hidden Dimension (d_hid): {args.d_hid}")
    # print(f"Token Number: {args.token_num}")
    # print(f"Layer Index: {args.layer_index}")
    # print("---------------------------------\n")
    
    # Start video capture
    
    # GStreamer
    gst_pipeline = (
        "tcpclientsrc host=192.168.1.113 port=5003 ! jpegdec ! videoconvert ! appsink sync=false"
    )
  
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    # TCP
    cap.open("tcp://192.168.1.113:5003")
    print("VideoCapture opened:", cap.isOpened())
    
    # Webcam
    # cap = cv2.VideoCapture(args.camera_index)
    
    if not cap.isOpened():
        print("Error: Could not open video stream. Check if the stream is running and the pipeline is correct.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize pose buffer
    pose_buffer_2d = deque(maxlen=args.frames)

    # Setup 3D visualization
    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # 2D pose estimation with bounding box
        keypoints, scores, bbox = hrnet_model.detect_poses(frame)
        
        if keypoints is not None and scores is not None and bbox is not None:
            # Draw bounding box
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Format keypoints for h36m_coco_format
            keypoints_formatted = keypoints[np.newaxis, np.newaxis, :, :]  # Add N and T dimensions
            scores_formatted = scores[np.newaxis, np.newaxis, :]  # Add N and T dimensions
            
            # Convert to H36M format
            keypoints_h36m, scores_h36m, valid_frames = h36m_coco_format(keypoints_formatted, scores_formatted)
            
            if valid_frames is not None and len(valid_frames) > 0:
                keypoints_h36m = revise_kpts(keypoints_h36m, scores_h36m, valid_frames)
                
                if len(keypoints_h36m) > 0:
                    # Extract single frame result and add to buffer
                    pose_2d = keypoints_h36m[0, 0]  # Extract first video, first frame
                    pose_buffer_2d.append(pose_2d)
                    
                    # Draw 2D skeleton
                    display_frame = show2Dpose(pose_2d, display_frame)
                    
                    # Add confidence visualization
                    cv2.putText(display_frame, 
                              f"Pose confidence: {np.mean(scores):.2f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)

            # 3D pose estimation when buffer is full
            if len(pose_buffer_2d) == args.frames:
                input_3d = np.array(list(pose_buffer_2d))
                input_3d = normalize_screen_coordinates(input_3d, w=frame_width, h=frame_height)
                input_3d = torch.from_numpy(input_3d).float().unsqueeze(0)
                
                if args.use_gpu:
                    input_3d = input_3d.cuda()

                with torch.no_grad():
                    pred_3d = model_3d(input_3d)
                    pred_3d = pred_3d[0, -1].cpu().numpy()

                # Convert to world coordinates
                rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
                pred_3d = camera_to_world(pred_3d, R=np.array(rot, dtype='float32'), t=0)
                
                # Update only the separate 3D visualization window
                ax.clear()
                show3Dpose_plt(pred_3d, ax)
                plt.draw()
                plt.pause(0.001)  # Add small pause to update the plot

        # Show 2D visualization
        cv2.imshow('HoT Real-time 2D Pose Estimation', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')

if __name__ == "__main__":
    args = parse_args()

    # ----- Optional: Check for available cameras -----
    # for idx in range(10):
    #     cap = cv2.VideoCapture(idx)
    #     if cap.isOpened():
    #         print("Camera found at index:", idx)
    #         cap.release()
    #     else:
    #         print("No camera found at index:", idx)
    # --------------------------------------------------


    realtime_pose_estimation(args)

'''
python demo/vis_realtime.py \
    --checkpoint_hrnet demo/lib/checkpoint/pose_hrnet_w48_384x288.pth \
    --checkpoint_dir_3d checkpoint/pretrained/hot_mixste \
    --frames 243 \
    --gpu_id 0
'''