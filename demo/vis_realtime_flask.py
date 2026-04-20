import os
import sys
import cv2
import torch
import argparse
import numpy as np
from collections import deque
from pathlib import Path
import torchvision.transforms as transforms
from flask import Flask, render_template, Response
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import threading

# Setup paths properly
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from lib.hrnet.lib.config import cfg, update_config
from lib.hrnet.lib.models.pose_hrnet import get_pose_net
from lib.preprocess import h36m_coco_format, revise_kpts
from common.utils import normalize_screen_coordinates
from common.camera import camera_to_world
from model.mixste.hot_mixste import Model

app = Flask(__name__)

# --- Shared video capture thread ---
class SharedVideoCapture:
    def __init__(self, pipeline):
        # print(f"[DEBUG] Initializing SharedVideoCapture with pipeline: {pipeline}")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        # print(f"[DEBUG] After constructor, cap.isOpened(): {self.cap.isOpened()}")
        # Try explicit open as in working script
        open_result = self.cap.open("tcp://127.0.0.1:5000")
        # print(f"[DEBUG] Called cap.open('tcp://192.168.1.112:5001'), result: {open_result}")
        # print(f"[DEBUG] After open(), cap.isOpened(): {self.cap.isOpened()}")
        if not self.cap.isOpened():
            print("[ERROR] SharedVideoCapture: Could not open video stream!")
        else:
            print("[INFO] SharedVideoCapture: Video stream opened.")
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        fail_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                fail_count = 0
                # print(f"[DEBUG] Frame received: shape={frame.shape}, dtype={frame.dtype}")
            else:
                fail_count += 1
                print(f"[WARNING] SharedVideoCapture: cap.read() failed (fail_count={fail_count})")
                if fail_count % 100 == 0:
                    print("[WARNING] SharedVideoCapture: No frame received for 100 attempts.")
                time.sleep(0.005)

    def read(self):
        return self.frame  # No copy, always latest

    def release(self):
        self.running = False
        self.cap.release()

# --- Helper functions (copied from your original script) ---
def show2Dpose_flask(kps, img):
    colors = [(138, 201, 38), (25, 130, 196), (255, 202, 58)]
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    LR = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    thickness = 3
    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j]-1], thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=colors[LR[j]-1], radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=colors[LR[j]-1], radius=3)
    return img

def plot3Dpose_flask(pose_3d):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'blue', 'green']
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]
    X = pose_3d[:, 0]
    Y = pose_3d[:, 1]
    Z = pose_3d[:, 2]
    for i in range(len(I)):
        x, y, z = [np.array([pose_3d[I[i], j], pose_3d[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=colors[LR[i]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose')
    # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    # mid_x = (X.max()+X.min()) * 0.5
    # mid_y = (Y.max()+Y.min()) * 0.5
    # mid_z = (Z.max()+Z.min()) * 0.5
    # ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    # ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    # ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    ax.set_xlim(-0.5, 1)
    ax.set_ylim(-0.5, 1)
    ax.set_zlim(-0.5, 1)

    ax.view_init(elev=15, azim=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

class HRNetWrapper:
    def __init__(self, model_path, hrnet_cfg_path, use_gpu=True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        cfg_args = argparse.Namespace()
        cfg_args.cfg = hrnet_cfg_path
        cfg_args.opts = []
        update_config(cfg, cfg_args)
        self.model = get_pose_net(cfg, pretrained="", is_train=False)
        state_dict = torch.load(model_path, map_location=self.device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.target_height = cfg.MODEL.IMAGE_SIZE[0]
        self.target_width = cfg.MODEL.IMAGE_SIZE[1]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_height, self.target_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def detect_poses(self, frame):
        with torch.no_grad():
            input_tensor = self.preprocess_frame(frame)
            output = self.model(input_tensor)
            coords, scores = self._get_max_preds(output.cpu().numpy())
            h, w = frame.shape[:2]
            coords = coords[0]
            scores = scores[0]
            if np.mean(scores) > 0.3:
                coords[:, 0] = coords[:, 0] * self.target_width / output.shape[3]
                coords[:, 1] = coords[:, 1] * self.target_height / output.shape[2]
                coords[:, 0] = coords[:, 0] * w / self.target_width
                coords[:, 1] = coords[:, 1] * h / self.target_height
                valid_coords = coords[scores[:, 0] > 0.3]
                if len(valid_coords) > 0:
                    x_min = np.min(valid_coords[:, 0])
                    x_max = np.max(valid_coords[:, 0])
                    y_min = np.min(valid_coords[:, 1])
                    y_max = np.max(valid_coords[:, 1])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    padding = max(bbox_width, bbox_height) * 0.2
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
    model_3d = Model(args).to(args.device)
    model_path = None
    if args.checkpoint_3d is not None and os.path.isfile(args.checkpoint_3d):
        model_path = args.checkpoint_3d
    elif args.checkpoint_dir_3d is not None and os.path.isdir(args.checkpoint_dir_3d):
        import glob
        checkpoints = glob.glob(os.path.join(args.checkpoint_dir_3d, '*.pth'))
        if not checkpoints:
            raise RuntimeError(f"No .pth files found in {args.checkpoint_dir_3d}")
        model_path = sorted(checkpoints)[0]
    else:
        raise RuntimeError("No valid model checkpoint path provided")
    state_dict = torch.load(model_path, map_location=args.device)
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
    model_3d.load_state_dict(state_dict, strict=True)
    model_3d.eval()
    return model_3d

# --- Shared video capture instance ---
# gst_pipeline = "tcpclientsrc host=192.168.1.78 port=5005 ! jpegdec ! videoconvert ! appsink sync=false"
gst_pipeline = "tcpclientsrc host=192.168.1.119 port=5000 ! jpegdec ! videoconvert ! appsink sync=false"
shared_capture = SharedVideoCapture(gst_pipeline)

#self.rgb_cap = cv2.VideoCapture(self.rgb_pipeline, cv2.CAP_GSTREAMER)
#   self.rgb_cap.open(f"tcp://{self.PLATFORM_IP}:{self.RGB_GST_PORT}")


# --- Flask routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose3d_feed')
def pose3d_feed():
    return Response(gen_3d_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Video generator ---
def gen_frames():
    args = argparse.Namespace()
    args.checkpoint_hrnet = "demo/lib/checkpoint/pose_hrnet_w48_384x288.pth"
    args.hrnet_cfg = "demo/lib/hrnet/lib/config/w48_384x288_adam_lr1e-3.yaml"
    args.use_gpu = torch.cuda.is_available()
    hrnet_model = HRNetWrapper(
        model_path=args.checkpoint_hrnet,
        hrnet_cfg_path=args.hrnet_cfg,
        use_gpu=args.use_gpu
    )
    frame_count = 0
    while True:
        frame = shared_capture.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_draw = frame.copy()
        if frame is None:
            print("[DEBUG] gen_frames: No frame from shared_capture. Waiting...")
            time.sleep(0.05)
            continue
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[DEBUG] gen_frames: Streaming frame {frame_count}. Shape: {frame.shape}, dtype: {frame.dtype}, mean: {np.mean(frame):.2f}")
        # print(frame_count)
        keypoints, scores, bbox = hrnet_model.detect_poses(frame)
        # print(keypoints, scores, bbox)
        if keypoints is not None and scores is not None and bbox is not None:
	    # print('draw2D pose')
            x_min, y_min, x_max, y_max = bbox
            frame_draw = frame.copy()
            cv2.rectangle(frame_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            keypoints_formatted = keypoints[np.newaxis, np.newaxis, :, :]
            scores_formatted = scores[np.newaxis, np.newaxis, :]
            keypoints_h36m, scores_h36m, valid_frames = h36m_coco_format(keypoints_formatted, scores_formatted)
            if valid_frames is not None and len(valid_frames) > 0:
                keypoints_h36m = revise_kpts(keypoints_h36m, scores_h36m, valid_frames)
                if len(keypoints_h36m) > 0:
                    pose_2d = keypoints_h36m[0, 0]
                    # frame = show2Dpose_flask(pose_2d, frame)
                    frame_draw = show2Dpose_flask(pose_2d, frame_draw)
                    cv2.putText(frame_draw, f"Pose confidence: {np.mean(scores):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]  # Lower quality for speed
        ret, buffer = cv2.imencode('.jpg', frame_draw, encode_param)
        # ret, buffer = cv2.imencode('.jpg', frame_vis, encode_param)
        if not ret:
            print("[ERROR] gen_frames: cv2.imencode failed!")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 3D pose generator

def gen_3d_frames():
    args = argparse.Namespace()
    args.checkpoint_hrnet = "demo/lib/checkpoint/pose_hrnet_w48_384x288.pth"
    args.hrnet_cfg = "demo/lib/hrnet/lib/config/w48_384x288_adam_lr1e-3.yaml"
    args.checkpoint_dir_3d = "checkpoint/pretrained/hot_mixste"
    args.use_gpu = torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_gpu else 'cpu')
    args.frames = 243
    args.pad = (args.frames - 1) // 2
    args.n_joints = args.out_joints = 17
    args.checkpoint_3d = None
    args.model_type = 'hot_mixste'
    args.layers = 8
    args.channel = 512
    args.d_hid = 1024
    args.token_num = 81
    args.layer_index = 3
    hrnet_model = HRNetWrapper(
        model_path=args.checkpoint_hrnet,
        hrnet_cfg_path=args.hrnet_cfg,
        use_gpu=args.use_gpu
    )
    model_3d = load_pose_model(args)
    pose_buffer_2d = deque(maxlen=args.frames)
    frame_width = 1920  # Default, update on first frame
    frame_height = 1080
    while True:
        frame = shared_capture.read()
        if frame is None:
            time.sleep(0.005)
            continue
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        keypoints, scores, bbox = hrnet_model.detect_poses(frame)
        if keypoints is not None and scores is not None and bbox is not None:
            keypoints_formatted = keypoints[np.newaxis, np.newaxis, :, :]
            scores_formatted = scores[np.newaxis, np.newaxis, :]
            keypoints_h36m, scores_h36m, valid_frames = h36m_coco_format(keypoints_formatted, scores_formatted)
            if valid_frames is not None and len(valid_frames) > 0:
                keypoints_h36m = revise_kpts(keypoints_h36m, scores_h36m, valid_frames)
                if len(keypoints_h36m) > 0:
                    pose_2d = keypoints_h36m[0, 0]
                    pose_buffer_2d.append(pose_2d)
        if len(pose_buffer_2d) == args.frames:
            input_3d = np.array(list(pose_buffer_2d))
            input_3d = normalize_screen_coordinates(input_3d, w=frame_width, h=frame_height)
            input_3d = torch.from_numpy(input_3d).float().unsqueeze(0)
            if args.use_gpu:
                input_3d = input_3d.cuda()
            with torch.no_grad():
                pred_3d = model_3d(input_3d)
                pred_3d = pred_3d[0, -1].cpu().numpy()
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            pred_3d = camera_to_world(pred_3d, R=np.array(rot, dtype='float32'), t=0)
            img_b64 = plot3Dpose_flask(pred_3d)
            yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' + base64.b64decode(img_b64) + b'\r\n')
        else:
            time.sleep(0.005)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8505, debug=True)
    finally:
        shared_capture.release()
