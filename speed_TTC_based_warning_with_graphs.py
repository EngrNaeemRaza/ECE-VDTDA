import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install pandas and matplotlib to generate graphs:")
    print("pip install pandas matplotlib")
    sys.exit(1)

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import select_device, time_sync
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

alert_dict = {}

def plot_graphs(save_dir, frame_nums, pre_process_times, inference_times, post_process_times, deepsort_times, fps_values, avg_speed_values, avg_ttc_values):
    """Generates and saves a single PNG file with all performance graphs."""
    
    graphs_dir = save_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        'frame': frame_nums,
        'pre_time_ms': pre_process_times,
        'inf_time_ms': inference_times,
        'post_time_ms': post_process_times,
        'deepsort_time_ms': deepsort_times,
        'fps': fps_values,
        'avg_speed_kmh': avg_speed_values,
        'avg_ttc_s': avg_ttc_values
    }
    
    df = pd.DataFrame(data)

    # --- Create a single figure with subplots ---
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))

    # Plot 1: Performance Metrics
    axes[0].plot(df['frame'], df['pre_time_ms'], label='Pre-processing Time (ms)')
    axes[0].plot(df['frame'], df['inf_time_ms'], label='Inference Time (ms)')
    axes[0].plot(df['frame'], df['post_time_ms'], label='Post-processing Time (ms)')
    axes[0].plot(df['frame'], df['deepsort_time_ms'], label='DeepSORT Update Time (ms)')
    axes[0].set_xlabel('Frame Number')
    axes[0].set_ylabel('Time (milliseconds)')
    axes[0].set_title('Performance Metrics Over Time')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: FPS over Time
    axes[1].plot(df['frame'], df['fps'], label='FPS', color='green')
    axes[1].set_xlabel('Frame Number')
    axes[1].set_ylabel('Frames per Second (FPS)')
    axes[1].set_title('Frames per Second (FPS) Over Time')
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Average Speed and TTC Over Time
    axes[2].plot(df['frame'], df['avg_speed_kmh'], label='Average Speed (km/h)', color='blue')
    axes[2].plot(df['frame'], df['avg_ttc_s'], label='Average TTC (s)', color='orange')
    axes[2].set_xlabel('Frame Number')
    axes[2].set_ylabel('Value')
    axes[2].set_title('Average Speed and TTC Over Time')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(graphs_dir / 'all_metrics_and_graphs.png')
    plt.close(fig)

    print(f"✅ All graphs saved to a single file at {graphs_dir / 'all_metrics_and_graphs.png'}")

# -------------------------
# Metric color thresholds
# -------------------------
def get_speed_color(speed_kmh):
    if speed_kmh < 53:
        return (255, 255, 255)         # white
    elif speed_kmh <= 60:
        return (0, 255, 255)           # yellow
    else:
        return (0, 0, 255)             # red

def get_ttc_color(speed_kmh):
    if speed_kmh <= 52:
        return (255, 255, 255)         # white
    elif speed_kmh <= 60:
        return (0, 255, 255)           # yellow
    else:
        return (0, 0, 255)             # red

# -------------------------
# Color Legend Drawer
# -------------------------
def draw_color_legend(im0, show_speed=False, show_ttc=False):
    x, y = 10, 50
    h_step = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1

    if show_speed or show_ttc:
        cv2.putText(im0, "Speed (km/h) / TTC (s):", (x, y), font, scale, (255, 255, 255), thickness)
        cv2.rectangle(im0, (x + 170, y - 10), (x + 190, y + 10), (255, 255, 255), -1)
        cv2.putText(im0, "<=52 / >150s", (x + 195, y), font, scale, (255, 255, 255), thickness)
        cv2.rectangle(im0, (x + 310, y - 10), (x + 330, y + 10), (0, 255, 255), -1)
        cv2.putText(im0, "53-60 / 15s", (x + 335, y), font, scale, (0, 255, 255), thickness)
        cv2.rectangle(im0, (x + 420, y - 10), (x + 440, y + 10), (0, 0, 255), -1)
        cv2.putText(im0, ">60 / <1.5s", (x + 445, y), font, scale, (0, 0, 255), thickness)
        y += h_step
        
# -------------------------
# Main detection + tracking
# -------------------------
@torch.no_grad()
def run(
        weights='best.pt',
        source='DrivinginFoggyConditions_2.mp4',
        data='data/coco128.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        project='runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=2,
        half=False,
        dnn=False,
        config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml",
        show_speed=False,
        show_distance=False,
        show_ttc=False,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Data loader
    webcam = source.isnumeric() or source.endswith('.txt')
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # DeepSORT init
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Output dir
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Video writer
    vid_writer = None
    output_path = str(save_dir / "output.mp4")

    # In-memory storage for graphs
    frame_nums = []
    pre_process_times = []
    inference_times = []
    post_process_times = []
    deepsort_times = []
    fps_values = []
    avg_speed_values = []
    avg_ttc_values = []
    frame_count = 0

    for path, im, im0s, vid_cap, s in dataset:
        frame_start_time = time.time()
        
        # Pre-processing
        pre_process_start = time.time()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pre_process_time = (time.time() - pre_process_start) * 1000

        # Inference
        t1 = time_sync()
        pred = model(im, augment=augment, visualize=False)
        t2 = time_sync()
        inference_time = (t2 - t1) * 1000

        # Post-processing
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t3 = time_sync()
        post_process_time = (t3 - t2) * 1000

        deepsort_time = 0 
        speeds = []
        ttcs = []

        for i, det in enumerate(pred):
            im0 = im0s.copy() if not webcam else im0s[i].copy()
            im0 = cv2.resize(im0, (1280, 720))
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                deepsort_start_time = time.time()
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]

                outputs = deepsort.update(xywhs, confs, im0)
                deepsort_end_time = time.time()
                deepsort_time = (deepsort_end_time - deepsort_start_time) * 1000

                if len(outputs) > 0:
                    for j, (out, conf) in enumerate(zip(outputs, confs)):
                        x1, y1, x2, y2 = map(int, out[:4])
                        track_id = int(out[4])

                        # Simplified speed calculation and TTC with factors
                        bbox_height = y2 - y1
                        
                        speed_kmh = 50 + 5 * (bbox_height / 100)
                        
                        warning_text = ""
                        warning_color = (255, 255, 255)
                        ttc_factor = 0

                        if speed_kmh <= 52:
                            warning_text = "Safe"
                            warning_color = (255, 255, 255)
                            ttc_factor = 150
                        elif speed_kmh <= 60:
                            warning_text = "Attention Required"
                            warning_color = (0, 255, 255)
                            ttc_factor = 15
                        else:
                            warning_text = "Imminent Collision Alert"
                            warning_color = (0, 0, 255)
                            ttc_factor = 1.5

                        fixed_distance = 30
                        speed_mps = speed_kmh / 3.6
                        
                        ttc = float('inf')
                        if speed_mps > 0:
                            ttc = (fixed_distance / speed_mps) * ttc_factor

                        speeds.append(speed_kmh)
                        ttcs.append(ttc)
                        
                        if show_speed or show_ttc:
                            combined_text = f"Speed = {int(speed_kmh)} km/h, TTC = {ttc:.1f}s, {warning_text}"
                            cv2.putText(im0, combined_text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, warning_color, 2)

                        annotator.box_label([x1, y1, x2, y2], f"ID {track_id}", color=(255, 0, 0))

            else:
                deepsort.increment_ages()

        # Calculate average speed and TTC for the frame
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        avg_ttc = sum(ttcs) / len(ttcs) if ttcs else 0
        
        frame_count += 1
        fps = 1.0 / (time.time() - frame_start_time)

        # Store metrics in lists
        frame_nums.append(frame_count)
        pre_process_times.append(pre_process_time)
        inference_times.append(inference_time)
        post_process_times.append(post_process_time)
        deepsort_times.append(deepsort_time)
        fps_values.append(fps)
        avg_speed_values.append(avg_speed)
        avg_ttc_values.append(avg_ttc)

        im0 = annotator.result()
        draw_color_legend(im0, show_speed=show_speed, show_ttc=show_ttc)

        if view_img:
            cv2.imshow(str(path), im0)
            if cv2.waitKey(1) == ord('q'):
                raise StopIteration

        if save_img:
            if vid_writer is None:
                fps_cap = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                h, w = im0.shape[:2]
                vid_writer = cv2.VideoWriter(output_path,
                                             cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps_cap, (w, h))
            vid_writer.write(im0)

    if vid_writer:
        vid_writer.release()
        print(f"✅ Video saved at {output_path}")

    # Generate graphs after processing all frames
    plot_graphs(save_dir, frame_nums, pre_process_times, inference_times, post_process_times, deepsort_times, fps_values, avg_speed_values, avg_ttc_values)

# -------------------------
# Argument parser
# -------------------------
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt')
    parser.add_argument('--source', type=str, default='DrivinginFoggyConditions_2.mp4')
    parser.add_argument('--data', type=str, default='data/coco128.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='')
    parser.add_argument('--view-img', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-thickness', type=int, default=2)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--config_deepsort', type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument('--show-speed', action='store_true')
    parser.add_argument('--show-distance', action='store_true')
    parser.add_argument('--show-ttc', action='store_true')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)