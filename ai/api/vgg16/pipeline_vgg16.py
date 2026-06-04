import os
import shutil
import cv2
import json
import random
import numpy as np
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import tensorflow as tf
import subprocess
import re
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse


HEATMAP_W = 1024
HEATMAP_H = 512
THRESHOLD = 0.40
SCAN_ANGLE_DEG = 45

THICKNESS_COLOR_RANGES = [
    (0,   0,   '#FFFFFF'),
    (1,   3,   '#f5cecf'),
    (4,   6,   '#ffa6b0'),
    (7,   10,  '#fc8b8c'),
    (11,  13,  '#f97d7b'),
    (14,  17,  '#f4605d'),
    (18,  20,  '#fb4548'),
    (21,  23,  '#fc180d'),
    (24,  27,  '#fe3300'),
    (28,  32,  '#f47504'),
    (33,  36,  '#f9d500'),
    (37,  39,  '#fff710'),
    (40,  44,  '#b1fa0c'),
    (45,  50,  '#00fe07'),
    (51,  56,  '#03be41'),
    (57,  62,  '#00a855'),
    (63,  67,  '#098c6a'),
    (68,  74,  '#0546b4'),
    (75,  80,  '#0216e6'),
    (81,  86,  '#0000fa'),
    (87,  92,  '#0000e7'),
    (93,  98,  '#0100dc'),
    (99,  104, '#0300c6'),
    (105, 110, '#00019f'),
    (111, 116, '#00006c'),
    (117, 122, '#000030'),
    (123, 128, '#000022'),
    (129, 134, '#000015'),
    (135, 999, '#000000'),
]


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def release_gpu(model):
    del model
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


def build_cso_colormap():
    color_mapping = [
        (140, '#FFFFFF'),
        (170, '#f5cecf'),
        (200, '#ffa6b0'),
        (230, '#fc8b8c'),
        (260, '#f97d7b'),
        (290, '#f4605d'),
        (320, '#fb4548'),
        (350, '#fc180d'),
        (380, '#fe3300'),
        (410, '#f47504'),
        (440, '#f9d500'),
        (470, '#fff710'),
        (500, '#b1fa0c'),
        (530, '#00fe07'),
        (560, '#03be41'),
        (590, '#00a855'),
        (620, '#098c6a'),
        (650, '#0546b4'),
        (680, '#0216e6'),
        (710, '#0000fa'),
        (740, '#0000e7'),
        (770, '#0100dc'),
        (800, '#0300c6'),
        (830, '#00019f'),
        (860, '#00006c'),
        (890, '#000030'),
        (920, '#000022'),
        (950, '#000015'),
        (980, '#000008'),
    ]
    min_um = color_mapping[0][0]
    max_um = color_mapping[-1][0]
    color_stops = []
    for um, hex_color in color_mapping:
        position = (um - min_um) / (max_um - min_um)
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        color_stops.append((position, (r, g, b)))
    return mcolors.LinearSegmentedColormap.from_list("cso", color_stops)


CSO_COLORMAP = build_cso_colormap()

app = FastAPI()


class VideoPipeline:
    def __init__(self, video_path, direction, output_dir="output"):
        self.video_path = video_path
        self.direction = direction
        self.frames_dir = os.path.join(output_dir, "frames")
        self.bar_masks_dir = os.path.join(output_dir, "barMasks")
        self.output_dir = output_dir

        for d in [self.frames_dir, self.bar_masks_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        for fname in ["heatmapData.npy", "heatmap.png"]:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(self.frames_dir, f"barFrame_{count}.jpg"), frame)
            count += 1
        cap.release()

    def preprocess_img(self, img_bgr, img_size=256):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        return img_resized.astype(np.float32) / 255.0

    def validate_frames(self, model_path, num_samples=3, img_size=256):
        fnames = sorted(
            os.listdir(self.frames_dir),
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        if len(fnames) == 0:
            return False

        samples = random.sample(fnames, min(num_samples, len(fnames)))
        model = tf.keras.models.load_model(model_path, compile=False)

        any_mask_found = False
        for fname in samples:
            img_bgr = cv2.imread(os.path.join(self.frames_dir, fname))
            tensor = self.preprocess_img(img_bgr, img_size)[np.newaxis]
            prob = model.predict(tensor, verbose=0)[0, :, :, 0]
            mask = (prob > 0.5).astype(np.uint8)
            if mask.sum() > 0:
                any_mask_found = True

        release_gpu(model)
        return any_mask_found

    def compute_gray_hist(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def run_segmentation(self, model_path, threshold=0.5, img_size=256, batch_size=8):
        model = tf.keras.models.load_model(model_path, compile=False)

        fnames = sorted(os.listdir(self.frames_dir), key=lambda x: int(x.split("_")[1].split(".")[0]))
        prev_center_x = None
        mean_hist = None
        mean_hist_count = 0

        for batch_start in range(0, len(fnames), batch_size):
            batch_files = fnames[batch_start:batch_start + batch_size]
            orig_sizes = []
            tensors = []
            raw_imgs = []

            for fname in batch_files:
                img_bgr = cv2.imread(os.path.join(self.frames_dir, fname))
                orig_sizes.append(img_bgr.shape[:2])
                tensors.append(self.preprocess_img(img_bgr, img_size))
                raw_imgs.append(img_bgr)

            batch = np.stack(tensors)
            probs = model.predict(batch, verbose=0)[:, :, :, 0]

            for i, fname in enumerate(batch_files):
                idx = fname.split("_")[1].split(".")[0]
                h, w = orig_sizes[i]
                img_bgr = raw_imgs[i]

                frame_hist = self.compute_gray_hist(img_bgr)

                if mean_hist is None:
                    mean_hist = frame_hist.copy()
                    mean_hist_count = 1
                    lighting_rejected = False
                else:
                    dist = cv2.compareHist(mean_hist, frame_hist, cv2.HISTCMP_BHATTACHARYYA)
                    if dist > THRESHOLD:
                        lighting_rejected = True
                    else:
                        lighting_rejected = False
                        mean_hist = (mean_hist * mean_hist_count + frame_hist) / (mean_hist_count + 1)
                        mean_hist_count += 1

                bar_mask = (probs[i] > threshold).astype(np.uint8)
                bar_mask = cv2.resize(bar_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                cols = np.where(bar_mask == 1)[1]
                current_center_x = int(cols.mean()) if len(cols) > 0 else None
                thickness = int(len(np.where(bar_mask == 1)[0]))

                direction_rejected = False
                if current_center_x is not None and prev_center_x is not None:
                    if self.direction == "right" and current_center_x <= prev_center_x:
                        direction_rejected = True
                    elif self.direction == "left" and current_center_x >= prev_center_x:
                        direction_rejected = True

                if not direction_rejected and current_center_x is not None:
                    prev_center_x = current_center_x

                coords = np.argwhere(bar_mask == 1).tolist()
                prefix = "barMask"
                if direction_rejected:
                    prefix += "_x"
                if lighting_rejected:
                    prefix += "_h"

                save_path = os.path.join(self.bar_masks_dir, f"{prefix}_{idx}.json")
                with open(save_path, "w") as f:
                    json.dump({
                        "coordinates": coords,
                        "centerX": current_center_x,
                        "thickness": thickness,
                        "frameIdx": int(idx),
                        "frameWidth": w
                    }, f)

        release_gpu(model)

    def apply_manual_color_ranges(self, smoothed):
        h, w = smoothed.shape
        output = np.zeros((h, w, 3), dtype=np.uint8)
        for low, high, hex_color in THICKNESS_COLOR_RANGES:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            mask = (smoothed >= low) & (smoothed < high)
            output[mask] = [r, g, b]
        return output

    def compute_heatmap(self):
        all_rows = []

        mask_files = sorted(
            [
                f for f in os.listdir(self.bar_masks_dir)
                if f.startswith("barMask_") and "_x_" not in f and "_h_" not in f
            ],
            key=lambda x: int(x.replace("barMask_", "").replace(".json", ""))
        )

        for fname in mask_files:
            with open(os.path.join(self.bar_masks_dir, fname)) as f:
                data = json.load(f)
            coords = data["coordinates"]
            if not coords:
                continue

            coords = np.array(coords)
            h = coords[:, 0].max() + 1
            row_counts = np.zeros(h, dtype=np.float32)
            for r in range(h):
                row_counts[r] = np.sum(coords[:, 0] == r)

            all_rows.append(row_counts)

        if not all_rows:
            return

        max_len = max(len(r) for r in all_rows)
        padded = np.array([np.pad(r, (0, max_len - len(r))) for r in all_rows])

        np.save(os.path.join(self.output_dir, "heatmapData.npy"), padded)

        resized = cv2.resize(padded, (HEATMAP_W, HEATMAP_H), interpolation=cv2.INTER_LINEAR)
        smoothed = gaussian_filter(resized, sigma=(4, 9))

        heatmap_img = self.apply_manual_color_ranges(smoothed)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.output_dir, "heatmap.png"), heatmap_img)


def merge_heatmaps(left2right_heatmap, right2left_heatmap, output_dir="output"):
    left = cv2.imread(left2right_heatmap)
    right = cv2.imread(right2left_heatmap)

    left = cv2.resize(left, (1024, 512))
    right = cv2.resize(right, (1024, 512))

    cv2.putText(left, "Left to Right", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(right, "Right to Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    merged = np.hstack([left, right])
    save_path = os.path.join(output_dir, "mergedHeatmap.png")
    cv2.imwrite(save_path, merged)


def generate_masked_video(frames_dir, bar_masks_dir, output_path, fps=25, crf=28, preset="fast"):
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda x: int(re.search(r"(\d+)", x).group(1))
    )
    if not frame_files:
        return

    mask_index = {}
    for fname in os.listdir(bar_masks_dir):
        if not fname.endswith(".json"):
            continue
        stem = fname.replace(".json", "")
        parts = stem.split("_")
        idx = int(parts[-1])
        is_dir_rejected = "_x_" in fname
        is_hist_rejected = "_h_" in fname
        mask_index[idx] = (os.path.join(bar_masks_dir, fname), is_dir_rejected, is_hist_rejected)

    probe = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    H, W = probe.shape[:2]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "bgr24",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx265",
        "-crf", str(crf),
        "-preset", preset,
        "-tag:v", "hvc1",
        output_path
    ]
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    alpha_mask = 0.45
    alpha_purple = 0.50
    green = np.array([0, 200, 0], dtype=np.float32)
    red = np.array([0, 0, 200], dtype=np.float32)
    purple = np.array([180, 0, 180], dtype=np.float32)

    for fname in frame_files:
        frame_idx = int(re.search(r"(\d+)", fname).group(1))
        img_bgr = cv2.imread(os.path.join(frames_dir, fname))
        if img_bgr is None:
            continue

        canvas = img_bgr.astype(np.float32)

        if frame_idx in mask_index:
            mask_path, is_dir_rejected, is_hist_rejected = mask_index[frame_idx]

            if is_hist_rejected:
                purple_layer = np.full_like(canvas, purple)
                canvas = cv2.addWeighted(canvas, 1.0 - alpha_purple, purple_layer, alpha_purple, 0)
            else:
                with open(mask_path) as f:
                    coords = json.load(f)["coordinates"]

                if coords:
                    coords = np.array(coords)
                    mask_img = np.zeros((H, W), dtype=np.uint8)

                    rows = np.clip(coords[:, 0], 0, H - 1)
                    cols = np.clip(coords[:, 1], 0, W - 1)
                    mask_img[rows, cols] = 1

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask_img = cv2.dilate(mask_img, kernel, iterations=1)

                    colour = red if is_dir_rejected else green
                    colour_layer = np.zeros_like(canvas)
                    colour_layer[mask_img == 1] = colour

                    blended = cv2.addWeighted(canvas, 1.0 - alpha_mask, colour_layer, alpha_mask, 0)
                    mask_bool = mask_img.astype(bool)
                    canvas[mask_bool] = blended[mask_bool]

        frame_out = np.clip(canvas, 0, 255).astype(np.uint8)
        pipe.stdin.write(frame_out.tobytes())

    pipe.stdin.close()
    pipe.wait()


def read_file_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def run_pipeline(video_path, direction, output_dir):
    pipeline = VideoPipeline(video_path=video_path, direction=direction, output_dir=output_dir)
    pipeline.extract_frames()
    if not pipeline.validate_frames(model_path="models/vgg16_unet_cornea.h5"):
        return None
    pipeline.run_segmentation(model_path="models/vgg16_unet_cornea.h5")
    pipeline.compute_heatmap()
    return pipeline


@app.post("/api/Samples/maps")
async def process_samples(
    left2right: UploadFile = File(...),
    right2left: UploadFile = File(...)
):
    os.makedirs("/app/vids", exist_ok=True)
    os.makedirs("/app/output", exist_ok=True)

    lr_video_path = "/app/vids/left2right.mp4"
    rl_video_path = "/app/vids/right2left.mp4"

    with open(lr_video_path, "wb") as f:
        f.write(await left2right.read())

    with open(rl_video_path, "wb") as f:
        f.write(await right2left.read())

    lr_pipeline = run_pipeline(lr_video_path, "right", "/app/output/left2right")
    rl_pipeline = run_pipeline(rl_video_path, "left", "/app/output/right2left")

    lr_video_bytes = None
    lr_heatmap_bytes = None
    rl_video_bytes = None
    rl_heatmap_bytes = None
    full_map_bytes = None

    if lr_pipeline is not None:
        generate_masked_video(
            frames_dir=lr_pipeline.frames_dir,
            bar_masks_dir=lr_pipeline.bar_masks_dir,
            output_path="/app/output/left2right/masked_video.mkv"
        )
        lr_video_bytes = read_file_bytes("/app/output/left2right/masked_video.mkv")
        lr_heatmap_bytes = read_file_bytes("/app/output/left2right/heatmap.png")

    if rl_pipeline is not None:
        generate_masked_video(
            frames_dir=rl_pipeline.frames_dir,
            bar_masks_dir=rl_pipeline.bar_masks_dir,
            output_path="/app/output/right2left/masked_video.mkv"
        )
        rl_video_bytes = read_file_bytes("/app/output/right2left/masked_video.mkv")
        rl_heatmap_bytes = read_file_bytes("/app/output/right2left/heatmap.png")

    if lr_pipeline is not None and rl_pipeline is not None:
        merge_heatmaps(
            left2right_heatmap="/app/output/left2right/heatmap.png",
            right2left_heatmap="/app/output/right2left/heatmap.png",
            output_dir="/app/output"
        )
        full_map_bytes = read_file_bytes("/app/output/mergedHeatmap.png")

    return JSONResponse({
        "trackingVideos": {
            "left2right": base64.b64encode(lr_video_bytes).decode() if lr_video_bytes else None,
            "right2left": base64.b64encode(rl_video_bytes).decode() if rl_video_bytes else None,
        },
        "maps": {
            "left2right": base64.b64encode(lr_heatmap_bytes).decode() if lr_heatmap_bytes else None,
            "right2left": base64.b64encode(rl_heatmap_bytes).decode() if rl_heatmap_bytes else None,
            "fullMap":    base64.b64encode(full_map_bytes).decode()    if full_map_bytes    else None,
        }
    })