import streamlit as st
import os
import io
import time
import gc
import psutil
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor

import torch
from ultralytics import YOLO
import clip
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import cv2
import pynvml

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("🔍 Image Gender Detector")

# --- Constants ---
LIST_THUMBNAIL_SIZE = (250, 200)
CONF_THRESHOLD = 0.45
BORDER_COLORS = {
    "male": "#0064FF",
    "female": "#FF3296",
    "mixed": "#FFA500",
    "none": "#808080",
    "other": "#32C800",
}

# Initialize NVML for GPU metrics
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except pynvml.NVMLError:
    GPU_AVAILABLE = False

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # YOLOv11 small for person detection
    yolo_model = YOLO("yolo11s.pt").to(device)
    # CLIP ViT-L/14@336px for high-accuracy classification
    clf_model, preprocess = clip.load("ViT-L/14@336px", device=device)
    categories = ["man", "woman", "object"]
    prompts = [f"a photo of a {c}" for c in categories]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = clf_model.encode_text(text_tokens)
    return device, yolo_model, clf_model, preprocess, text_features, categories

DEVICE, YOLO_MODEL, CLF_MODEL, PREPROCESS, TEXT_FEATURES, CATEGORIES = load_models()

# --- Utility Functions ---

def get_system_metrics():
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    ram_used = mem.used / (1024 ** 3)
    ram_pct = mem.percent
    if GPU_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_pct = util.gpu
        vid = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used = vid.used / (1024 ** 3)
        vram_total = vid.total / (1024 ** 3)
        vram_pct = (vid.used / vid.total) * 100
    else:
        gpu_pct = vram_used = vram_total = vram_pct = None
    return cpu_pct, ram_used, ram_pct, gpu_pct, vram_used, vram_total, vram_pct

@st.cache_data(show_spinner=False)
def scrape_images_from_url(url, max_images=30):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "image/*"})
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except Exception:
        st.sidebar.error(f"Failed to fetch URL: {url}")
        return []
    ctype = resp.headers.get("content-type", "").lower()
    # Single image URL
    if "image" in ctype and "html" not in ctype:
        return [(resp.content, url)]
    soup = BeautifulSoup(resp.text, "html.parser")
    images, seen = [], set()
    for img in soup.find_all("img"):
        src = img.get("data-src") or img.get("src")
        if not src:
            continue
        full_url = urljoin(resp.url, src)
        if full_url in seen:
            continue
        seen.add(full_url)
        try:
            r2 = session.get(full_url, timeout=15)
            ct2 = r2.headers.get("content-type", "").lower()
            if "image" in ct2 and len(r2.content) > 2000:
                images.append((r2.content, full_url))
                if len(images) >= max_images:
                    break
        except:
            continue
    return images

@st.cache_data(show_spinner=False)
def load_directory_images(path):
    files = []
    for fn in os.listdir(path):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            full = os.path.join(path, fn)
            try:
                with open(full, "rb") as f:
                    files.append((f.read(), fn))
            except:
                pass
    return files


def open_image_safe(data, name):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img, cv_img
    except UnidentifiedImageError:
        return None, None


def classify_and_thumbnail(pil_img, cv_img, yolo_res):
    # Crop persons and classify each
    coords, crops = [], []
    for box in yolo_res.boxes:
        if yolo_res.names[int(box.cls)] != "person":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        coords.append((x1, y1, x2, y2))
        crop = pil_img.crop((x1, y1, x2, y2))
        crops.append(PREPROCESS(crop).to(DEVICE))
    # If no person detected, classify the entire image
    if not crops:
        h, w = cv_img.shape[:2]
        coords = [(0, 0, w, h)]
        crops = [PREPROCESS(pil_img).to(DEVICE)]
    batch = torch.stack(crops)
    with torch.no_grad():
        feats = CLF_MODEL.encode_image(batch)
        logits = feats @ TEXT_FEATURES.t()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    # Draw and tally
    summary = {"Male": 0, "Female": 0, "Other": 0}
    proc_img = cv_img.copy()
    for i, (x1, y1, x2, y2) in enumerate(coords):
        scores = {cat: probs[i][j] for j, cat in enumerate(CATEGORIES)}
        cat, conf = max(scores.items(), key=lambda x: x[1])
        if cat == "man" and conf > 0.5:
            label, color = "Male", (255, 100, 0)
            summary["Male"] += 1
        elif cat == "woman" and conf > 0.5:
            label, color = "Female", (150, 50, 255)
            summary["Female"] += 1
        else:
            label, color = "Other", (0, 200, 50)
            summary["Other"] += 1
        cv2.rectangle(proc_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(proc_img, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Create thumbnail with colored border
    thumb = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
    thumb.thumbnail(LIST_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    m = summary["Male"]
    f = summary["Female"]
    if m + f == 0:
        border = BORDER_COLORS["none"]
    elif m > f:
        border = BORDER_COLORS["male"]
    elif f > m:
        border = BORDER_COLORS["female"]
    else:
        border = BORDER_COLORS["mixed"]
    return ImageOps.expand(thumb, border=5, fill=border), summary

# --- Sidebar UI ---
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload Image(s)", type=["png","jpg","jpeg","bmp","webp"], accept_multiple_files=True)
    dir_path = st.text_input("Directory Path")
    url_input = st.text_input("URL to scrape")
    threads = st.slider("Threads", 1, max(1, os.cpu_count() or 4), value=min(8, os.cpu_count() or 4))
    batch_size = st.number_input("Batch Size", 1, 32, value=8)
    start = st.button("Start Analysis")

# --- Main Processing ---
if start:
    items = []
    for file in uploaded or []:
        items.append((file.read(), file.name))
    if dir_path and os.path.isdir(dir_path):
        items.extend(load_directory_images(dir_path))
    if url_input:
        items.extend(scrape_images_from_url(url_input))
    if not items:
        st.sidebar.warning("No images to process")
        st.stop()

    total = len(items)
    processed = 0
    total_bytes = sum(len(d) for d, _ in items)
    start_time = time.time()

    progress_bar = st.progress(0)
    metric_box = st.empty()
    result_container = st.container()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i in range(0, total, batch_size):
            batch = items[i:i+batch_size]
            valid_items, invalid_names = [], []
            for data, name in batch:
                pil_img, cv_img = open_image_safe(data, name)
                if pil_img is None:
                    invalid_names.append(name)
                else:
                    valid_items.append((pil_img, cv_img, name))

            # Process valid images
            if valid_items:
                pil_imgs, cv_imgs, names = zip(*valid_items)
                try:
                    yolo_results = YOLO_MODEL.predict(source=list(pil_imgs), device=DEVICE, conf=CONF_THRESHOLD)
                except Exception:
                    yolo_results = [None] * len(valid_items)
                futures = [executor.submit(classify_and_thumbnail, img, cv, res)
                           for (img, cv, _), res in zip(valid_items, yolo_results)]
                for future, name in zip(futures, names):
                    thumb, summary = future.result()
                    with result_container:
                        st.image(thumb, caption=f"{name} — {summary}")
                    processed += 1
                    progress_bar.progress(processed / total)
                    cpu, ram_u, ram_p, gpu_p, vram_u, vram_t, vram_p = get_system_metrics()
                    elapsed = time.time() - start_time
                    ips = processed / elapsed if elapsed > 0 else 0
                    gpu_text = f" • GPU: {gpu_p}% • VRAM: {vram_u:.2f}/{vram_t:.2f} GB ({vram_p:.1f}%)" if GPU_AVAILABLE else ""
                    metric_box.markdown(
                        f"**Benchmark:** {processed}/{total} items ({total_bytes/1024**2:.2f} MB) — "
                        f"{ips:.2f} img/s in {elapsed:.2f}s  \n"
                        f"CPU: {cpu}% • RAM: {ram_u:.2f} GB ({ram_p}%)" + gpu_text
                    )

            # Display placeholders for invalid images
            for name in invalid_names:
                placeholder = Image.new('RGB', LIST_THUMBNAIL_SIZE, (200, 200, 200))
                st.image(placeholder, caption=f"{name} — unreadable")
                processed += 1
                progress_bar.progress(processed / total)
                cpu, ram_u, ram_p, gpu_p, vram_u, vram_t, vram_p = get_system_metrics()
                elapsed = time.time() - start_time
                ips = processed / elapsed if elapsed > 0 else 0
                gpu_text = f" • GPU: {gpu_p}% • VRAM: {vram_u:.2f}/{vram_t:.2f} GB ({vram_p:.1f}%)" if GPU_AVAILABLE else ""
                metric_box.markdown(
                    f"**Benchmark:** {processed}/{total} items ({total_bytes/1024**2:.2f} MB) — "
                    f"{ips:.2f} img/s in {elapsed:.2f}s  \n"
                    f"CPU: {cpu}% • RAM: {ram_u:.2f} GB ({ram_p}%)" + gpu_text
                )

            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    if GPU_AVAILABLE:
        pynvml.nvmlShutdown()
