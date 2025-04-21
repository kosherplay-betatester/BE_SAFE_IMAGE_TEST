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

# --- Page Config (must be first) ---
st.set_page_config(layout='wide')
st.title('🔍 Image Gender Detector')

# --- Constants & Configuration ---
LIST_THUMBNAIL_SIZE = (250, 200)
CONFIDENCE_THRESHOLD = 0.45

BORDER_COLORS = {
    "male": "#0064FF",
    "female": "#FF3296",
    "mixed": "#FFA500",
    "none": "#808080",
    "error": "#FF0000",
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
    # Detection model
    detection_model = YOLO("yolov8s.pt").to(device)
    # Classification model + preprocessing
    clf_model, preprocess = clip.load("ViT-B/32", device=device)
    categories = ["man", "woman", "object"]
    # Prepare text features for zero-shot classification
    text_inputs = torch.cat([
        clip.tokenize(f"a photo of a {c}") for c in categories
    ]).to(device)
    with torch.no_grad():
        text_features = clf_model.encode_text(text_inputs)
    return device, detection_model, clf_model, preprocess, text_features, categories

DEVICE, MODEL_DETECTION, MODEL_CLF, PREPROCESS, TEXT_FEATURES, CATEGORIES = load_models()

# --- Utility Functions ---

def get_system_metrics():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    ram_used = mem.used / (1024 ** 3)
    ram_pct = mem.percent
    if GPU_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_pct = util.gpu
        vram = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used = vram.used / (1024 ** 3)
        vram_total = vram.total / (1024 ** 3)
        vram_pct = (vram.used / vram.total) * 100
    else:
        gpu_pct = vram_used = vram_total = vram_pct = None
    return cpu, ram_used, ram_pct, gpu_pct, vram_used, vram_total, vram_pct

@st.cache_data(show_spinner=False)
def scrape_images_from_url(url, max_images=30):
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0', 'Accept': 'image/*'})
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except Exception:
        st.sidebar.error(f"Failed to fetch URL: {url}")
        return []
    ctype = resp.headers.get('content-type', '').lower()
    if 'image' in ctype and 'html' not in ctype:
        return [(resp.content, url)]
    soup = BeautifulSoup(resp.text, 'html.parser')
    images, seen = [], set()
    for img in soup.find_all('img'):
        src = img.get('data-src') or img.get('src')
        if not src:
            continue
        full = urljoin(resp.url, src)
        if full in seen:
            continue
        seen.add(full)
        try:
            r2 = session.get(full, timeout=15)
            if 'image' in r2.headers.get('content-type', '') and len(r2.content) > 2000:
                images.append((r2.content, full))
                if len(images) >= max_images:
                    break
        except Exception:
            continue
    return images

@st.cache_data(show_spinner=False)
def load_directory_images(path):
    imgs = []
    for fn in os.listdir(path):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            full = os.path.join(path, fn)
            try:
                with open(full, 'rb') as f:
                    imgs.append((f.read(), fn))
            except Exception:
                continue
    return imgs


def open_image_safe(data, name):
    try:
        pil_img = Image.open(io.BytesIO(data)).convert('RGB')
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return pil_img, cv_img
    except UnidentifiedImageError:
        return None, None


def classify_and_thumbnail(pil_img, img_cv, yolo_res):
    summary = {'Male': 0, 'Female': 0, 'Else': 0}
    proc = img_cv.copy()
    crops, coords = [], []
    for box in yolo_res.boxes:
        if yolo_res.names[int(box.cls)] != 'person':
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
        coord = (x1, y1, x2, y2)
        coords.append(coord)
        crop = pil_img.crop(coord)
        crops.append(PREPROCESS(crop).to(DEVICE))
    if crops:
        batch = torch.stack(crops)
        with torch.no_grad():
            feats = MODEL_CLF.encode_image(batch)
            logits = feats @ TEXT_FEATURES.t()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for i, (x1, y1, x2, y2) in enumerate(coords):
            scores = {cat: probs[i][j] for j, cat in enumerate(CATEGORIES)}
            cat, conf = max(scores.items(), key=lambda x: x[1])
            if cat == 'man' and conf > 0.5:
                gender, color = 'Male', (255, 100, 0)
            elif cat == 'woman' and conf > 0.5:
                gender, color = 'Female', (150, 50, 255)
            else:
                gender, color = 'Else', (0, 200, 50)
            summary[gender] += 1
            cv2.rectangle(proc, (x1, y1), (x2, y2), color, 2)
            cv2.putText(proc, gender, (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    else:
        summary['Else'] = 1
    thumb = Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    thumb.thumbnail(LIST_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    # Determine border color
    total = summary['Male'] + summary['Female']
    if total == 0:
        bcol = BORDER_COLORS['none']
    elif summary['Male'] > summary['Female']:
        bcol = BORDER_COLORS['male']
    elif summary['Female'] > summary['Male']:
        bcol = BORDER_COLORS['female']
    else:
        bcol = BORDER_COLORS['mixed']
    return ImageOps.expand(thumb, border=5, fill=bcol), summary

# --- Sidebar Controls ---
with st.sidebar:
    st.header('Controls')
    uploaded = st.file_uploader('Upload Image(s)', type=['png','jpg','jpeg','bmp','webp'], accept_multiple_files=True)
    dir_path = st.text_input('Directory Path')
    url_input = st.text_input('URL to scrape')
    threads = st.slider('Threads', 1, max(1, os.cpu_count() or 4), value=min(8, os.cpu_count() or 4))
    batch_size = st.number_input('Batch Size', 1, 32, value=8)
    start = st.button('Start Analysis')

if start:
    # Gather inputs
    items = []
    for f in uploaded or []:
        items.append((f.read(), f.name))
    if dir_path and os.path.isdir(dir_path):
        items.extend(load_directory_images(dir_path))
    if url_input:
        items.extend(scrape_images_from_url(url_input))
    if not items:
        st.sidebar.warning('No images to process')
        st.stop()

    total = len(items)
    processed = 0
    total_bytes = sum(len(d) for d, _ in items)
    start_time = time.time()

    pb = st.progress(0)
    metrics = st.empty()
    results = st.container()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for start_idx in range(0, total, batch_size):
            batch = items[start_idx:start_idx+batch_size]
            valid, invalid = [], []
            for data, name in batch:
                pil_img, cv_img = open_image_safe(data, name)
                if pil_img is None:
                    invalid.append(name)
                else:
                    valid.append((pil_img, cv_img, name))
            # Process valid images
            if valid:
                pil_imgs, cv_imgs, names = zip(*valid)
                try:
                    yolo_res = MODEL_DETECTION.predict(source=list(pil_imgs), device=DEVICE, conf=CONFIDENCE_THRESHOLD)
                except Exception:
                    yolo_res = [None] * len(pil_imgs)
                futures = []
                for img_pil, img_cv, res in zip(pil_imgs, cv_imgs, yolo_res):
                    if res is None:
                        futures.append(executor.submit(lambda: (Image.new('RGB', LIST_THUMBNAIL_SIZE, (200,200,200)), {'Else':1})))
                    else:
                        futures.append(executor.submit(classify_and_thumbnail, img_pil, img_cv, res))
                for future, name in zip(futures, names):
                    thumb, summary = future.result()
                    with results:
                        st.image(thumb, caption=f"{name} — {summary}", use_column_width=False)
                    processed += 1
                    pb.progress(processed/total)
                    cpu, ram_u, ram_p, gpu_p, vram_u, vram_t, vram_p = get_system_metrics()
                    elapsed = time.time() - start_time
                    ips = processed/elapsed if elapsed>0 else 0
                    gpu_text = f" • GPU: {gpu_p}% • VRAM: {vram_u:.2f}/{vram_t:.2f} GB ({vram_p:.1f}%)" if GPU_AVAILABLE else ""
                    metrics.markdown(
                        f"**Benchmark:** {processed}/{total} items ({total_bytes/1024**2:.2f} MB) — {ips:.2f} img/s in {elapsed:.2f}s  \n"
                        f"CPU: {cpu}% • RAM: {ram_u:.2f} GB ({ram_p}%)" + gpu_text
                    )
            # Display invalid placeholders
            for name in invalid:
                placeholder = Image.new('RGB', LIST_THUMBNAIL_SIZE, (200,200,200))
                st.image(placeholder, caption=f"{name} — unreadable", use_column_width=False)
                processed += 1
                pb.progress(processed/total)
                cpu, ram_u, ram_p, gpu_p, vram_u, vram_t, vram_p = get_system_metrics()
                elapsed = time.time() - start_time
                ips = processed/elapsed if elapsed>0 else 0
                gpu_text = f" • GPU: {gpu_p}% • VRAM: {vram_u:.2f}/{vram_t:.2f} GB ({vram_p:.1f}%)" if GPU_AVAILABLE else ""
                metrics.markdown(
                    f"**Benchmark:** {processed}/{total} items ({total_bytes/1024**2:.2f} MB) — {ips:.2f} img/s in {elapsed:.2f}s  \n"
                    f"CPU: {cpu}% • RAM: {ram_u:.2f} GB ({ram_p}%)" + gpu_text
                )
            # Cleanup cache
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

    if GPU_AVAILABLE:
        pynvml.nvmlShutdown()
