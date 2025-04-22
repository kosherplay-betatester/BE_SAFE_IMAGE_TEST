import streamlit as st

# --- Page Config (must be first) ---
st.set_page_config(layout='wide')
st.title('🔍 Image Gender Detector')

import os
import io
import time
import gc
import psutil
from urllib.parse import urljoin, urlparse

import torch
from ultralytics import YOLO
import clip

import requests
from bs4 import BeautifulSoup

from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import cv2
import pynvml

# --- Constants & Configuration ---
LIST_THUMBNAIL_SIZE = (250, 200)
CONFIDENCE_THRESHOLD_PERSON = 0.45

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

# --- Model loading (cached) ---
@st.cache_resource(show_spinner=False)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detection_model = YOLO("yolo11n.pt").to(device)
# CLIP gender classifier: higher-accuracy ViT-L/14@336px
    clf_model, preprocess = clip.load('ViT-L/14@336px', device=device)
    clf_model.to(device)
    categories = ["man", "woman", "object"]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)
    with torch.no_grad():
        text_features = clf_model.encode_text(text_inputs)
    return device, detection_model, clf_model, preprocess, text_features, categories

DEVICE, MODEL_DETECTION, MODEL_CLASSIFICATION, PREPROCESS, TEXT_FEATURES, CATEGORIES = load_models()

# --- Utility functions ---
def get_system_metrics():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    ram_used = mem.used / (1024 ** 3)
    ram_pct = mem.percent
    if GPU_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_pct = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        vram = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used = vram.used / (1024 ** 3)
        vram_total = vram.total / (1024 ** 3)
        vram_pct = (vram.used / vram.total) * 100
    else:
        gpu_pct = vram_used = vram_total = vram_pct = None
    return cpu, ram_used, ram_pct, gpu_pct, vram_used, vram_total, vram_pct

@st.cache_data(show_spinner=False)
def scrape_images_from_url(url, max_images=30):
    images = []
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'image/*'
    }
    session = requests.Session()
    session.headers.update(headers)
    resp = session.get(url, timeout=20)
    ctype = resp.headers.get('content-type', '').lower()
    # Single image URL
    if 'html' not in ctype and 'image' in ctype:
        return [(resp.content, url)]
    # HTML page: scrape up to max_images
    soup = BeautifulSoup(resp.text, 'html.parser')
    seen = set()
    for img in soup.find_all('img'):
        src = img.get('data-src') or img.get('src')
        if not src: continue
        full = urljoin(resp.url, src)
        if full in seen: continue
        seen.add(full)
        try:
            r2 = session.get(full, timeout=15)
            ct2 = r2.headers.get('content-type', '').lower()
            if 'image' in ct2 and len(r2.content) > 2000:
                images.append((r2.content, full))
                if len(images) >= max_images:
                    break
        except:
            pass
    return images

@st.cache_data(show_spinner=False)
def load_directory_images(path):
    imgs = []
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    for fn in os.listdir(path):
        if fn.lower().endswith(exts):
            p = os.path.join(path, fn)
            with open(p, 'rb') as f:
                data = f.read()
            imgs.append((data, fn))
    return imgs

# Gender classification + thumbnail

def classify_and_thumbnail(pil_img, img_cv, yolo_res):
    summary = {'Male': 0, 'Female': 0, 'Else': 0, 'Error': 0}
    proc_img = img_cv.copy()
    crops = []
    coords = []
    for box in yolo_res.boxes:
        if yolo_res.names[int(box.cls)] != 'person': continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        crop = pil_img.crop((x1, y1, x2, y2))
        crops.append(PREPROCESS(crop).to(DEVICE))
        coords.append((x1, y1, x2, y2))

    if crops:
        imgs_t = torch.stack(crops)
        with torch.no_grad():
            feats = MODEL_CLASSIFICATION.encode_image(imgs_t)
            logits = feats @ TEXT_FEATURES.t()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for i, (x1, y1, x2, y2) in enumerate(coords):
            scores = {c: probs[i][j] for j, c in enumerate(CATEGORIES)}
            cat, conf = max(scores.items(), key=lambda x: x[1])
            if cat == 'man' and conf >= 0.5:
                gender, color = 'Male', (255, 100, 0)
            elif cat == 'woman' and conf >= 0.5:
                gender, color = 'Female', (150, 50, 255)
            else:
                gender, color = 'Else', (0, 200, 50)
            summary[gender] += 1
            cv2.rectangle(proc_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(proc_img, gender, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        summary['Else'] += 1

    # Thumbnail + border
    img_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
    thumb = Image.fromarray(img_rgb)
    thumb.thumbnail(LIST_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

    total_people = summary['Male'] + summary['Female']
    if summary['Error'] > 0:
        bcol = BORDER_COLORS['error']
    elif total_people > 0:
        if summary['Male'] > summary['Female']:
            bcol = BORDER_COLORS['male']
        elif summary['Female'] > summary['Male']:
            bcol = BORDER_COLORS['female']
        else:
            bcol = BORDER_COLORS['mixed']
    elif summary['Else'] == 0:
        bcol = BORDER_COLORS['none']
    else:
        bcol = BORDER_COLORS['other']

    return ImageOps.expand(thumb, border=5, fill=bcol), summary

# --- Interface & Processing ---
with st.sidebar:
    st.header('Controls')
    uploaded = st.file_uploader('Upload Image(s)', type=['png','jpg','jpeg','bmp','webp'], accept_multiple_files=True)
    dir_path = st.text_input('Directory Path')
    url_input = st.text_input('URL to scrape')
    threads = st.slider('Threads', 1, os.cpu_count() or 8, min(32, os.cpu_count() or 8))
    batch_size = st.number_input('Batch Size', 1, 64, 64)
    start = st.button('Start Analysis')

if start:
    items = []
    # Gather inputs
    for f in uploaded or []:
        items.append((f.read(), f.name))
    if dir_path:
        if os.path.isdir(dir_path):
            items += load_directory_images(dir_path)
        else:
            st.sidebar.error('Directory not found')
    if url_input:
        scraped = scrape_images_from_url(url_input)
        for data, src in scraped:
            name = os.path.basename(urlparse(src).path) or src
            items.append((data, name))

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

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i in range(0, total, batch_size):
            batch = items[i:i+batch_size]
            pil_imgs, cv_imgs, names = [], [], []
            for data, name in batch:
                img = Image.open(io.BytesIO(data)).convert('RGB')
                pil_imgs.append(img)
                cv_imgs.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                names.append(name)

            try:
                yolo_res = MODEL_DETECTION.predict(pil_imgs, device=DEVICE, conf=CONFIDENCE_THRESHOLD_PERSON)
            except:
                yolo_res = [None] * len(batch)

            futures = []
            for img_pil, img_cv, yres in zip(pil_imgs, cv_imgs, yolo_res):
                if yres is None:
                    futures.append(executor.submit(lambda: (Image.new('RGB', LIST_THUMBNAIL_SIZE, (200,200,200)), {'Error':1})))
                else:
                    futures.append(executor.submit(classify_and_thumbnail, img_pil, img_cv, yres))

            for future, (_, name) in zip(futures, batch):
                thumb, summary = future.result()
                with results:
                    st.image(
                        thumb,
                        caption=(f"{name} — Male: {summary.get('Male',0)}, "
                                 f"Female: {summary.get('Female',0)}, "
                                 f"Other/Error: {summary.get('Else',0)+summary.get('Error',0)}"),
                        use_column_width=False
                    )
                processed += 1
                pb.progress(processed / total)

                cpu, ram_u, ram_p, gpu_p, vram_u, vram_t, vram_p = get_system_metrics()
                elapsed = time.time() - start_time
                ips = processed / elapsed if elapsed > 0 else 0
                gpu_line = (f" • GPU: {gpu_p}% • VRAM: {vram_u:.2f}/{vram_t:.2f} GB ({vram_p:.1f}%)" if GPU_AVAILABLE else "")
                metrics.markdown(
                    f"**Benchmark:** {processed}/{total} items ({total_bytes/1024**2:.2f} MB) — "
                    f"{ips:.2f} img/s in {elapsed:.2f}s  \n"
                    f"CPU: {cpu}% • RAM: {ram_u:.2f} GB ({ram_p}%)" + gpu_line
                )

            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

    if GPU_AVAILABLE:
        pynvml.nvmlShutdown()