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
CONFIDENCE_THRESHOLD_PERSON = 0.45

BORDER_COLORS = {
    "male": "#0064FF",
    "female": "#FF3296",
    "mixed": "#FFA500",
    "none": "#808080",
    "error": "#FF0000",
    "other": "#32C800",
}

# --- GPU Metrics Initialization ---
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except pynvml.NVMLError:
    GPU_AVAILABLE = False
    nvml_handle = None

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # YOLO person detector
    det_model = YOLO("yolo11n.pt")
    det_model.to(device)

    # CLIP gender classifier, ViT-L/14@336px
    clf_model, preprocess = clip.load("ViT-L/14@336px", device=device)
    categories = ["man", "woman", "object"]
    text_tokens = clip.tokenize([f"a photo of a {c}" for c in categories]).to(device)
    with torch.no_grad():
        text_features = clf_model.encode_text(text_tokens)

    return device, det_model, clf_model, preprocess, text_features, categories

DEVICE, MODEL_DET, MODEL_CLF, PREPROCESS, TEXT_FEATURES, CATEGORIES = load_models()

# --- Utility Functions ---
def get_system_metrics():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    ram_used = mem.used / (1024 ** 3)
    ram_pct = mem.percent
    if GPU_AVAILABLE:
        util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        gpu_pct = util.gpu
        vram_used = meminfo.used / (1024 ** 3)
        vram_total = meminfo.total / (1024 ** 3)
        vram_pct = (meminfo.used / meminfo.total) * 100
    else:
        gpu_pct = vram_used = vram_total = vram_pct = None
    return cpu, ram_used, ram_pct, gpu_pct, vram_used, vram_total, vram_pct

@st.cache_data(show_spinner=False)
def scrape_images_from_url(url, max_images=30):
    out = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    try:
        resp = session.get(url, timeout=15)
        ctype = resp.headers.get("content-type", "").lower()
        if "image" in ctype and "html" not in ctype:
            out.append((resp.content, url))
            return out
        soup = BeautifulSoup(resp.text, "html.parser")
        seen = set()
        for img in soup.find_all("img"):
            src = img.get("data-src") or img.get("src")
            if not src:
                continue
            full = urljoin(resp.url, src)
            if full in seen:
                continue
            seen.add(full)
            try:
                r2 = session.get(full, timeout=10)
                if "image" in r2.headers.get("content-type", "").lower() and len(r2.content) > 2000:
                    out.append((r2.content, full))
                    if len(out) >= max_images:
                        break
            except:
                pass
    except:
        pass
    return out

@st.cache_data(show_spinner=False)
def load_directory_images(path):
    imgs = []
    for fn in os.listdir(path):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            full = os.path.join(path, fn)
            try:
                with open(full, "rb") as f:
                    data = f.read()
                imgs.append((data, fn))
            except:
                continue
    return imgs

def safe_open(data):
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        return None

def classify_and_draw(pil_img, raw_cv, yolo_out):
    summary = {"Male": 0, "Female": 0, "Else": 0, "Error": 0}
    proc = raw_cv.copy()
    crops = []
    locs = []
    for b in yolo_out.boxes:
        if yolo_out.names[int(b.cls)] != "person":
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        crops.append(PREPROCESS(pil_img.crop((x1, y1, x2, y2))).to(DEVICE))
        locs.append((x1, y1, x2, y2))

    if crops:
        batch = torch.stack(crops)
        with torch.no_grad():
            feats = MODEL_CLF.encode_image(batch)
            logits = feats @ TEXT_FEATURES.t()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        for i, (x1, y1, x2, y2) in enumerate(locs):
            idx = np.argmax(probs[i])
            conf = probs[i][idx]
            cat = CATEGORIES[idx]
            if cat == "man" and conf >= 0.5:
                gender, color = "Male", (255, 100, 0)
            elif cat == "woman" and conf >= 0.5:
                gender, color = "Female", (150, 50, 255)
            else:
                gender, color = "Else", (0, 200, 50)
            summary[gender] += 1
            cv2.rectangle(proc, (x1, y1), (x2, y2), color, 2)
            cv2.putText(proc, gender, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        summary["Else"] += 1

    thumb = Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    thumb.thumbnail(LIST_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

    total = summary["Male"] + summary["Female"]
    if summary["Error"] > 0:
        bcol = BORDER_COLORS["error"]
    elif total > 0:
        bcol = (
            BORDER_COLORS["male"]
            if summary["Male"] > summary["Female"]
            else BORDER_COLORS["female"]
            if summary["Female"] > summary["Male"]
            else BORDER_COLORS["mixed"]
        )
    elif summary["Else"] == 0:
        bcol = BORDER_COLORS["none"]
    else:
        bcol = BORDER_COLORS["other"]

    return ImageOps.expand(thumb, border=5, fill=bcol), summary

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Inputs")
    uploads = st.file_uploader("Upload Images", type=['png','jpg','jpeg','bmp','webp'], accept_multiple_files=True)
    dir_path = st.text_input("Directory Path")
    url_in = st.text_input("URL to Scrape")
    threads = st.slider("Threads", 1, os.cpu_count() or 4, os.cpu_count() or 4)
    batch_size = st.number_input("Batch Size", 1, 128, 112)
    go = st.button("Start Analysis")

if go:
    items = []
    for f in uploads or []:
        data = f.read()
        items.append((data, f.name))
    if dir_path:
        if os.path.isdir(dir_path):
            items += load_directory_images(dir_path)
        else:
            st.sidebar.error("Directory not found")
    if url_in:
        items += scrape_images_from_url(url_in)

    if not items:
        st.sidebar.warning("No images to process.")
        st.stop()

    total = len(items)
    processed = 0
    total_bytes = sum(len(d) for d, _ in items)
    start_time = time.time()

    progress = st.progress(0)
    stats = st.empty()
    gallery = st.container()

    with ThreadPoolExecutor(max_workers=threads) as exe:
        for i in range(0, total, batch_size):
            batch = items[i : i + batch_size]
            pil_imgs, cv_imgs, names = [], [], []
            for data, name in batch:
                img = safe_open(data)
                if img is None:
                    pil_imgs.append(None)
                    cv_imgs.append(np.zeros((LIST_THUMBNAIL_SIZE[1], LIST_THUMBNAIL_SIZE[0], 3), np.uint8))
                else:
                    pil_imgs.append(img)
                    cv_imgs.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                names.append(name)

            try:
                yolo_outs = MODEL_DET.predict(pil_imgs, device=DEVICE, conf=CONFIDENCE_THRESHOLD_PERSON)
            except:
                yolo_outs = [None] * len(batch)

            futures = []
            for img_pil, img_cv, yres in zip(pil_imgs, cv_imgs, yolo_outs):
                if img_pil is None or yres is None:
                    futures.append(exe.submit(lambda: (
                        Image.new("RGB", LIST_THUMBNAIL_SIZE, (200, 200, 200)),
                        {"Male":0, "Female":0, "Else":0, "Error":1}
                    )))
                else:
                    futures.append(exe.submit(classify_and_draw, img_pil, img_cv, yres))

            for fut, (_, name) in zip(futures, batch):
                thumb, summ = fut.result()
                with gallery:
                    st.image(
                        thumb,
                        caption=f"{name} — Male: {summ['Male']}, Female: {summ['Female']}, Other/Error: {summ['Else'] + summ['Error']}",
                        use_column_width=False
                    )
                processed += 1
                progress.progress(processed / total)

                cpu, ram_u, ram_p, gpu_p, vram_u, vram_t, vram_p = get_system_metrics()
                elapsed = time.time() - start_time
                ips = processed / elapsed if elapsed > 0 else 0
                gpu_line = (
                    f" • GPU: {gpu_p}% • VRAM: {vram_u:.2f}/{vram_t:.2f} GB ({vram_p:.1f}%)"
                    if GPU_AVAILABLE
                    else ""
                )
                stats.markdown(
                    f"**Processed:** {processed}/{total} ({total_bytes/1024**2:.2f} MB) — "
                    f"{ips:.2f} img/s in {elapsed:.1f}s  \n"
                    f"CPU: {cpu}% • RAM: {ram_u:.2f} GB ({ram_p}%)" + gpu_line
                )

            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    if GPU_AVAILABLE:
        pynvml.nvmlShutdown()
