#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# image_test_app_nsfw.py  –  FULL REWRITE (2025-06-12)
# ----------------------------------------------------
# Three-layer image analyser:
#   1) YOLOv10-s            – fast person / object filter
#   2) OpenCLIP ViT-bigG/14 – dynamic free-text labels
#   3) NudeDetector 3.x     – always-on nudity scan
#
# Red frame + “🚫 NUDITY” whenever any crop or full frame
# exceeds the NSFW threshold you set in the sidebar.
#
# Tested on Python 3.12 with:
#   pip install -U streamlit ultralytics torch torchvision torchaudio
#   pip install -U open_clip_torch nudenet onnxruntime numpy psutil \
#                  opencv-python-headless pillow requests bs4 pynvml
# ----------------------------------------------------

import os, io, time, gc, hashlib, re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import streamlit as st
import requests, psutil, torch, numpy as np, cv2, pynvml
from PIL import Image, ImageOps, UnidentifiedImageError
from bs4 import BeautifulSoup
from ultralytics import YOLO
import open_clip                                  # ⇦ OpenCLIP
from nudenet import NudeDetector                  # ⇦ NudeNet 3.x

# ──────────────────────────  Streamlit setup  ────────────────────────────────
st.set_page_config(page_title="🔍 Image Classifier + Nudity", layout="wide")
st.title("🔍 **Customisable Image Classifier + Nudity Detector**")

# ──────────────────────────  GPU diagnostics   ───────────────────────────────
try:
    pynvml.nvmlInit()
    GPU_OK  = True
    GPU_HND = pynvml.nvmlDeviceGetHandleByIndex(0)
except pynvml.NVMLError:
    GPU_OK  = False
    GPU_HND = None

def sys_metrics():
    cpu = psutil.cpu_percent(None)
    mem = psutil.virtual_memory()
    if GPU_OK:
        util  = pynvml.nvmlDeviceGetUtilizationRates(GPU_HND)
        mem_i = pynvml.nvmlDeviceGetMemoryInfo(GPU_HND)
        return cpu, mem.percent, util.gpu, mem_i.used/2**30, mem_i.total/2**30
    return cpu, mem.percent, None, None, None

# ──────────────────────────  Cached models  ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo():
    # 47 MB YOLOv10-s checkpoint
    return YOLO("yolov10s.pt").to("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_clip():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=dev
    )
    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
    return model, preprocess, tokenizer, dev

@st.cache_resource(show_spinner=False)
def load_nudedet():
    return NudeDetector()                        # downloads ONNX weights first run

YOLO_MODEL                     = load_yolo()
CLIP_MODEL, PREPROC, TOK, DEV  = load_clip()
NUDE_DET                       = load_nudedet()

# ──────────────────────────  Session defaults  ───────────────────────────────
if "yolo_list" not in st.session_state:
    st.session_state.yolo_list = ["person"]
if "clip_list" not in st.session_state:
    st.session_state.clip_list = ["man", "woman", "object"]

# ──────────────────────────  Utility helpers  ────────────────────────────────
LIST_THUMB = (250, 200)
NUDE_RE    = re.compile(r"(EXPOSED|GENITALIA|BUTTOCKS)", re.I)

def rand_color(lbl):
    base = {"man": (255,100,0), "woman": (150,50,255), "NUDITY": (0,0,255)}
    if lbl in base: return base[lbl]
    h = hashlib.sha1(lbl.encode()).hexdigest()
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def safe_open(buf):
    try:
        return Image.open(io.BytesIO(buf)).convert("RGB")
    except UnidentifiedImageError:
        return None

@st.cache_data(show_spinner=False)
def scrape(url, max_images=30):
    out, seen = [], set()
    s = requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0"})
    try:
        r = s.get(url, timeout=15)
        if "image" in r.headers.get("content-type",""):
            return [(r.content, url)]
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup.find_all("img"):
            src = tag.get("data-src") or tag.get("src")
            src = src and urljoin(r.url, src)
            if not src or src in seen: continue
            seen.add(src)
            try:
                r2 = s.get(src, timeout=10)
                if "image" in r2.headers.get("content-type","") and len(r2.content)>2000:
                    out.append((r2.content, src))
                    if len(out) >= max_images: break
            except: pass
    except: pass
    return out

@st.cache_data(show_spinner=False)
def load_dir(path):
    imgs=[]
    for fn in os.listdir(path):
        if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp")):
            try: imgs.append((open(os.path.join(path,fn),"rb").read(), fn))
            except: pass
    return imgs

def encode_text(cats):
    tokens = TOK([f"a photo of a {c}" for c in cats]).to(DEV)
    with torch.no_grad(): return CLIP_MODEL.encode_text(tokens)

# ──────────────────────────  Crop pipeline  ──────────────────────────────────
def analyse(pil_img, cv_img, y_pred, txt_feats, clip_cats,
            yolo_list, nude_thr):

    summary = defaultdict(int)
    vis = cv_img.copy()
    locs, crops = [], []

    # 1) YOLO filter
    for b in y_pred.boxes:
        cls = y_pred.names[int(b.cls)]
        if cls not in yolo_list: continue
        x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
        locs.append((x1,y1,x2,y2))
        crops.append(PREPROC(pil_img.crop((x1,y1,x2,y2))).to(DEV))

    # 2) CLIP + nudity per crop
    nude_flag = False
    if crops:
        with torch.no_grad():
            feats  = CLIP_MODEL.encode_image(torch.stack(crops))
            logits = feats @ txt_feats.T
            probs  = torch.softmax(logits,1).detach().cpu().numpy()

        for (x1,y1,x2,y2), p, crop in zip(locs, probs, crops):
            lbl = clip_cats[int(np.argmax(p))]
            summary[lbl] += 1

            # NudeDetector on crop
            crop_np = (crop.detach().cpu().permute(1,2,0).numpy()*255).astype("uint8")[:,:,::-1]
            unsafe  = any(NUDE_RE.search(d["class"]) and d["score"]>=nude_thr
                          for d in NUDE_DET.detect(crop_np))
            if unsafe:
                lbl, nude_flag = "NUDITY", True
                summary[lbl] += 1

            cv2.rectangle(vis,(x1,y1),(x2,y2), rand_color(lbl), 2)
            cv2.putText(vis,lbl,(x1,max(0,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)

    # 3) Full-frame nudity fallback
    if not nude_flag:
        unsafe_full = any(NUDE_RE.search(d["class"]) and d["score"]>=nude_thr
                          for d in NUDE_DET.detect(cv_img))
        if unsafe_full:
            nude_flag = True
            summary["NUDITY"] += 1

    # 4) Thumbnail
    thumb = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    thumb.thumbnail(LIST_THUMB, Image.Resampling.LANCZOS)
    border = "#FF0000" if nude_flag else "#888"
    return ImageOps.expand(thumb, border=4, fill=border), summary, nude_flag

# ──────────────────────────  Sidebar UI  ──────────────────────────────────────
with st.sidebar:
    st.header("📋 Tests")

    # YOLO list
    st.subheader("YOLO classes")
    with st.form("add_yolo", clear_on_submit=True):
        y_in = st.text_input("Add YOLO class")
        if st.form_submit_button("＋ Add") and y_in.strip():
            if y_in.strip() not in st.session_state.yolo_list:
                st.session_state.yolo_list.append(y_in.strip())
    st.markdown("**Current:** " + ", ".join(st.session_state.yolo_list))

    # CLIP list
    st.subheader("CLIP categories")
    with st.form("add_clip", clear_on_submit=True):
        c_in = st.text_input("Add CLIP category")
        if st.form_submit_button("＋ Add") and c_in.strip():
            if c_in.strip() not in st.session_state.clip_list:
                st.session_state.clip_list.append(c_in.strip())
    st.markdown("**Current:** " + ", ".join(st.session_state.clip_list))

    # Nudity threshold
    st.subheader("Nudity detector")
    nude_thr = st.slider("Score threshold", 0.10, 0.90, 0.40, 0.01)

    # Inputs
    st.subheader("Inputs")
    uploads  = st.file_uploader("Upload images",["png","jpg","jpeg","bmp","webp"], True)
    dir_path = st.text_input("Directory")
    url_src  = st.text_input("Scrape URL")
    threads  = st.slider("Threads",1,os.cpu_count() or 4,os.cpu_count() or 4)
    batch_sz = st.number_input("Batch size",1,512,16)
    run_btn  = st.button("🚀 Start")

# ──────────────────────────  Main loop  ───────────────────────────────────────
if run_btn:
    jobs=[]
    if uploads: jobs += [(f.read(), f.name) for f in uploads]
    if dir_path:
        if os.path.isdir(dir_path): jobs += load_dir(dir_path)
        else: st.sidebar.error("Directory not found")
    if url_src.strip(): jobs += scrape(url_src.strip())

    if not jobs:
        st.warning("No images to process."); st.stop()

    clip_cats = st.session_state.clip_list
    yolo_list = st.session_state.yolo_list
    TXT_FEATS = encode_text(clip_cats)

    total, done = len(jobs), 0
    t0 = time.time()
    prog  = st.progress(0.0)
    stats = st.empty()
    gallery = st.container()

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for off in range(0,total,batch_sz):
            chunk = jobs[off:off+batch_sz]
            pil, cv, names = [], [], []
            for buf,nm in chunk:
                im = safe_open(buf)
                pil.append(im)
                cv.append(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)
                          if im else np.zeros((*LIST_THUMB[::-1],3),np.uint8))
                names.append(nm)

            try:
                preds = YOLO_MODEL.predict(pil, device=DEV, conf=0.25)
            except: preds = [None]*len(chunk)

            futs=[]
            for p_img,c_img,pd in zip(pil,cv,preds):
                if not (p_img and pd):
                    futs.append(ex.submit(lambda: (
                        Image.new("RGB",LIST_THUMB,(200,200,200)),
                        defaultdict(int), False)))
                else:
                    futs.append(ex.submit(analyse, p_img, c_img, pd,
                                          TXT_FEATS, clip_cats, yolo_list, nude_thr))

            for fut,nm in zip(futs,names):
                thumb, summ, nude = fut.result()
                cap = ", ".join(f"{k}:{v}" for k,v in summ.items())
                if nude: cap += " 🚫 NUDITY"
                with gallery:
                    st.image(thumb, caption=f"{nm} — {cap}", use_column_width=False)

                done += 1; prog.progress(done/total)

                cpu, ram, gpu, vu, vt = sys_metrics()
                ips = done / (time.time()-t0)
                gpu_line = f" • GPU {gpu}%  VRAM {vu:.1f}/{vt:.1f}GB" if GPU_OK else ""
                stats.markdown(f"**{done}/{total}** • {ips:.2f} img/s • "
                               f"CPU {cpu}% • RAM {ram}%{gpu_line}")

            if DEV=="cuda":
                torch.cuda.empty_cache(); gc.collect()

    if GPU_OK: pynvml.nvmlShutdown()
