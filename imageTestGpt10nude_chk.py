#!/usr/bin/env python3
# image_test_app_fast_nsfw.py – 12 Jun 2025
# ============================================================================
# Three-layer image analyser (fast, mixed-precision safe)
#   1) YOLO-v10-s (FP16 on GPU)         – object filter
#   2) OpenCLIP ViT-H/14 (FP16 on GPU)  – user-editable labels
#   3) NudeNet 3  (ONNX)                – nudity check (crop + full frame)
#
# Fixes the “FloatTensor vs HalfTensor” runtime error by always casting
# *every* image batch to the model’s own dtype before calling `encode_image`.
# ============================================================================
#  dependencies  (Python 3.12):
#     pip install -U streamlit ultralytics torch torchvision torchaudio \
#                     open_clip_torch nudenet onnxruntime-gpu numpy psutil \
#                     opencv-python-headless pillow requests bs4 pynvml
# ============================================================================

import os, io, time, gc, hashlib, re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import streamlit as st
import torch, requests, psutil, numpy as np, cv2, pynvml
from PIL import Image, ImageOps, UnidentifiedImageError
from bs4 import BeautifulSoup
from ultralytics import YOLO
import open_clip
from nudenet import NudeDetector

# ─────────────────────────  UI + GPU  ───────────────────────────────
st.set_page_config(page_title="🔍 CLIP + NudeNet", layout="wide")
st.title("🔍 **Fast CLIP + NudeNet Image Classifier**")

try:
    pynvml.nvmlInit()
    NVML = pynvml.nvmlDeviceGetHandleByIndex(0);  GPU_OK = True
except pynvml.NVMLError:
    GPU_OK, NVML = False, None

def sys_stats():
    cpu = psutil.cpu_percent(None)
    ram = psutil.virtual_memory().percent
    if GPU_OK:
        util = pynvml.nvmlDeviceGetUtilizationRates(NVML).gpu
        mem  = pynvml.nvmlDeviceGetMemoryInfo(NVML)
        return cpu, ram, util, mem.used/2**30, mem.total/2**30
    return cpu, ram, None, None, None

# ─────────────────────────  Models  ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo():
    mdl = YOLO("yolov10s.pt")
    if torch.cuda.is_available():
        mdl.model.half().to("cuda")
    return mdl

@st.cache_resource(show_spinner=False)
def load_clip():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=dev
    )
    if dev == "cuda":
        model.half()                    # run weights in FP16
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    return model, preprocess, tokenizer, dev

@st.cache_resource(show_spinner=False)
def load_nude():
    return NudeDetector()               # ONNX; auto GPU if available

YOLO_MODEL                     = load_yolo()
CLIP_MODEL, PREPROC, TOK, DEV  = load_clip()
NUDE_DET                       = load_nude()
MODEL_DTYPE                    = next(CLIP_MODEL.parameters()).dtype

# ─────────────────────────  State  ──────────────────────────────────
st.session_state.setdefault("yolo_list", ["person"])
st.session_state.setdefault("clip_list", ["man", "woman", "object"])

# ─────────────────────────  Helpers  ────────────────────────────────
THUMB = (250, 200)
NUDE_RE = re.compile(r"(EXPOSED|GENITALIA|BUTTOCKS)", re.I)

def rand_col(l):
    base={"man":(255,100,0),"woman":(150,50,255),"NUDITY":(0,0,255)}
    if l in base: return base[l]
    h=hashlib.sha1(l.encode()).hexdigest()
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def safe_img(buf):
    try: return Image.open(io.BytesIO(buf)).convert("RGB")
    except UnidentifiedImageError: return None

@st.cache_data(show_spinner=False)
def scrape(url,max_img=30):
    out,seen=[],set(); s=requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0"})
    try:
        r=s.get(url,timeout=15)
        if "image" in r.headers.get("content-type",""): return[(r.content,url)]
        soup=BeautifulSoup(r.text,"html.parser")
        for tag in soup.find_all("img"):
            src=tag.get("data-src") or tag.get("src"); full=src and urljoin(r.url,src)
            if not full or full in seen: continue; seen.add(full)
            try:
                r2=s.get(full,timeout=10)
                if "image" in r2.headers.get("content-type","").lower() and len(r2.content)>2000:
                    out.append((r2.content,full));  # noqa
                    if len(out)>=max_img: break
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

def encode_prompts(cats):
    tok = TOK([f"a photo of a {c}" for c in cats]).to(DEV)
    with torch.no_grad(): return CLIP_MODEL.encode_text(tok)

# ─────────────────────────  Sidebar  ───────────────────────────────
with st.sidebar:
    st.header("📋 Settings")

    st.subheader("YOLO classes")
    with st.form("yolo_form",clear_on_submit=True):
        add = st.text_input("Add class")
        if st.form_submit_button("＋ Add") and add.strip():
            if add not in st.session_state.yolo_list:
                st.session_state.yolo_list.append(add.strip())
    st.markdown("• " + ", ".join(st.session_state.yolo_list))

    st.subheader("CLIP categories")
    with st.form("clip_form",clear_on_submit=True):
        new = st.text_input("Add label")
        if st.form_submit_button("＋ Add") and new.strip():
            if new not in st.session_state.clip_list:
                st.session_state.clip_list.append(new.strip())
    st.markdown("• " + ", ".join(st.session_state.clip_list))

    st.subheader("NudeNet threshold")
    nude_thr = st.slider("NSFW ≥", 0.10, 0.90, 0.40, 0.01)

    st.subheader("Inputs")
    uploads = st.file_uploader("Upload",["png","jpg","jpeg","bmp","webp"],True)
    dir_p   = st.text_input("Directory")
    url_in  = st.text_input("Scrape URL")
    threads = st.slider("Threads",1,os.cpu_count() or 4, value=os.cpu_count() or 4)
    bsz     = st.number_input("Batch size",1,512,16)
    run     = st.button("🚀 Start")

# ─────────────────────────  Crop evaluator  ────────────────────────
def analyse_batch(pil_list, cv_list, det_list,
                  clip_feats, clip_labels, yolo_set, thr):
    # collect crops across whole mini-batch
    all_crops, meta = [], []           # meta-> (img_idx, (x1,y1,x2,y2))
    for idx,(pil,det) in enumerate(zip(pil_list,det_list)):
        if pil is None or det is None: continue
        for b in det.boxes:
            if det.names[int(b.cls)] not in yolo_set: continue
            x1,y1,x2,y2 = map(int,b.xyxy[0].cpu().numpy())
            meta.append((idx,(x1,y1,x2,y2)))
            crop_tensor = PREPROC(pil.crop((x1,y1,x2,y2))).to(DEV)
            all_crops.append(crop_tensor)

    # run CLIP once
    if all_crops:
        batch_t = torch.stack(all_crops).to(dtype=MODEL_DTYPE)
        with torch.no_grad():
            logits = CLIP_MODEL.encode_image(batch_t) @ clip_feats.T
            probs  = torch.softmax(logits,1).cpu().numpy()
    else:
        probs=[]

    # per-image results
    out_thumbs, out_sums, out_flags = [], [], []
    prob_iter = iter(probs)
    for idx,(pil,cv,det) in enumerate(zip(pil_list,cv_list,det_list)):
        vis = cv.copy()
        summ = defaultdict(int); nude_any=False
        if pil and det:
            for (i,(img_i,box)) in enumerate(meta):
                if img_i!=idx: continue
                p  = next(prob_iter)
                x1,y1,x2,y2 = box
                lbl = clip_labels[int(np.argmax(p))]
                summ[lbl]+=1
                crop_rgb = cv[y1:y2,x1:x2,::-1]
                unsafe = any(NUDE_RE.search(d["class"]) and d["score"]>=thr
                             for d in NUDE_DET.detect(crop_rgb))
                if unsafe:
                    lbl, nude_any = "NUDITY", True
                    summ[lbl]+=1
                cv2.rectangle(vis,(x1,y1),(x2,y2),rand_col(lbl),2)
                cv2.putText(vis,lbl,(x1,max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)

        if not nude_any and pil:
            if any(NUDE_RE.search(d["class"]) and d["score"]>=thr
                   for d in NUDE_DET.detect(cv)):
                nude_any=True; summ["NUDITY"]+=1

        th = Image.fromarray(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))
        th.thumbnail(THUMB,Image.Resampling.LANCZOS)
        th = ImageOps.expand(th,border=4,fill="#F00" if nude_any else "#888")
        out_thumbs.append(th); out_sums.append(summ); out_flags.append(nude_any)
    return out_thumbs,out_sums,out_flags

# ─────────────────────────  Main loop  ──────────────────────────────
if run:
    tasks=[]
    if uploads: tasks += [(f.read(),f.name) for f in uploads]
    if dir_p:
        if os.path.isdir(dir_p): tasks += load_dir(dir_p)
        else: st.sidebar.error("Dir not found")
    if url_in.strip(): tasks += scrape(url_in.strip())
    if not tasks: st.warning("No images."); st.stop()

    clip_feats = encode_prompts(st.session_state.clip_list)
    yolo_set   = set(st.session_state.yolo_list)

    total, done = len(tasks),0; start=time.time()
    prog = st.progress(0.); stat=st.empty(); gallery=st.container()

    with ThreadPoolExecutor(max_workers=threads) as exe:
        for off in range(0,total,bsz):
            batch = tasks[off:off+bsz]
            pil,cv,name=[],[],[]
            for buf,nm in batch:
                im=safe_img(buf); pil.append(im); name.append(nm)
                cv.append(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)
                          if im is not None else np.zeros((*THUMB[::-1],3),np.uint8))
            try:
                dets = YOLO_MODEL.predict(pil, device=DEV, conf=0.25, half=True)
            except: dets=[None]*len(batch)

            thumbs,sums,flags = analyse_batch(pil,cv,dets,
                                              clip_feats,st.session_state.clip_list,
                                              yolo_set,nude_thr)

            for t,s,f,nm in zip(thumbs,sums,flags,name):
                caption = "; ".join(f"{k}:{v}" for k,v in s.items()) or "—"
                if f: caption += "  🚫 NUDITY"
                gallery.image(t,caption=f"{nm}\n{caption}",use_column_width=False)
                done+=1; prog.progress(done/total)
                cpu,ram,gpu,vu,vt = sys_stats(); ips=done/(time.time()-start)
                gline=f" • GPU {gpu}% VRAM {vu:.1f}/{vt:.1f}GB" if GPU_OK else ""
                stat.markdown(f"**{done}/{total}** • {ips:.2f} img/s • CPU {cpu}% • RAM {ram}%{gline}")

            if DEV=="cuda":
                torch.cuda.empty_cache(); gc.collect()

    if GPU_OK: pynvml.nvmlShutdown()
