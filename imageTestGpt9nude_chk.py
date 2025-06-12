# image_test_app.py  –  FULL DROP-IN REWRITE
# -------------------------------------------------
# Streamlit GUI that lets you add *YOLO* object classes
# and *CLIP* text categories on-the-fly, then analyse
# local / scraped images with the chosen tests.
#
# — fixes “category doesn’t show” by printing the list
#   *after* each form so the update appears immediately
#   on the same rerun.
# — no direct assignment to widget keys → no exceptions
# -------------------------------------------------

import os, io, time, gc, hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import streamlit as st
import psutil, requests, torch, cv2, numpy as np, pynvml
from PIL import Image, ImageOps, UnidentifiedImageError
from bs4 import BeautifulSoup
from ultralytics import YOLO
import clip


# ╭─ Streamlit & GPU bootstrap ────────────────────────────────────────────────╮
st.set_page_config(page_title="🔍 Image Classifier", layout="wide")
st.title("🔍 **Customisable Image Classifier**")

try:
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_OK = True
except pynvml.NVMLError:
    GPU_OK = False
    GPU_HANDLE = None


# ╭─ Models (cached) ───────────────────────────────────────────────────────────╮
@st.cache_resource(show_spinner=False)
def load_yolo():
    m = YOLO("yolo11n.pt")
    m.to("cuda" if torch.cuda.is_available() else "cpu")
    return m

@st.cache_resource(show_spinner=False)
def load_clip():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=dev)
    return model, preprocess, dev

YOLO_MODEL = load_yolo()
CLIP_MODEL, PREPROC, DEVICE = load_clip()


# ╭─ Session-state defaults ────────────────────────────────────────────────────╮
if "yolo_list" not in st.session_state:
    st.session_state.yolo_list = ["person"]
if "clip_list" not in st.session_state:
    st.session_state.clip_list = ["man", "woman", "object"]


# ╭─ Helpers ───────────────────────────────────────────────────────────────────╮
LIST_THUMBNAIL = (250, 200)

def sys_metrics():
    cpu = psutil.cpu_percent(None)
    mem = psutil.virtual_memory()
    if GPU_OK:
        util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE)
        memi = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
        return cpu, mem.percent, util.gpu, memi.used/2**30, memi.total/2**30
    return cpu, mem.percent, None, None, None

def rand_color(label: str):
    base = {"man": (255,100,0), "woman": (150,50,255)}
    if label in base: return base[label]
    h = hashlib.sha1(label.encode()).hexdigest()
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def safe_open(buf: bytes):
    try:
        return Image.open(io.BytesIO(buf)).convert("RGB")
    except UnidentifiedImageError:
        return None

@st.cache_data(show_spinner=False)
def scrape_site(url, max_img=30):
    out, seen = [], set()
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    try:
        r = s.get(url, timeout=15)
        if "image" in r.headers.get("content-type", ""):
            out.append((r.content, url))
            return out
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup.find_all("img"):
            src = tag.get("data-src") or tag.get("src")
            if not src: continue
            full = urljoin(r.url, src)
            if full in seen: continue
            seen.add(full)
            try:
                r2 = s.get(full, timeout=10)
                if "image" in r2.headers.get("content-type", "") and len(r2.content) > 2_000:
                    out.append((r2.content, full))
                    if len(out) >= max_img:
                        break
            except: pass
    except: pass
    return out

@st.cache_data(show_spinner=False)
def load_directory(path):
    imgs=[]
    for fn in os.listdir(path):
        if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp")):
            try:
                imgs.append((open(os.path.join(path, fn),"rb").read(), fn))
            except: pass
    return imgs

def encode_clip_text(categories):
    tokens = clip.tokenize([f"a photo of a {c}" for c in categories]).to(DEVICE)
    with torch.no_grad():
        return CLIP_MODEL.encode_text(tokens)


def analyse(pil_img, cv_img, yolo_pred, text_feats, clip_cats, yolo_cats):
    summary = defaultdict(int)
    vis = cv_img.copy()
    locs, crops = [], []

    for box in yolo_pred.boxes:
        cls_name = yolo_pred.names[int(box.cls)]
        if cls_name not in yolo_cats: continue
        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
        locs.append((x1,y1,x2,y2))
        crops.append(PREPROC(pil_img.crop((x1,y1,x2,y2))).to(DEVICE))

    if crops:
        with torch.no_grad():
            feats = CLIP_MODEL.encode_image(torch.stack(crops))
            probs = torch.softmax(feats @ text_feats.T, 1).cpu().numpy()

        for (x1,y1,x2,y2), prob in zip(locs, probs):
            label = clip_cats[int(np.argmax(prob))]
            summary[label] += 1
            cv2.rectangle(vis,(x1,y1),(x2,y2), rand_color(label), 2)
            cv2.putText(vis, label, (x1,max(0,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)

    thumb = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    thumb.thumbnail(LIST_THUMBNAIL, Image.Resampling.LANCZOS)
    return ImageOps.expand(thumb, border=4, fill="#888"), summary


# ╭─ Sidebar UI ────────────────────────────────────────────────────────────────╮
with st.sidebar:
    st.header("📋 Choose tests")

    # ‣ YOLO classes -----------------------------------------------------------
    st.subheader("YOLO classes")
    with st.form("add_yolo_form", clear_on_submit=True):
        new_yolo = st.text_input("Add YOLO class")
        if st.form_submit_button("＋ Add"):
            if new_yolo.strip() and new_yolo.strip() not in st.session_state.yolo_list:
                st.session_state.yolo_list.append(new_yolo.strip())
    st.markdown("🔸 **Current YOLO tests:** " + ", ".join(st.session_state.yolo_list))

    # ‣ CLIP categories --------------------------------------------------------
    st.subheader("CLIP categories")
    with st.form("add_clip_form", clear_on_submit=True):
        new_clip = st.text_input("Add CLIP category")
        if st.form_submit_button("＋ Add"):
            if new_clip.strip() and new_clip.strip() not in st.session_state.clip_list:
                st.session_state.clip_list.append(new_clip.strip())
    st.markdown("🔸 **Current CLIP tests:** " + ", ".join(st.session_state.clip_list))

    # ‣ I/O controls -----------------------------------------------------------
    st.subheader("Processing")
    uploads  = st.file_uploader("Upload images", ["png","jpg","jpeg","bmp","webp"], accept_multiple_files=True)
    dir_path = st.text_input("Directory path")
    url_src  = st.text_input("Scrape URL")
    threads  = st.slider("Threads", 1, os.cpu_count() or 4, os.cpu_count() or 4)
    batch_sz = st.number_input("Batch size", 1, 512, 16)
    run_btn  = st.button("🚀 Start")


# ╭─ Main workflow ─────────────────────────────────────────────────────────────╮
if run_btn:
    tasks=[]
    if uploads:
        tasks += [(f.read(), f.name) for f in uploads]
    if dir_path:
        if os.path.isdir(dir_path):
            tasks += load_directory(dir_path)
        else:
            st.sidebar.error("Directory not found")
    if url_src.strip():
        tasks += scrape_site(url_src.strip())

    if not tasks:
        st.warning("No images to process.")
        st.stop()

    clip_cats = st.session_state.clip_list
    yolo_cats = st.session_state.yolo_list
    TEXT_FEATS = encode_clip_text(clip_cats)

    total, done = len(tasks), 0
    t0 = time.time()
    prog  = st.progress(0.0)
    stats = st.empty()
    gallery = st.container()

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for offset in range(0, total, batch_sz):
            chunk = tasks[offset:offset+batch_sz]

            pil_imgs, cv_imgs, names = [], [], []
            for buf, name in chunk:
                im = safe_open(buf)
                pil_imgs.append(im)
                cv_imgs.append(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                               if im is not None else np.zeros((*LIST_THUMBNAIL[::-1],3), np.uint8))
                names.append(name)

            try:
                yolo_outs = YOLO_MODEL.predict(pil_imgs, device=DEVICE, conf=0.25)
            except:
                yolo_outs = [None] * len(chunk)

            futs=[]
            for p_img, c_img, y_pred in zip(pil_imgs, cv_imgs, yolo_outs):
                if p_img is None or y_pred is None:
                    futs.append(ex.submit(lambda: (Image.new("RGB",LIST_THUMBNAIL,(200,200,200)),
                                                   defaultdict(int))))
                else:
                    futs.append(ex.submit(analyse, p_img, c_img, y_pred,
                                          TEXT_FEATS, clip_cats, yolo_cats))

            for fut, nm in zip(futs, names):
                thumb, summ = fut.result()
                with gallery:
                    st.image(thumb, caption=f"{nm} — " +
                             ", ".join(f"{k}:{v}" for k,v in summ.items()),
                             use_column_width=False)
                done += 1
                prog.progress(done/total)

                cpu, ram, gpu, v_used, v_tot = sys_metrics()
                ips = done / (time.time() - t0)
                gpu_line = f" • GPU {gpu}%  VRAM {v_used:.1f}/{v_tot:.1f} GB" if GPU_OK else ""
                stats.markdown(f"**{done}/{total}** • {ips:.2f} img/s • "
                               f"CPU {cpu}% • RAM {ram}%{gpu_line}")

            if DEVICE == "cuda":
                torch.cuda.empty_cache(); gc.collect()

    if GPU_OK:
        pynvml.nvmlShutdown()
