import sys
import os
import io
import time
import gc
import psutil
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures

import torch
from ultralytics import YOLO
import clip

import requests
from bs4 import BeautifulSoup

from PIL import Image, ImageOps, UnidentifiedImageError, ImageDraw
# from PIL.ImageQt import ImageQt # May require Pillow[qt] or PySide
import numpy as np
import cv2
import pynvml

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QSlider, QSpinBox,
    QFileDialog, QProgressBar, QScrollArea, QGridLayout, QMessageBox,
    QGroupBox, QCheckBox, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont

# --- Constants ---
LIST_THUMBNAIL_SIZE = (250, 200)
CONFIDENCE_THRESHOLD_PERSON = 0.45
# GENDER_CONFIDENCE_THRESHOLD_SLIDER is set by UI

BORDER_COLORS = {
    "male": "#0064FF",
    "female": "#FF3296",
    "mixed": "#FFA500",
    "none": "#808080",
    "no_person_detected": "#C0C0C0",
    "error": "#FF0000",
    "other": "#32C800",
}

# --- GPU Metrics Initialization ---
GPU_AVAILABLE = False
nvml_handle = None
try:
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        GPU_AVAILABLE = True
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    else:
        print("No CUDA GPU found. Models on CPU.")
except pynvml.NVMLError as e:
    print(f"NVIDIA ML library issue: {e}. CPU mode.")
    GPU_AVAILABLE = False
except Exception as e:
    print(f"Error initializing NVIDIA ML library: {e}. CPU mode.")
    GPU_AVAILABLE = False

# --- Model Storage (Global or in App Class) ---
MODELS_LOADED = False
DEVICE_STR = "cpu"
MODEL_DET = None
MODEL_CLF = None
PREPROCESS = None
NORMALIZED_TEXT_FEATURES = None
LOGIT_SCALE = None
CATEGORIES_SIMPLE = None

# --- Utility Functions (largely unchanged, st.toast replaced) ---
def show_toast(message, icon="✅", duration=3000, parent_widget=None):
    # In PyQt, a status bar message is a good equivalent for st.toast
    if parent_widget and hasattr(parent_widget, 'statusBar'):
        parent_widget.statusBar().showMessage(f"{icon} {message}", duration)
    else:
        print(f"TOAST: {icon} {message}")


def get_system_metrics():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    ram_used = mem.used / (1024 ** 3); ram_pct = mem.percent
    gpu_pct = vram_used = vram_total = vram_pct = "N/A"
    if GPU_AVAILABLE and nvml_handle:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            gpu_pct = util.gpu; vram_used = meminfo.used/(1024**3); vram_total = meminfo.total/(1024**3)
            vram_pct = (meminfo.used/meminfo.total)*100 if meminfo.total > 0 else 0
        except pynvml.NVMLError: pass
    return cpu, ram_used, ram_pct, gpu_pct, vram_used, vram_total, vram_pct

# @st.cache_data - PyQt doesn't have this directly, simple Python dict cache or skip for now
def scrape_images_from_url(_url, max_images=50, toast_emitter=None):
    out_items = []; session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
    try:
        resp = session.get(_url, timeout=20, allow_redirects=True); resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        if "image" in content_type and "html" not in content_type:
            if len(resp.content) > 2000:
                 out_items.append({"data": resp.content, "name": os.path.basename(urlparse(_url).path) or _url,
                                   "source_type": "scrape_direct_image", "source_page_url": _url, "image_url": _url})
            return out_items
        soup = BeautifulSoup(resp.text, "html.parser"); seen_image_urls = set()
        for img_tag in soup.find_all("img"):
            src = img_tag.get("data-src") or img_tag.get("src")
            if not src or src.startswith("data:image"): continue
            full_image_url = urljoin(resp.url, src)
            if full_image_url in seen_image_urls: continue
            seen_image_urls.add(full_image_url)
            try:
                img_head_resp = session.head(full_image_url, timeout=5, allow_redirects=True); img_head_resp.raise_for_status()
                img_content_type = img_head_resp.headers.get("content-type", "").lower()
                img_content_length = int(img_head_resp.headers.get("content-length", 0))
                if "image" in img_content_type and (img_content_length == 0 or img_content_length > 2000):
                    if img_content_length > 0 and img_content_length < 2000: continue
                    img_resp = session.get(full_image_url, timeout=10, stream=True); img_resp.raise_for_status()
                    final_img_content_type = img_resp.headers.get("content-type", "").lower()
                    if "image" not in final_img_content_type: continue
                    img_data = img_resp.content
                    if len(img_data) > 2000:
                        out_items.append({"data": img_data, "name": os.path.basename(urlparse(full_image_url).path) or full_image_url,
                                          "source_type": "scrape_html", "source_page_url": _url, "image_url": full_image_url})
                        if len(out_items) >= max_images: break
            except: pass # Silently ignore errors for individual images
    except Exception as e:
        if toast_emitter: toast_emitter.emit(f"Error processing URL {_url}: {e}", "🔥")
        else: print(f"Error processing URL {_url}: {e}")
    return out_items

def load_directory_images(path, toast_emitter=None):
    imgs_info = []
    if not os.path.isdir(path):
        if toast_emitter: toast_emitter.emit(f"Directory not found: {path}", "🔥")
        else: print(f"Directory not found: {path}")
        return imgs_info
    for fn in os.listdir(path):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            full_path = os.path.join(path, fn)
            try:
                with open(full_path, "rb") as f: data = f.read()
                if len(data) > 1000:
                    imgs_info.append({"data": data, "name": fn, "source_type": "directory",
                                      "source_page_url": None, "image_url": full_path})
            except Exception as e:
                if toast_emitter: toast_emitter.emit(f"Error loading {fn}: {e}", "🔥")
                else: print(f"Error loading {fn}: {e}")
    return imgs_info

def safe_open_image(data):
    try: return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError: return None
    except Exception: return None

def pil_to_qpixmap(pil_img):
    """Convert PIL Image to QPixmap."""
    if pil_img.mode == "RGB":
        r, g, b = pil_img.split()
        pil_img = Image.merge("RGB", (b, g, r)) # BGR for QImage format
    elif pil_img.mode == "RGBA":
        r, g, b, a = pil_img.split()
        pil_img = Image.merge("RGBA", (b, g, r, a))

    if pil_img.mode == "RGBA":
        img_data = pil_img.tobytes("raw", "BGRA")
        qimage = QImage(img_data, pil_img.width, pil_img.height, QImage.Format_ARGB32)
    else: # RGB
        img_data = pil_img.tobytes("raw", "BGR")
        qimage = QImage(img_data, pil_img.width, pil_img.height, QImage.Format_RGB888)
    
    return QPixmap.fromImage(qimage)


def create_no_person_thumbnail(pil_img):
    thumb = pil_img.copy()
    thumb.thumbnail(LIST_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    expanded_thumb = ImageOps.expand(thumb, border=5, fill=BORDER_COLORS["no_person_detected"])
    summary = {"Male": 0, "Female": 0, "Else": 0, "Error": 0, "NoPerson": 1}
    return pil_to_qpixmap(expanded_thumb), summary, 0.0

def create_error_thumbnail(item_info, error_message="Err"):
    thumb = Image.new("RGB", LIST_THUMBNAIL_SIZE, (220, 220, 220))
    try:
        draw = ImageDraw.Draw(thumb); draw.text((10, 10), error_message[:30], fill=(0,0,0))
        if item_info and "name" in item_info: draw.text((10,30),item_info["name"][:30],fill=(50,50,50))
    except: pass
    expanded_thumb = ImageOps.expand(thumb, border=5, fill=BORDER_COLORS["error"])
    summary = {"Male": 0, "Female": 0, "Else": 0, "Error": 1, "NoPerson": 0}
    return pil_to_qpixmap(expanded_thumb), summary, 0.0

def draw_and_summarize_single_image(raw_cv_bgr_copy, gender_loc_predictions_list):
    summary = {"Male": 0, "Female": 0, "Else": 0, "Error": 0, "NoPerson": 0}
    proc_img = raw_cv_bgr_copy; draw_start_time = time.time()
    if gender_loc_predictions_list:
        for (gender_info, (x1, y1, x2, y2)) in gender_loc_predictions_list:
            gender_str, color_bgr_hex = gender_info # color_bgr_hex is actually a tuple (B,G,R)
            
            # Convert BORDER_COLORS hex to BGR tuple if needed, or pass tuple directly
            color_bgr = color_bgr_hex # Assuming color_bgr_hex is already (B,G,R)
            
            if gender_str in summary: summary[gender_str] += 1
            else: summary["Else"] +=1
            cv2.rectangle(proc_img, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(proc_img, gender_str, (x1,max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(proc_img, gender_str, (x1,max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX,0.7,color_bgr,1,cv2.LINE_AA)

    thumb_pil = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
    thumb_pil.thumbnail(LIST_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    
    total_classified_gender = summary["Male"] + summary["Female"]
    if total_classified_gender > 0:
        bcol = BORDER_COLORS["male"] if summary["Male"] > summary["Female"] else \
               BORDER_COLORS["female"] if summary["Female"] > summary["Male"] else BORDER_COLORS["mixed"]
    elif summary["Else"] > 0 : bcol = BORDER_COLORS["other"]
    elif summary["NoPerson"] > 0: bcol = BORDER_COLORS["no_person_detected"]
    else: bcol = BORDER_COLORS["none"]
    expanded_thumb_pil = ImageOps.expand(thumb_pil, border=5, fill=bcol)
    qpixmap_thumb = pil_to_qpixmap(expanded_thumb_pil)
    return qpixmap_thumb, summary, (time.time() - draw_start_time)

def send_result_to_server(item_payload, endpoint_url, reporting_session):
    if not endpoint_url or not reporting_session: return False
    try:
        response = reporting_session.post(endpoint_url, json=item_payload, timeout=10); response.raise_for_status()
        return True
    except: return False

# --- Worker Thread for Processing ---
class Worker(QThread):
    progress = pyqtSignal(int)
    image_processed = pyqtSignal(QPixmap, str, dict) # pixmap, caption, summary
    metrics_update = pyqtSignal(str, str, str) # processed_line, system_line, component_times_line
    finished = pyqtSignal(str) # Final message
    toast = pyqtSignal(str, str) # message, icon

    def __init__(self, all_items, settings):
        super().__init__()
        self.all_items_to_process = all_items
        self.settings = settings
        self.running = True

    def run(self):
        global DEVICE_STR, MODEL_DET, MODEL_CLF, PREPROCESS, NORMALIZED_TEXT_FEATURES, LOGIT_SCALE, CATEGORIES_SIMPLE

        if not MODELS_LOADED:
            self.finished.emit("Models not loaded. Cannot start analysis.")
            return

        current_gender_confidence_threshold = self.settings['gender_conf_threshold']
        yolo_batch_size = self.settings['yolo_batch_size']
        threads = self.settings['threads']
        server_endpoint = self.settings['server_endpoint']
        enable_reporting = self.settings['enable_reporting']

        total_images_to_process = len(self.all_items_to_process)
        if total_images_to_process == 0:
            self.finished.emit("No images to process.")
            return
        
        self.toast.emit(f"Starting analysis for {total_images_to_process} images.", "⏳")

        processed_count = 0
        total_initial_bytes = sum(len(item["data"]) for item in self.all_items_to_process)
        start_time_global = time.time()
        total_yolo_time, total_clip_time, total_draw_time = 0,0,0
        yolo_batches_run, clip_batches_run = 0,0

        reporting_session = requests.Session() if enable_reporting and server_endpoint else None
        reporting_executor = ThreadPoolExecutor(max_workers=threads) if reporting_session else None

        gallery_col_idx = 0 # Used for assigning to columns in main GUI if needed

        for i_batch_start in range(0, total_images_to_process, yolo_batch_size):
            if not self.running: break
            batch_items_info = self.all_items_to_process[i_batch_start : i_batch_start + yolo_batch_size]
            pil_imgs_batch, cv_imgs_batch, valid_items_info_batch = [], [], []
            
            for item_info in batch_items_info:
                if not self.running: break
                pil_img = safe_open_image(item_info["data"])
                if pil_img:
                    pil_imgs_batch.append(pil_img)
                    cv_imgs_batch.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
                    valid_items_info_batch.append(item_info)
                else:
                    err_thumb_pixmap, err_sum, _ = create_error_thumbnail(item_info, "Load Fail")
                    cap = f"{item_info.get('name','Unk')[:25]} (LoadErr)"
                    self.image_processed.emit(err_thumb_pixmap, cap, err_sum)
                    processed_count += 1
                    if reporting_executor:
                        payload = {**{k:v for k,v in item_info.items() if k!='data'}, "summary":err_sum, "status":"load_error", "timestamp":time.time()}
                        reporting_executor.submit(send_result_to_server, payload, server_endpoint, reporting_session)
            if not self.running: break
            
            if not pil_imgs_batch:
                if total_images_to_process > 0: self.progress.emit(int((processed_count / total_images_to_process) * 100))
                continue

            yolo_s = time.time()
            try:
                yolo_results_list = MODEL_DET.predict(pil_imgs_batch, device=DEVICE_STR, conf=CONFIDENCE_THRESHOLD_PERSON, verbose=False)
                yolo_batches_run += 1
            except Exception as e:
                self.toast.emit(f"YOLO batch error: {e}", "🔥")
                yolo_results_list = [None] * len(pil_imgs_batch)
            total_yolo_time += (time.time() - yolo_s)

            all_crops_for_clip = []; crop_to_pil_idx_map = []; crop_locations_batch = []
            image_had_persons_yolo = [False] * len(pil_imgs_batch)

            for idx_in_batch, yolo_result_single_img in enumerate(yolo_results_list):
                if not self.running: break
                if yolo_result_single_img is None or not hasattr(yolo_result_single_img, 'boxes'):
                    continue 
                
                current_pil_img = pil_imgs_batch[idx_in_batch]
                for box in yolo_result_single_img.boxes:
                    cls_id = int(box.cls)
                    if cls_id < len(yolo_result_single_img.names) and \
                       yolo_result_single_img.names[cls_id] == "person" and \
                       box.conf >= CONFIDENCE_THRESHOLD_PERSON:
                        image_had_persons_yolo[idx_in_batch] = True
                        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                        if x2 > x1 and y2 > y1:
                            crop = current_pil_img.crop((x1,y1,x2,y2))
                            if crop.width > 0 and crop.height > 0:
                                all_crops_for_clip.append(PREPROCESS(crop))
                                crop_to_pil_idx_map.append(idx_in_batch)
                                crop_locations_batch.append((x1,y1,x2,y2))
            if not self.running: break

            clip_predictions_for_all_valid_crops = []
            clip_s = time.time()
            if all_crops_for_clip:
                clip_batches_run += 1
                try:
                    stacked_crops_tensor = torch.stack(all_crops_for_clip).to(DEVICE_STR)
                    with torch.no_grad():
                        img_feat_unnorm = MODEL_CLF.encode_image(stacked_crops_tensor)
                        img_feat_norm = img_feat_unnorm / img_feat_unnorm.norm(dim=-1,keepdim=True)
                        logit_s_val = LOGIT_SCALE.clone().exp() # Use .clone() if LOGIT_SCALE is a tensor
                        logits = logit_s_val * img_feat_norm @ NORMALIZED_TEXT_FEATURES.t()
                        probs = logits.softmax(dim=-1).cpu().numpy()
                    for crop_prob in probs:
                        best_idx = np.argmax(crop_prob)
                        conf = crop_prob[best_idx]; cat_clip = CATEGORIES_SIMPLE[best_idx]
                        # Use BGR tuples for OpenCV drawing
                        if cat_clip=="man" and conf>=current_gender_confidence_threshold: clip_predictions_for_all_valid_crops.append(("Male",(255,100,0))) # Blue BGR
                        elif cat_clip=="woman" and conf>=current_gender_confidence_threshold: clip_predictions_for_all_valid_crops.append(("Female",(150,50,255))) # Pink BGR
                        else: clip_predictions_for_all_valid_crops.append(("Else",(0,200,50))) # Green BGR
                except Exception as e: self.toast.emit(f"CLIP batch error: {e}", "🔥")
            total_clip_time += (time.time() - clip_s)
            
            # Drawing can be slow, consider if it needs its own small thread pool inside this worker
            # For now, doing it sequentially in the worker.
            for idx_pil_in_batch in range(len(pil_imgs_batch)):
                if not self.running: break
                item_info_current_img = valid_items_info_batch[idx_pil_in_batch]
                cv_img_current_img = cv_imgs_batch[idx_pil_in_batch]
                pil_img_current_img = pil_imgs_batch[idx_pil_in_batch]
                thumb_pixmap, summary_data, single_draw_time = None, None, 0

                try:
                    if yolo_results_list[idx_pil_in_batch] is None:
                        thumb_pixmap, summary_data, _ = create_error_thumbnail(item_info_current_img, "YOLO Fail")
                    elif not image_had_persons_yolo[idx_pil_in_batch]:
                        thumb_pixmap, summary_data, _ = create_no_person_thumbnail(pil_img_current_img)
                    else:
                        gender_locs_for_this_image = []
                        for k_crop_global in range(len(crop_to_pil_idx_map)):
                            if crop_to_pil_idx_map[k_crop_global] == idx_pil_in_batch:
                                if k_crop_global < len(clip_predictions_for_all_valid_crops):
                                    gender_pred_tuple = clip_predictions_for_all_valid_crops[k_crop_global]
                                    loc = crop_locations_batch[k_crop_global]
                                    gender_locs_for_this_image.append((gender_pred_tuple, loc))
                        thumb_pixmap, summary_data, single_draw_time = draw_and_summarize_single_image(cv_img_current_img.copy(), gender_locs_for_this_image)
                    total_draw_time += single_draw_time
                except Exception as e:
                    self.toast.emit(f"Draw/Summarize error for {item_info_current_img['name']}: {e}", "🔥")
                    thumb_pixmap, summary_data, _ = create_error_thumbnail(item_info_current_img, "Draw Fail")
                    if summary_data: summary_data["Error"] = 1
                    else: summary_data = {"Male": 0, "Female": 0, "Else": 0, "Error": 1, "NoPerson": 0}


                cap_parts = [f"{item_info_current_img['name'][:25]}..." if len(item_info_current_img['name']) > 28 else item_info_current_img['name']]
                if summary_data["Male"]: cap_parts.append(f"M:{summary_data['Male']}")
                if summary_data["Female"]: cap_parts.append(f"F:{summary_data['Female']}")
                if summary_data["Else"]: cap_parts.append(f"P:{summary_data['Else']}")
                if summary_data["NoPerson"]: cap_parts.append("No Persons")
                if summary_data["Error"]: cap_parts.append("ERR")
                
                self.image_processed.emit(thumb_pixmap, " - ".join(cap_parts), summary_data)
                processed_count += 1

                if reporting_executor:
                    payload = {"image_name":item_info_current_img["name"], "image_url":item_info_current_img["image_url"],
                               "source_page_url":item_info_current_img.get("source_page_url"), "source_type":item_info_current_img["source_type"],
                               "summary":summary_data, "timestamp":time.time(),
                               "status":"processed_ok" if not summary_data["Error"] else "processed_error"}
                    reporting_executor.submit(send_result_to_server, payload, server_endpoint, reporting_session)
                
                if total_images_to_process > 0: self.progress.emit(int((processed_count / total_images_to_process) * 100))
                
                elapsed_g = time.time()-start_time_global; current_ips = processed_count/elapsed_g if elapsed_g>0 else 0
                cpu,ram_u,ram_p,gpu_p,vram_u,vram_t,vram_p = get_system_metrics()
                gpu_l = f"• GPU:{gpu_p}%•VRAM:{vram_u:.2f}/{vram_t:.2f}GB({vram_p:.1f}%)" if GPU_AVAILABLE and isinstance(gpu_p,(int,float)) else ""
                
                processed_line = f"Processed: {processed_count}/{total_images_to_process} ({total_initial_bytes/(1024**2):.2f}MB) — Speed: {current_ips:.2f}img/s in {elapsed_g:.1f}s"
                system_line = f"Sys: CPU {cpu}% | RAM {ram_u:.2f}GB({ram_p}%) {gpu_l}"
                component_times_line = (f"Avg Times(s): YOLO:{total_yolo_time/yolo_batches_run if yolo_batches_run>0 else 0:.3f} "
                                        f"| CLIP:{total_clip_time/clip_batches_run if clip_batches_run>0 else 0:.3f} "
                                        f"| Draw:{total_draw_time/processed_count if processed_count>0 else 0:.3f}")
                self.metrics_update.emit(processed_line, system_line, component_times_line)
            if not self.running: break

            if DEVICE_STR == "cuda": torch.cuda.empty_cache(); gc.collect()

        if reporting_executor:
            self.toast.emit("Finishing server reports...", "⏳")
            reporting_executor.shutdown(wait=True)
        
        elapsed_final = time.time()-start_time_global
        final_ips = processed_count/elapsed_final if elapsed_final > 0 else 0
        
        final_msg = f"Analysis Complete! Processed {processed_count} images.\n"
        final_msg += f"Overall: {processed_count}/{total_images_to_process} imgs | Total Time: {elapsed_final:.2f}s | Avg Speed: {final_ips:.2f}img/s\n"
        final_msg += (f"Avg Component Times(s): YOLO:{total_yolo_time/yolo_batches_run if yolo_batches_run>0 else 0:.3f} "
                      f"| CLIP:{total_clip_time/clip_batches_run if clip_batches_run>0 else 0:.3f} "
                      f"| Draw:{total_draw_time/processed_count if processed_count>0 else 0:.3f}")
        self.finished.emit(final_msg)

    def stop(self):
        self.running = False

class ImageGenderDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Advanced Image Gender Detector & Performance Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        self.all_items_to_process = []
        self.gallery_items_count = 0
        self.worker_thread = None

        self._load_ai_models() # Load models on startup

        self.init_ui()

    def _load_ai_models(self):
        global MODELS_LOADED, DEVICE_STR, MODEL_DET, MODEL_CLF, PREPROCESS, NORMALIZED_TEXT_FEATURES, LOGIT_SCALE, CATEGORIES_SIMPLE
        
        if MODELS_LOADED: return True

        self.statusBar().showMessage("Loading AI Models...")
        QApplication.processEvents() # Allow GUI to update

        DEVICE_STR = "cuda" if GPU_AVAILABLE and torch.cuda.is_available() else "cpu"
        print(f"Models loading on: {DEVICE_STR.upper()}")
        
        categories_instance = ["a photo of a man", "a photo of a woman", "a photo of an object"]
        simple_categories_instance = ["man", "woman", "object"]
        
        try:
            MODEL_DET = YOLO("yolo11n.pt") # Ensure this model file exists
            MODEL_DET.to(DEVICE_STR)
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", f"YOLO load error (yolo11n.pt): {e}\nMake sure 'yolo11n.pt' is in the current directory or path.")
            self.statusBar().showMessage("YOLO model loading failed.", 5000)
            return False
        
        try:
            MODEL_CLF, PREPROCESS = clip.load("ViT-L/14@336px", device=DEVICE_STR)
            text_tokens = clip.tokenize(categories_instance).to(DEVICE_STR)
            with torch.no_grad():
                unnormalized_text_features = MODEL_CLF.encode_text(text_tokens)
                NORMALIZED_TEXT_FEATURES = unnormalized_text_features / unnormalized_text_features.norm(dim=-1, keepdim=True)
                LOGIT_SCALE = MODEL_CLF.logit_scale.detach() # Already a scalar tensor
            CATEGORIES_SIMPLE = simple_categories_instance
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", f"CLIP load error (ViT-L/14@336px): {e}")
            self.statusBar().showMessage("CLIP model loading failed.", 5000)
            return False

        MODELS_LOADED = True
        self.statusBar().showMessage("AI Models loaded successfully.", 5000)
        if hasattr(self, 'start_button'): # Check if UI is initialized
             self.start_button.setEnabled(True)
        return True


    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Controls Panel ---
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)
        controls_group.setMaximumWidth(350)

        # Inputs
        inputs_group = QGroupBox("Inputs")
        inputs_layout = QVBoxLayout()
        inputs_group.setLayout(inputs_layout)
        
        inputs_layout.addWidget(QLabel("Enter URLs (one per line):"))
        self.url_list_input = QTextEdit()
        self.url_list_input.setFixedHeight(100)
        inputs_layout.addWidget(self.url_list_input)

        dir_layout = QHBoxLayout()
        self.dir_path_input = QLineEdit()
        self.dir_path_input.setPlaceholderText("Local Directory Path")
        dir_button = QPushButton("Browse Dir")
        dir_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_path_input)
        dir_layout.addWidget(dir_button)
        inputs_layout.addLayout(dir_layout)
        
        upload_button = QPushButton("Upload Images")
        upload_button.clicked.connect(self.upload_files)
        inputs_layout.addWidget(upload_button)
        controls_layout.addWidget(inputs_group)

        # Processing Settings
        settings_group = QGroupBox("Processing Settings")
        settings_layout = QGridLayout() # Use QGridLayout for better alignment
        settings_group.setLayout(settings_layout)

        settings_layout.addWidget(QLabel("Max Imgs/URL:"), 0, 0)
        self.max_images_slider = QSlider(Qt.Horizontal)
        self.max_images_slider.setRange(1, 100)
        self.max_images_slider.setValue(30)
        self.max_images_label = QLabel("30")
        self.max_images_slider.valueChanged.connect(lambda v: self.max_images_label.setText(str(v)))
        settings_layout.addWidget(self.max_images_slider, 0, 1)
        settings_layout.addWidget(self.max_images_label, 0, 2)
        
        default_threads = os.cpu_count() or 4
        settings_layout.addWidget(QLabel("CPU Tasks:"), 1, 0)
        self.threads_slider = QSlider(Qt.Horizontal)
        self.threads_slider.setRange(1, default_threads * 2)
        self.threads_slider.setValue(default_threads)
        self.threads_label = QLabel(str(default_threads))
        self.threads_slider.valueChanged.connect(lambda v: self.threads_label.setText(str(v)))
        settings_layout.addWidget(self.threads_slider, 1, 1)
        settings_layout.addWidget(self.threads_label, 1, 2)

        settings_layout.addWidget(QLabel("YOLO Batch Size:"), 2, 0)
        self.yolo_batch_spinbox = QSpinBox()
        self.yolo_batch_spinbox.setRange(1, 64)
        self.yolo_batch_spinbox.setValue(8 if GPU_AVAILABLE else 4)
        settings_layout.addWidget(self.yolo_batch_spinbox, 2, 1, 1, 2) # Span 2 columns

        settings_layout.addWidget(QLabel("Gallery Cols:"), 3, 0)
        self.gallery_cols_slider = QSlider(Qt.Horizontal)
        self.gallery_cols_slider.setRange(1, 8)
        self.gallery_cols_slider.setValue(4)
        self.gallery_cols_label = QLabel("4")
        self.gallery_cols_slider.valueChanged.connect(lambda v: self.gallery_cols_label.setText(str(v)))
        self.gallery_cols_slider.valueChanged.connect(self.update_gallery_layout) # Update layout on change
        settings_layout.addWidget(self.gallery_cols_slider, 3, 1)
        settings_layout.addWidget(self.gallery_cols_label, 3, 2)

        settings_layout.addWidget(QLabel("Gender Conf. Thr:"), 4, 0)
        self.gender_conf_slider = QSlider(Qt.Horizontal)
        self.gender_conf_slider.setRange(10, 99) # Represent as 0.10 to 0.99
        self.gender_conf_slider.setValue(50)
        self.gender_conf_label = QLabel("0.50")
        self.gender_conf_slider.valueChanged.connect(lambda v: self.gender_conf_label.setText(f"{v/100:.2f}"))
        settings_layout.addWidget(self.gender_conf_slider, 4, 1)
        settings_layout.addWidget(self.gender_conf_label, 4, 2)
        controls_layout.addWidget(settings_group)

        # Reporting
        reporting_group = QGroupBox("Optional: Report Image Results")
        reporting_layout = QVBoxLayout()
        reporting_group.setLayout(reporting_layout)
        self.enable_reporting_checkbox = QCheckBox("Enable Reporting")
        self.server_endpoint_input = QLineEdit("http://localhost:8000/report")
        self.server_endpoint_input.setEnabled(False)
        self.enable_reporting_checkbox.stateChanged.connect(
            lambda state: self.server_endpoint_input.setEnabled(state == Qt.Checked)
        )
        reporting_layout.addWidget(self.enable_reporting_checkbox)
        reporting_layout.addWidget(self.server_endpoint_input)
        controls_layout.addWidget(reporting_group)

        self.start_button = QPushButton("🚀 Start Analysis")
        self.start_button.setFixedHeight(40)
        font = self.start_button.font(); font.setPointSize(12); self.start_button.setFont(font)
        self.start_button.clicked.connect(self.start_analysis)
        self.start_button.setEnabled(MODELS_LOADED) # Disabled until models are loaded
        controls_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("🛑 Stop Analysis")
        self.stop_button.setFixedHeight(40)
        self.stop_button.setFont(font) # Reuse font
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)


        controls_layout.addStretch(1) # Pushes everything up
        main_layout.addWidget(controls_group)

        # --- Results Area (Right Panel) ---
        results_area = QVBoxLayout()
        
        # Metrics Display
        self.metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()
        self.metrics_group.setLayout(metrics_layout)
        self.metrics_processed_label = QLabel("Processed: 0/0")
        self.metrics_system_label = QLabel("Sys: CPU N/A% | RAM N/A GB(N/A%)")
        self.metrics_components_label = QLabel("Avg Times(s): YOLO:N/A | CLIP:N/A | Draw:N/A")
        metrics_layout.addWidget(self.metrics_processed_label)
        metrics_layout.addWidget(self.metrics_system_label)
        metrics_layout.addWidget(self.metrics_components_label)
        results_area.addWidget(self.metrics_group)

        # Progress Bar
        self.progress_bar = QProgressBar()
        results_area.addWidget(self.progress_bar)

        # Gallery
        self.gallery_scroll_area = QScrollArea()
        self.gallery_scroll_area.setWidgetResizable(True)
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_scroll_area.setWidget(self.gallery_widget)
        results_area.addWidget(self.gallery_scroll_area)
        
        main_layout.addLayout(results_area)

        # Status Bar
        self.setStatusBar(QStatusBar(self))
        if GPU_AVAILABLE:
            self.statusBar().showMessage(f"NVIDIA GPU Detected. Models using CUDA: {DEVICE_STR.upper()}", 5000)
        else:
            self.statusBar().showMessage(f"No CUDA GPU. Models using CPU: {DEVICE_STR.upper()}", 5000)


    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.dir_path_input.setText(dir_path)

    def upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)")
        if files:
            for file_path in files:
                try:
                    with open(file_path, "rb") as f:
                        data = f.read()
                    if len(data) > 1000: # Basic sanity check
                        self.all_items_to_process.append({
                            "data": data, 
                            "name": os.path.basename(file_path), 
                            "source_type": "upload",
                            "source_page_url": None,
                            "image_url": file_path
                        })
                except Exception as e:
                    self.statusBar().showMessage(f"Error loading {os.path.basename(file_path)}: {e}", 3000)
            self.statusBar().showMessage(f"Added {len(files)} files for processing.", 3000)


    def update_gallery_layout(self):
        # Clear existing layout (widgets will be parented to gallery_widget, remove them)
        while self.gallery_layout.count():
            child = self.gallery_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Re-add items (this part is tricky if you want to preserve existing items;
        # for simplicity, this is called before adding new items or on config change)
        # A better way would be to store QLabels and re-flow them.
        # For now, just clear. New items will be added by add_to_gallery.
        self.gallery_items_count = 0 # Reset for re-adding

    def add_to_gallery(self, pixmap, caption_text, summary_data): # summary_data not used for display here but passed
        item_widget = QWidget()
        item_layout = QVBoxLayout(item_widget)
        
        img_label = QLabel()
        img_label.setPixmap(pixmap.scaled(
            LIST_THUMBNAIL_SIZE[0], LIST_THUMBNAIL_SIZE[1], 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        img_label.setAlignment(Qt.AlignCenter)
        
        caption_label = QLabel(caption_text)
        caption_label.setWordWrap(True)
        caption_label.setAlignment(Qt.AlignCenter)
        
        item_layout.addWidget(img_label)
        item_layout.addWidget(caption_label)
        item_widget.setFixedSize(LIST_THUMBNAIL_SIZE[0] + 20, LIST_THUMBNAIL_SIZE[1] + 60) # Adjust for caption

        num_cols = self.gallery_cols_slider.value()
        row = self.gallery_items_count // num_cols
        col = self.gallery_items_count % num_cols
        self.gallery_layout.addWidget(item_widget, row, col)
        self.gallery_items_count += 1

    def update_metrics(self, processed_line, system_line, component_times_line):
        self.metrics_processed_label.setText(processed_line)
        self.metrics_system_label.setText(system_line)
        self.metrics_components_label.setText(component_times_line)

    def on_worker_finished(self, message):
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Analysis Complete", message)
        self.start_button.setEnabled(MODELS_LOADED) # Re-enable if models are loaded
        self.stop_button.setEnabled(False)
        self.worker_thread = None # Allow for a new run
        self.statusBar().showMessage("Analysis finished.", 5000)

    def on_worker_toast(self, message, icon):
        # For PyQt, could use status bar or a timed QMessageBox
        self.statusBar().showMessage(f"{icon} {message}", 4000)


    def start_analysis(self):
        if not MODELS_LOADED:
            QMessageBox.warning(self, "Models Not Loaded", "AI models are not loaded or failed to load. Cannot start analysis.")
            # Try loading them again, or instruct user.
            if not self._load_ai_models(): # Attempt to load again
                 return # Still failed
            if not MODELS_LOADED: # Double check
                 return

        self.all_items_to_process = [] # Reset from previous runs
        self.gallery_items_count = 0
        # Clear previous gallery items
        while self.gallery_layout.count():
            child = self.gallery_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Collect URLs
        urls_to_scrape = [url.strip() for url in self.url_list_input.toPlainText().splitlines() if url.strip()]
        max_images_per_url = self.max_images_slider.value()
        
        # Scrape (this part is blocking if not threaded, worker will handle it)
        # For now, let's assume worker handles scraping too, pass URLs.
        # Or, do scraping here and pass image data items. For simplicity with existing worker:

        temp_items_from_scrape = []
        if urls_to_scrape:
            self.statusBar().showMessage(f"Scraping {len(urls_to_scrape)} URL(s)...")
            QApplication.processEvents() # Update GUI
            # This should ideally be in a pre-worker or a part of the worker
            with ThreadPoolExecutor(max_workers=self.threads_slider.value()) as executor:
                future_to_url = {executor.submit(scrape_images_from_url, url, max_images_per_url, self.worker_thread.toast if self.worker_thread else None): url for url in urls_to_scrape}
                for i, future in enumerate(as_completed(future_to_url)):
                    url = future_to_url[future]
                    try:
                        data = future.result()
                        temp_items_from_scrape.extend(data)
                        self.statusBar().showMessage(f"Scraped {len(data)} from {url}", 2000)
                    except Exception as exc:
                        self.statusBar().showMessage(f"Error scraping {url}: {exc}", 3000)
                    QApplication.processEvents()
            self.all_items_to_process.extend(temp_items_from_scrape)
            self.statusBar().showMessage(f"Found {len(temp_items_from_scrape)} images from URLs.", 3000)


        # Collect from Directory
        dir_p = self.dir_path_input.text()
        if dir_p and os.path.isdir(dir_p):
            self.statusBar().showMessage(f"Loading from directory: {dir_p}...")
            QApplication.processEvents()
            dir_items = load_directory_images(dir_p, self.worker_thread.toast if self.worker_thread else None)
            self.all_items_to_process.extend(dir_items)
            self.statusBar().showMessage(f"Found {len(dir_items)} images from directory.", 3000)
        elif dir_p:
            QMessageBox.warning(self, "Invalid Directory", f"Directory not found or invalid: {dir_p}")

        # Uploaded files are already in self.all_items_to_process (added by upload_files)
        # We might need to re-think how self.all_items_to_process is populated if uploads are persistent

        if not self.all_items_to_process:
            QMessageBox.warning(self, "No Images", "No images found from URLs, directory, or uploads to process.")
            return

        settings = {
            'gender_conf_threshold': self.gender_conf_slider.value() / 100.0,
            'yolo_batch_size': self.yolo_batch_spinbox.value(),
            'threads': self.threads_slider.value(),
            'server_endpoint': self.server_endpoint_input.text(),
            'enable_reporting': self.enable_reporting_checkbox.isChecked()
        }

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        self.worker_thread = Worker(list(self.all_items_to_process), settings) # Pass a copy
        self.worker_thread.progress.connect(self.progress_bar.setValue)
        self.worker_thread.image_processed.connect(self.add_to_gallery)
        self.worker_thread.metrics_update.connect(self.update_metrics)
        self.worker_thread.finished.connect(self.on_worker_finished)
        self.worker_thread.toast.connect(self.on_worker_toast)
        self.worker_thread.start()
        
        # Clear uploaded items after they are passed to worker to avoid re-processing on next click
        # Or, better, provide a "Clear Inputs" button. For now, let's assume it's for one run.
        self.all_items_to_process = [] 


    def stop_analysis(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.statusBar().showMessage("Attempting to stop analysis...", 3000)
            # Worker should check self.running and exit loops
            # self.worker_thread.wait(5000) # Wait for graceful shutdown
            # if self.worker_thread.isRunning():
            #     self.worker_thread.terminate() # Force terminate if necessary
            #     self.statusBar().showMessage("Analysis forcefully terminated.", 3000)
            # else:
            #     self.statusBar().showMessage("Analysis stop requested.", 3000)
        self.stop_button.setEnabled(False)
        # self.start_button.setEnabled(MODELS_LOADED) # Enable start if models are ok

    def closeEvent(self, event):
        # Clean up GPU resources
        if GPU_AVAILABLE and pynvml:
            try:
                pynvml.nvmlShutdown()
                print("NVMl Shutdown successful.")
            except Exception as e:
                print(f"Error during NVMl Shutdown: {e}")
        
        # Stop worker thread if running
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait(2000) # Give it some time to stop

        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = ImageGenderDetectorApp()
    main_win.show()
    sys.exit(app.exec_())