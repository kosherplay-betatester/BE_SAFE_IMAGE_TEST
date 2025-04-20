import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import threading
import queue
import os
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import io
import sys
import torch
import clip
from concurrent.futures import ThreadPoolExecutor
import gc
from collections import OrderedDict
import warnings
import psutil
import pynvml
from ultralytics import YOLO

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration & Constants ---
LIST_THUMBNAIL_SIZE = (250, 200)
PERSON_CLASS_ID = 0  # YOLO person class
CONFIDENCE_THRESHOLD_PERSON = 0.45
COLOR_MALE = (255, 100, 0)
COLOR_FEMALE = (150, 50, 255)
COLOR_ELSE = (0, 200, 50)
BORDER_COLOR_MALE = "#0064FF"
BORDER_COLOR_FEMALE = "#FF3296"
BORDER_COLOR_ELSE = "#32C800"
BORDER_COLOR_MIXED = "#FFA500"
BORDER_COLOR_NONE = "#808080"
BORDER_COLOR_ERROR = "#FF0000"
MAX_THREADS = min(os.cpu_count() or 8, 8)
BATCH_SIZE = 8
MAX_CACHE_SIZE = 100
BENCHMARK_UPDATE_INTERVAL = 500  # ms

# --- Initialize NVIDIA Management Library ---
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except pynvml.NVMLError:
    GPU_AVAILABLE = False
    print("Warning: pynvml initialization failed. GPU and VRAM usage will not be monitored.")

# --- Model Loading & Device Setup ---
MODELS_LOADED = False
image_cache = OrderedDict()
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading YOLOv8s model...")
    model_detection = YOLO('yolov8s.pt').to(device)
    print("YOLOv8s model loaded.")

    print("Loading CLIP ViT-B/32 model...")
    model_classification, preprocess = clip.load("ViT-B/32", device=device)
    model_classification.to(device)
    categories = ["man", "woman", "object"]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)
    with torch.no_grad():
        text_features = model_classification.encode_text(text_inputs)
    print("CLIP model loaded.")
    MODELS_LOADED = True

except Exception as e:
    print(f"Model Loading Error: {e}")
    try:
        root_temp = tk.Tk()
        root_temp.withdraw()
        messagebox.showerror("Model Loading Error", f"Failed to load AI models: {e}\nCheck your installation and CUDA setup.", parent=None)
        root_temp.destroy()
    except tk.TclError:
        print("Tkinter error during error message display.")
    MODELS_LOADED = False

# --- Main Application Class ---
class GenderDetectorApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        if not MODELS_LOADED:
            print("Critical Error: AI Models failed to load.")
            try:
                self.destroy()
            except tk.TclError:
                sys.exit(1)
            return

        self.title("Gender Detector")
        self.geometry("1100x850")
        self.results_queue = queue.Queue()
        self.processing_active = False
        self._result_widgets = []
        self.start_time = None
        self.total_images_processed = 0
        self.total_size_processed_bytes = 0
        self.last_benchmark_update = 0

        # --- GUI Layout ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(4, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Control Area
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.browse_file_btn = ttk.Button(control_frame, text="Browse File", command=self.browse_file)
        self.browse_file_btn.pack(side=tk.LEFT, padx=5)
        self.browse_dir_btn = ttk.Button(control_frame, text="Browse Directory", command=self.browse_directory)
        self.browse_dir_btn.pack(side=tk.LEFT, padx=5)

        url_label = ttk.Label(control_frame, text="URL:")
        url_label.pack(side=tk.LEFT, padx=(15, 5))
        self.url_entry = ttk.Entry(control_frame, width=40)
        self.url_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.url_entry.bind("<Return>", self.process_url_input)
        self.fetch_url_btn = ttk.Button(control_frame, text="Fetch & Analyze", command=self.process_url_input)
        self.fetch_url_btn.pack(side=tk.LEFT, padx=5)

        # Thread Count Control
        thread_frame = ttk.Frame(main_frame)
        thread_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        self.thread_label = ttk.Label(thread_frame, text=f"Threads: {MAX_THREADS}")
        self.thread_label.pack(side=tk.LEFT, padx=5)
        self.thread_scale = ttk.Scale(thread_frame, from_=1, to_=os.cpu_count() or 8, orient=tk.HORIZONTAL, command=self.update_thread_count)
        self.thread_scale.set(MAX_THREADS)
        self.thread_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Status Area
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(0, 2))
        status_frame.grid_columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        self.status_label = ttk.Label(status_frame, text="Status: Idle. Drag & Drop files/folders here or use buttons/URL.")
        self.status_label.grid(row=0, column=0, sticky="ew", padx=5)

        # Benchmark Area
        benchmark_frame = ttk.Frame(main_frame)
        benchmark_frame.grid(row=3, column=0, sticky="ew", pady=(2, 5))
        self.benchmark_label = ttk.Label(benchmark_frame, text="Benchmark: -", anchor=tk.W, wraplength=1080)
        self.benchmark_label.pack(fill=tk.X, padx=5)

        # Results Area
        self.results_container = ttk.Frame(main_frame, borderwidth=1, relief="groove")
        self.results_container.grid(row=4, column=0, sticky="nsew", pady=(5, 0))
        self.results_container.grid_rowconfigure(0, weight=1)
        self.results_container.grid_columnconfigure(0, weight=1)

        self.results_canvas = tk.Canvas(self.results_container, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self.results_container, orient="vertical", command=self.results_canvas.yview)
        self.results_frame = ttk.Frame(self.results_canvas)
        self.results_frame_id = self.results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.results_canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        # Bindings
        self.results_frame.bind("<Configure>", self._on_frame_configure)
        self.results_canvas.bind("<Configure>", self._on_canvas_configure)
        self.results_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.results_canvas.bind("<Button-4>", self._on_mousewheel)
        self.results_canvas.bind("<Button-5>", self._on_mousewheel)
        self.results_frame.bind("<MouseWheel>", self._on_mousewheel)
        self.results_frame.bind("<Button-4>", self._on_mousewheel)
        self.results_frame.bind("<Button-5>", self._on_mousewheel)

        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_drop)
        self.results_container.drop_target_register(DND_FILES)
        self.results_container.dnd_bind('<<Drop>>', self.on_drop)
        self.results_canvas.drop_target_register(DND_FILES)
        self.results_canvas.dnd_bind('<<Drop>>', self.on_drop)

        self.after(200, self.check_queue)

    def update_thread_count(self, value):
        global MAX_THREADS
        MAX_THREADS = int(float(value))
        self.thread_label.config(text=f"Threads: {MAX_THREADS}")

    def _on_frame_configure(self, event=None):
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        canvas_width = event.width
        self.results_canvas.itemconfig(self.results_frame_id, width=canvas_width)

    def _on_mousewheel(self, event):
        target_canvas = None
        widget = event.widget
        while widget is not None:
            if widget == self.results_canvas:
                target_canvas = self.results_canvas
                break
            try:
                widget = widget.master
            except AttributeError:
                break
        if target_canvas:
            delta = 0
            if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
                delta = 1
            elif event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
                delta = -1
            if delta != 0:
                target_canvas.yview_scroll(delta, "units")

    def set_processing_state(self, active):
        self.processing_active = active
        state = tk.DISABLED if active else tk.NORMAL
        for widget in [self.browse_file_btn, self.browse_dir_btn, self.url_entry, self.fetch_url_btn, self.thread_scale]:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass
        if active:
            self.status_label.grid_forget()
            self.progress.grid(row=0, column=0, sticky="ew", padx=5)
            self.progress.start(10)
            self.status_label.config(text="Status: Processing...")
            self.last_benchmark_update = time.monotonic()
        else:
            self.progress.stop()
            self.progress.grid_forget()
            self.status_label.grid(row=0, column=0, sticky="ew", padx=5)

    def get_system_metrics(self):
        metrics = {}
        try:
            metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            metrics['ram_used'] = mem.used / (1024 ** 3)  # GB
            metrics['ram_percent'] = mem.percent
        except Exception:
            metrics['cpu_percent'] = metrics['ram_used'] = metrics['ram_percent'] = "N/A"

        if GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                metrics['gpu_percent'] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics['vram_used'] = mem_info.used / (1024 ** 3)  # GB
                metrics['vram_total'] = mem_info.total / (1024 ** 3)  # GB
                metrics['vram_percent'] = (mem_info.used / mem_info.total) * 100
            except pynvml.NVMLError:
                metrics['gpu_percent'] = metrics['vram_used'] = metrics['vram_percent'] = "N/A"
        else:
            metrics['gpu_percent'] = metrics['vram_used'] = metrics['vram_percent'] = "N/A"
        return metrics

    def update_benchmark(self):
        if not self.processing_active or not self.start_time:
            return
        current_time = time.monotonic()
        if current_time - self.last_benchmark_update < BENCHMARK_UPDATE_INTERVAL / 1000:
            return

        elapsed_time = current_time - self.start_time
        ips = self.total_images_processed / elapsed_time if elapsed_time > 0 else 0
        metrics = self.get_system_metrics()

        benchmark_text = (
            f"Benchmark: Processing {self.total_images_processed} items, "
            f"{ips:.2f} items/sec, Time: {elapsed_time:.2f}s\n"
            f"CPU: {metrics['cpu_percent']}% | "
            f"RAM: {metrics['ram_used']:.2f} GB ({metrics['ram_percent']}%) | "
            f"GPU: {metrics['gpu_percent']}% | "
            f"VRAM: {metrics['vram_used']:.2f}/{metrics['vram_total']:.2f} GB "
            f"({metrics['vram_percent']:.1f}%)" if GPU_AVAILABLE else
            f"Benchmark: Processing {self.total_images_processed} items, "
            f"{ips:.2f} items/sec, Time: {elapsed_time:.2f}s\n"
            f"CPU: {metrics['cpu_percent']}% | "
            f"RAM: {metrics['ram_used']:.2f} GB ({metrics['ram_percent']}%) | "
            f"GPU: N/A | VRAM: N/A"
        )
        try:
            self.benchmark_label.config(text=benchmark_text)
        except tk.TclError:
            pass
        self.last_benchmark_update = current_time

    def start_processing(self, items, source_type):
        if self.processing_active:
            return
        self.clear_results()
        self.start_time = time.monotonic()
        self.total_images_processed = 0
        self.total_size_processed_bytes = 0
        self.benchmark_label.config(text="Benchmark: Processing...")
        self.set_processing_state(True)
        thread = threading.Thread(target=self._processing_worker, args=(items, source_type), daemon=True)
        thread.start()

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")])
        if filepath:
            self.start_processing([filepath], "file")

    def browse_directory(self):
        dirpath = filedialog.askdirectory()
        if dirpath:
            image_files = [
                os.path.join(dirpath, fname)
                for fname in os.listdir(dirpath)
                if os.path.isfile(os.path.join(dirpath, fname)) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
            ]
            if not image_files:
                messagebox.showinfo("No Images Found", "No supported image files found.")
                return
            self.start_processing(image_files, "directory")

    def process_url_input(self, event=None):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Input Error", "Please enter a valid URL.")
            return
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        parsed_uri = urlparse(url)
        if not all([parsed_uri.scheme, parsed_uri.netloc]):
            messagebox.showwarning("Input Error", "Invalid URL format.")
            return
        self.start_processing([url], "url")

    def on_drop(self, event):
        if self.processing_active:
            return
        raw_paths = event.data
        paths = []
        try:
            potential_paths = self.tk.splitlist(raw_paths)
            paths = [p.strip('{}') for p in potential_paths if os.path.exists(p.strip('{}'))]
        except tk.TclError:
            temp_paths = raw_paths.replace('{', '').replace('}', '').split()
            current_path = ''
            for part in temp_paths:
                potential_path = (current_path + ' ' + part).strip()
                part_cleaned = part.strip("'\"")
                if os.path.exists(part_cleaned):
                    if current_path:
                        paths.append(current_path)
                    paths.append(part_cleaned)
                    current_path = ''
                elif os.path.exists(potential_path.strip("'\"")):
                    current_path = potential_path.strip("'\"")
                elif current_path:
                    paths.append(current_path)
                    current_path = ''
                    if os.path.exists(part_cleaned):
                        paths.append(part_cleaned)
            if current_path:
                paths.append(current_path)

        items_to_process = []
        for path in paths:
            if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                items_to_process.append(path)
            elif os.path.isdir(path):
                for fname in os.listdir(path):
                    fpath = os.path.join(path, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        items_to_process.append(fpath)

        if not items_to_process:
            messagebox.showinfo("No Images Found", "No processable image files found.")
            return
        self.start_processing(items_to_process, "drop")

    def clear_results(self):
        for widget in self._result_widgets:
            try:
                widget.destroy()
            except tk.TclError:
                pass
        self._result_widgets = []
        self.results_canvas.yview_moveto(0)
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        self.benchmark_label.config(text="Benchmark: -")

    def add_result_item(self, image_tk, text_summary, source_name, summary_counts):
        males = summary_counts.get("Male", 0)
        females = summary_counts.get("Female", 0)
        others = summary_counts.get("Else", 0)
        errors = summary_counts.get("Error", 0)
        total_persons = males + females

        border_color = BORDER_COLOR_ELSE
        if errors > 0:
            border_color = BORDER_COLOR_ERROR
        elif total_persons > 0:
            if males > females:
                border_color = BORDER_COLOR_MALE
            elif females > males:
                border_color = BORDER_COLOR_FEMALE
            else:
                border_color = BORDER_COLOR_MIXED
        elif others == 0 and total_persons == 0 and errors == 0:
            border_color = BORDER_COLOR_NONE

        outer_frame = tk.Frame(self.results_frame, bg=border_color, bd=0)
        item_frame = ttk.Frame(outer_frame, padding=5)
        item_frame.grid_columnconfigure(1, weight=1)

        source_label = ttk.Label(item_frame, text=source_name, wraplength=LIST_THUMBNAIL_SIZE[0] + 300, font=("Segoe UI", 9, "italic"), anchor=tk.W)
        source_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 3))

        if image_tk:
            img_label = ttk.Label(item_frame, image=image_tk)
            img_label.image = image_tk
        else:
            blank_img = Image.new('RGB', LIST_THUMBNAIL_SIZE, color=(200, 200, 200))
            draw = ImageDraw.Draw(blank_img)
            draw.text((10, 90), "[Image Error]", fill=(0, 0, 0))
            img_tk = ImageTk.PhotoImage(blank_img)
            img_label = ttk.Label(item_frame, image=img_tk)
            img_label.image = img_tk

        img_label.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        summary_label = ttk.Label(item_frame, text=text_summary, wraplength=450, justify=tk.LEFT, anchor=tk.NW)
        summary_label.grid(row=1, column=1, sticky="nsew")

        item_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        outer_frame.pack(pady=5, padx=5, fill=tk.X)
        self._result_widgets.append(outer_frame)
        self.results_canvas.update_idletasks()
        self.results_canvas.yview_moveto(1.0)

    def _processing_worker(self, items, source_type):
        worker_image_count = 0
        worker_total_size_bytes = 0
        has_errors = False

        try:
            items_to_process_with_size = []
            total_items = 0

            if source_type == "url":
                url = items[0]
                self.results_queue.put({"type": "text_only", "data": f"Fetching images from: {url}...\n"})
                image_sources = self._scrape_images_from_url(url)
                if not image_sources:
                    self.results_queue.put({"type": "text_only", "data": f"No suitable images found for URL: {url}\n"})
                    self.results_queue.put({"type": "status", "data": "Finished", "count": 0, "size": 0, "errors": False})
                    return

                source_name_prefix = f"URL ({urlparse(url).netloc})"
                total_items = len(image_sources)
                self.results_queue.put({"type": "text_only", "data": f"Analyzing {total_items} images from URL...\n"})
                for i, (img_data, src_url) in enumerate(image_sources):
                    base_name = os.path.basename(urlparse(src_url).path) or f"image_{i+1}"
                    source_name = f"{source_name_prefix} - {base_name}"
                    size_bytes = len(img_data)
                    items_to_process_with_size.append((img_data, source_name, size_bytes))

            else:
                source_name_prefix = "File" if source_type == "file" else "Drop/Dir"
                total_items = len(items)
                self.results_queue.put({"type": "text_only", "data": f"Analyzing {total_items} local images...\n"})
                for item_path in items:
                    source_name = os.path.basename(item_path)
                    size_bytes = os.path.getsize(item_path) if os.path.exists(item_path) else -1
                    items_to_process_with_size.append((item_path, source_name, size_bytes))

            processed_count = 0
            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                for batch_start in range(0, len(items_to_process_with_size), BATCH_SIZE):
                    batch = items_to_process_with_size[batch_start:batch_start + BATCH_SIZE]
                    processed_count += len(batch)
                    if total_items > 1 and (processed_count % 5 == 0 or processed_count == total_items or processed_count == 1):
                        self.results_queue.put({"type": "status", "data": f"Processing {processed_count}/{total_items}..."})

                    futures = []
                    batch_images = []
                    batch_pil_images = []
                    batch_metadata = []

                    for item_data_or_path, source_name, size_bytes in batch:
                        try:
                            if isinstance(item_data_or_path, str):
                                if not os.path.exists(item_data_or_path):
                                    raise FileNotFoundError(f"File not found: {item_data_or_path}")
                                img_cv = cv2.imread(item_data_or_path)
                            else:
                                img_np = np.frombuffer(item_data_or_path, np.uint8)
                                img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                            if img_cv is None:
                                raise ValueError(f"Could not load image: {source_name}")
                            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(img_rgb)
                            batch_images.append(img_cv)
                            batch_pil_images.append(img_pil)
                            batch_metadata.append((source_name, size_bytes))
                            if size_bytes > 0:
                                worker_total_size_bytes += size_bytes
                        except Exception as e:
                            self.results_queue.put({
                                "type": "result_item", "image_tk": None,
                                "text": f"Error: {str(e)}",
                                "summary": {"Error": 1}, "source_name": source_name,
                                "size_bytes": 0
                            })
                            has_errors = True
                            continue

                    if batch_pil_images:
                        try:
                            with torch.no_grad():
                                results_yolo = model_detection.predict(batch_pil_images, device=device, conf=CONFIDENCE_THRESHOLD_PERSON)
                            for img_cv, img_pil, yolo_result, (source_name, size_bytes) in zip(
                                batch_images, batch_pil_images, results_yolo, batch_metadata
                            ):
                                futures.append(executor.submit(
                                    self._analyze_image_thread,
                                    img_cv, img_pil, yolo_result, source_name, size_bytes
                                ))
                                worker_image_count += 1
                        except Exception as e:
                            print(f"Batch YOLOv8 error: {e}")
                            for source_name, size_bytes in batch_metadata:
                                self.results_queue.put({
                                    "type": "result_item", "image_tk": None,
                                    "text": f"Error: YOLOv8 batch processing failed: {e}",
                                    "summary": {"Error": 1}, "source_name": source_name,
                                    "size_bytes": 0
                                })
                            has_errors = True

                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Thread error: {e}")
                            has_errors = True

                    if device == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()

            final_status_msg = "Error" if has_errors else "Finished"
            self.results_queue.put({
                "type": "status",
                "data": final_status_msg,
                "count": worker_image_count,
                "size": worker_total_size_bytes,
                "errors": has_errors
            })

        except Exception as e:
            print(f"Fatal Worker Error: {e}")
            self.results_queue.put({"type": "text_only", "data": f"Fatal error: {e}\n"})
            self.results_queue.put({"type": "status", "data": "Error", "count": worker_image_count, "size": worker_total_size_bytes, "errors": True})

    def _scrape_images_from_url(self, url):
        images = []
        processed_urls = set()
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
            }
            session = requests.Session()
            session.headers.update(headers)
            response = session.get(url, timeout=20, stream=True, allow_redirects=True)
            response.raise_for_status()
            final_url = response.url
            content_type = response.headers.get('content-type', '').lower()

            if 'html' not in content_type:
                if 'image' in content_type:
                    img_data = response.content
                    if img_data:
                        return [(img_data, final_url)]
                    return []
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img')
            download_count = 0
            min_size_bytes = 2000

            for img_tag in img_tags:
                img_url = img_tag.get('data-src') or img_tag.get('src')
                if not img_url:
                    continue
                img_url = urljoin(final_url, img_url.strip())
                parsed_img_url = urlparse(img_url)
                if not all([parsed_img_url.scheme in ['http', 'https'], parsed_img_url.netloc]):
                    continue
                if img_url in processed_urls or parsed_img_url.path.lower().endswith(('.gif', '.svg', '.ico')):
                    continue

                processed_urls.add(img_url)
                try:
                    img_response = session.get(img_url, timeout=15, stream=True)
                    img_response.raise_for_status()
                    img_content_type = img_response.headers.get('content-type', '').lower()
                    if 'image' not in img_content_type or img_content_type.endswith(('gif', 'svg+xml')):
                        continue
                    img_data = img_response.content
                    if img_data and len(img_data) > min_size_bytes:
                        images.append((img_data, img_url))
                        download_count += 1
                        if download_count >= 30:
                            break
                except Exception:
                    pass
        except Exception:
            pass
        return images

    def _analyze_image_thread(self, img_cv, img_pil, yolo_result, source_name, size_bytes):
        img_tk = None
        processed_image = None
        summary = {"Male": 0, "Female": 0, "Else": 0, "Error": 0}
        detections = []
        result_text_summary = "Analysis Error."

        try:
            processed_image = img_cv.copy()
            crops = []
            for box in yolo_result.boxes:
                if yolo_result.names[int(box.cls)] == "person" and box.conf >= CONFIDENCE_THRESHOLD_PERSON:
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_cv.shape[1], x2), min(img_cv.shape[0], y2)
                    if x1 >= x2 or y1 >= y2:
                        continue
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if bbox_area / (img_pil.width * img_pil.height) < 0.1111:
                        continue
                    crop = img_pil.crop((x1, y1, x2, y2))
                    crops.append((crop, (x1, y1, x2, y2)))

            if crops:
                image_inputs = torch.stack([preprocess(crop) for crop, _ in crops]).to(device)
                with torch.no_grad():
                    image_features = model_classification.encode_image(image_inputs)
                    logits_per_image = image_features @ text_features.t()
                    probs = torch.softmax(logits_per_image, dim=-1).cpu().numpy()

                for i, (crop, box) in enumerate(crops):
                    scores = {cat: prob for cat, prob in zip(categories, probs[i])}
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    highest_category = sorted_scores[0][0]
                    confidence = sorted_scores[0][1]

                    gender = "Else"
                    color = COLOR_ELSE
                    if highest_category == "man" and confidence >= 0.5:
                        gender = "Male"
                        color = COLOR_MALE
                    elif highest_category == "woman" and confidence >= 0.5:
                        gender = "Female"
                        color = COLOR_FEMALE

                    x1, y1, x2, y2 = box
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                    font_scale = 0.5
                    thickness = 1
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (w, h), _ = cv2.getTextSize(gender, font, font_scale, thickness)
                    label_y = y1 - 5
                    bg_y1 = max(label_y - h, 0)
                    bg_y2 = label_y + 5
                    cv2.rectangle(processed_image, (x1, bg_y1), (x1 + w, bg_y2), color, -1)
                    cv2.putText(processed_image, gender, (x1, label_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                    detections.append({'box': box, 'gender': gender, 'color': color})
                    summary[gender] += 1

            if not detections:
                detections.append({'box': None, 'gender': 'No persons detected', 'color': None})
                result_text_summary = "No persons detected."
            else:
                result_text_summary = f"Male: {summary['Male']}, Female: {summary['Female']}, Other: {summary['Else'] + summary['Error']}"

            img_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            img_pil_processed = Image.fromarray(img_rgb)
            img_pil_processed.thumbnail(LIST_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil_processed)

        except Exception as e:
            print(f"Error in thread for {source_name}: {e}")
            result_text_summary = f"Error: {str(e)}"
            summary["Error"] += 1

        finally:
            self.results_queue.put({
                "type": "result_item",
                "image_tk": img_tk,
                "text": result_text_summary,
                "summary": summary,
                "source_name": source_name,
                "size_bytes": size_bytes if img_tk is not None and size_bytes >= 0 else 0
            })

    def check_queue(self):
        try:
            batch_updates = []
            while True:
                message = self.results_queue.get_nowait()
                batch_updates.append(message)
        except queue.Empty:
            pass

        try:
            for message in batch_updates:
                msg_type = message.get("type")
                if msg_type == "result_item":
                    image_tk = message.get("image_tk")
                    size_bytes = message.get("size_bytes", 0)
                    self.add_result_item(
                        image_tk,
                        message.get("text"),
                        message.get("source_name"),
                        message.get("summary")
                    )
                    if image_tk is not None:
                        self.total_images_processed += 1
                        if size_bytes > 0:
                            self.total_size_processed_bytes += size_bytes

                elif msg_type == "text_only":
                    print(f"Log: {message.get('data').strip()}")

                elif msg_type == "status":
                    msg_data = message.get("data")
                    if msg_data in ["Finished", "Error"]:
                        self.set_processing_state(False)
                        end_time = time.monotonic()
                        duration = end_time - self.start_time if self.start_time else 0
                        final_image_count = message.get("count", 0)
                        final_total_size = message.get("size", 0)
                        has_errors = message.get("errors", False)
                        ips = final_image_count / duration if duration > 0 else 0
                        total_mb = final_total_size / (1024 * 1024)
                        metrics = self.get_system_metrics()
                        benchmark_text = (
                            f"Benchmark: Processed {final_image_count} items "
                            f"({total_mb:.2f} MB) in {duration:.2f}s "
                            f"({ips:.2f} items/sec)\n"
                            f"Final CPU: {metrics['cpu_percent']}% | "
                            f"RAM: {metrics['ram_used']:.2f} GB ({metrics['ram_percent']}%) | "
                            f"GPU: {metrics['gpu_percent']}% | "
                            f"VRAM: {metrics['vram_used']:.2f}/{metrics['vram_total']:.2f} GB "
                            f"({metrics['vram_percent']:.1f}%)" if GPU_AVAILABLE else
                            f"Benchmark: Processed {final_image_count} items "
                            f"({total_mb:.2f} MB) in {duration:.2f}s "
                            f"({ips:.2f} items/sec)\n"
                            f"Final CPU: {metrics['cpu_percent']}% | "
                            f"RAM: {metrics['ram_used']:.2f} GB ({metrics['ram_percent']}%) | "
                            f"GPU: N/A | VRAM: N/A"
                        )
                        self.status_label.config(text=f"Status: {'Idle.' if msg_data == 'Finished' else 'Finished with errors.'}")
                        self.benchmark_label.config(text=benchmark_text)
                        self.start_time = None
                    else:
                        if self.processing_active:
                            self.status_label.config(text=f"Status: {msg_data}")

            # Update benchmark during processing
            self.update_benchmark()

        except Exception as e:
            print(f"Error processing queue message: {e}")
            try:
                self.status_label.config(text="Status: GUI Update Error!")
                self.benchmark_label.config(text="Benchmark: Error")
                if self.processing_active:
                    self.set_processing_state(False)
            except Exception as e2:
                print(f"Further error updating GUI: {e2}")
        finally:
            if self.winfo_exists():
                self.after(200, self.check_queue)

if __name__ == "__main__":
    if not MODELS_LOADED:
        print("Application cannot start because models failed to load.")
        sys.exit(1)

    app = None
    try:
        app = GenderDetectorApp()
        style = ttk.Style(app)
        style.theme_use('clam')
        app.mainloop()
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if app and app.winfo_exists():
            app.destroy()
        if GPU_AVAILABLE:
            pynvml.nvmlShutdown()
        print("Application finished.")
        sys.exit(0)