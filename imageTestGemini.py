import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinterdnd2 import DND_FILES, TkinterDnD # Use TkinterDnD instead of tk
import threading
import queue
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import io
import sys # To check platform for potential DnD issues

# --- Configuration & Constants ---
LIST_THUMBNAIL_SIZE = (250, 200) # Size for thumbnails in the results list
PERSON_CLASS_ID = 0  # COCO class ID for 'person'
CONFIDENCE_THRESHOLD_PERSON = 0.45 # Confidence for YOLO person detection
# Colors in BGR format for OpenCV
COLOR_MALE = (255, 100, 0)      # Blue
COLOR_FEMALE = (150, 50, 255)   # Pink/Purple
COLOR_ELSE = (0, 200, 50)       # Green
# Colors for result item borders (Hex for Tkinter)
BORDER_COLOR_MALE = "#0064FF"   # Blue
BORDER_COLOR_FEMALE = "#FF3296" # Pink
BORDER_COLOR_ELSE = "#32C800"   # Green
BORDER_COLOR_MIXED = "#FFA500"  # Orange for mixed results
BORDER_COLOR_NONE = "#808080"   # Gray for no persons
BORDER_COLOR_ERROR = "#FF0000"  # Red for processing errors

# --- Model Loading & Device Setup ---
MODELS_LOADED = False
try:
    import torch
    if torch.cuda.is_available():
        print("CUDA is available. Attempting to use GPU.")
        device = 'cuda'
    else:
        print("CUDA not available. Using CPU.")
        device = 'cpu'

    print("Loading YOLOv8 model...")
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8n.pt') # yolov8s.pt might be better balance
    print("YOLOv8 model loaded.")

    print("Initializing DeepFace (may download models on first run)...")
    from deepface import DeepFace
    # Pre-warm DeepFace by analyzing a dummy image
    _ = DeepFace.analyze(np.zeros((100, 100, 3), dtype=np.uint8),
                         actions=['gender'],
                         detector_backend='opencv', # Faster backend for pre-warming
                         enforce_detection=False,
                         silent=True)
    print("DeepFace initialized.")
    MODELS_LOADED = True

except ImportError as e:
    # Use tkinter messagebox only if Tk root is available, otherwise print
    try:
        root_temp = tk.Tk()
        root_temp.withdraw() # Hide the temporary window
        messagebox.showerror("Import Error", f"A required library is missing: {e}\nPlease install all dependencies using pip.", parent=None)
        root_temp.destroy()
    except tk.TclError:
         print(f"Import Error: A required library is missing: {e}\nPlease install all dependencies using pip.")
    MODELS_LOADED = False
except Exception as e:
    try:
        root_temp = tk.Tk()
        root_temp.withdraw()
        messagebox.showerror("Model Loading Error", f"Failed to load AI models: {e}\nCheck your installation and CUDA setup (if applicable).", parent=None)
        root_temp.destroy()
    except tk.TclError:
        print(f"Model Loading Error: Failed to load AI models: {e}\nCheck your installation and CUDA setup (if applicable).")
    MODELS_LOADED = False

# --- Main Application Class ---
class GenderDetectorApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        if not MODELS_LOADED:
            # If models didn't load, Tk might not be fully initialized for messagebox
            print("Critical Error: AI Models failed to load. Application cannot start.")
            # Attempt to destroy gracefully, but it might fail if called too early
            try:
                self.destroy()
            except tk.TclError:
                sys.exit(1) # Force exit if destroy fails
            return # Should not be reached if destroy works

        self.title("Gender Detector")
        self.geometry("1100x850") # Slightly taller for benchmark label

        self.results_queue = queue.Queue()
        self.processing_active = False
        self._result_widgets = [] # Keep track of result widgets for clearing

        # --- Benchmark Variables ---
        self.start_time = None
        self.total_images_processed = 0
        self.total_size_processed_bytes = 0


        # --- GUI Layout ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(3, weight=1) # Row 3 (results) expands
        main_frame.grid_columnconfigure(0, weight=1) # Main content column

        # --- Top Control Area ---
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

        # --- Status Area ---
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(0, 2))
        status_frame.grid_columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        # Progress bar will be gridded dynamically by set_processing_state

        self.status_label = ttk.Label(status_frame, text="Status: Idle. Drag & Drop files/folders here or use buttons/URL.")
        self.status_label.grid(row=0, column=0, sticky="ew", padx=5)

        # --- Benchmark Display Area ---
        benchmark_frame = ttk.Frame(main_frame)
        benchmark_frame.grid(row=2, column=0, sticky="ew", pady=(2, 5)) # Added padding
        self.benchmark_label = ttk.Label(benchmark_frame, text="Benchmark: -", anchor=tk.W)
        self.benchmark_label.pack(fill=tk.X, padx=5)

        # --- Results Area (Scrollable Canvas) ---
        self.results_container = ttk.Frame(main_frame, borderwidth=1, relief="groove") # Fixed: made instance attribute
        self.results_container.grid(row=3, column=0, sticky="nsew", pady=(5, 0)) # Grid at row 3
        self.results_container.grid_rowconfigure(0, weight=1)
        self.results_container.grid_columnconfigure(0, weight=1)

        # Fixed: used self.results_container as parent
        self.results_canvas = tk.Canvas(self.results_container, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self.results_container, orient="vertical", command=self.results_canvas.yview)
        self.results_frame = ttk.Frame(self.results_canvas) # Frame inside canvas

        self.results_frame_id = self.results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.results_canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")


        # --- Bindings & Initial Setup ---
        self.results_frame.bind("<Configure>", self._on_frame_configure)
        self.results_canvas.bind("<Configure>", self._on_canvas_configure)
        # Bind scrolling to the canvas and the frame inside it
        self.results_canvas.bind("<MouseWheel>", self._on_mousewheel) # Windows/Mac
        self.results_canvas.bind("<Button-4>", self._on_mousewheel)   # Linux scroll up
        self.results_canvas.bind("<Button-5>", self._on_mousewheel)   # Linux scroll down
        self.results_frame.bind("<MouseWheel>", self._on_mousewheel)  # Also bind to inner frame
        self.results_frame.bind("<Button-4>", self._on_mousewheel)
        self.results_frame.bind("<Button-5>", self._on_mousewheel)


        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_drop)
        # Allow dropping onto the results area as well
        self.results_container.drop_target_register(DND_FILES)
        self.results_container.dnd_bind('<<Drop>>', self.on_drop)
        self.results_canvas.drop_target_register(DND_FILES)
        self.results_canvas.dnd_bind('<<Drop>>', self.on_drop)

        self.after(100, self.check_queue) # Start queue checker

    # --- Scroll Handling ---
    def _on_frame_configure(self, event=None):
        """Reset scroll region when the frame holding results changes size."""
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Adjust the width of the frame inside the canvas."""
        canvas_width = event.width
        self.results_canvas.itemconfig(self.results_frame_id, width=canvas_width)


    def _on_mousewheel(self, event):
        """Scroll the results canvas if the mouse is over it or its contents."""
        # Determine which canvas to scroll based on the widget triggering the event
        target_canvas = None
        widget = event.widget
        while widget is not None:
            if widget == self.results_canvas:
                target_canvas = self.results_canvas
                break
            # Add checks for other potential scrollable areas here if needed
            try:
                widget = widget.master
            except AttributeError:
                break # Reached top level or error

        if target_canvas:
            # Platform-specific scroll delta calculation
            delta = 0
            if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0): # Scroll down
                delta = 1
            elif event.num == 4 or (hasattr(event, 'delta') and event.delta > 0): # Scroll up
                delta = -1

            if delta != 0:
                target_canvas.yview_scroll(delta, "units")


    # --- GUI Event Handlers ---
    def set_processing_state(self, active):
        """Enable/disable controls, show/hide progress bar."""
        self.processing_active = active
        state = tk.DISABLED if active else tk.NORMAL

        widgets_to_toggle = [
            self.browse_file_btn, self.browse_dir_btn,
            self.url_entry, self.fetch_url_btn
        ]
        for widget in widgets_to_toggle:
            try: # Add try-except in case a widget wasn't created properly
               widget.config(state=state)
            except tk.TclError as e:
                print(f"Warning: Could not configure widget state: {e}")


        if active:
            self.status_label.grid_forget()
            self.progress.grid(row=0, column=0, sticky="ew", padx=5)
            self.progress.start(10)
            self.status_label.config(text="Status: Processing...")
        else:
            self.progress.stop()
            self.progress.grid_forget()
            self.status_label.grid(row=0, column=0, sticky="ew", padx=5)
            # Final status text updated by check_queue

    def start_processing(self, items, source_type):
        """Clears results, resets benchmark, and starts processing."""
        if self.processing_active: return
        self.clear_results()

        # Reset benchmark counters
        self.start_time = time.monotonic()
        self.total_images_processed = 0
        self.total_size_processed_bytes = 0
        self.benchmark_label.config(text="Benchmark: Processing...") # Indicate processing start

        self.set_processing_state(True)
        thread = threading.Thread(target=self._processing_worker, args=(items, source_type), daemon=True)
        thread.start()

    def browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        if filepath:
            self.start_processing([filepath], "file")

    def browse_directory(self):
        dirpath = filedialog.askdirectory(title="Select Folder Containing Images")
        if dirpath:
            image_files = []
            try:
                for fname in os.listdir(dirpath):
                    fpath = os.path.join(dirpath, fname)
                    # Check if it's a file and has a supported extension
                    if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        image_files.append(fpath)
            except OSError as e:
                messagebox.showerror("Directory Error", f"Could not read directory:\n{e}")
                return

            if not image_files:
                messagebox.showinfo("No Images Found", "The selected directory contains no supported image files.")
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
        if self.processing_active: return
        raw_paths = event.data
        print(f"Raw drop data: {raw_paths}")
        paths = []
        # Try Tkinter's built-in list splitter first (often works well)
        try:
             potential_paths = self.tk.splitlist(raw_paths)
             # Clean paths (remove potential curly braces) and verify existence
             paths = [p.strip('{}') for p in potential_paths if os.path.exists(p.strip('{}'))]
             print(f"Parsed paths using splitlist: {paths}")
        except tk.TclError:
             paths = [] # splitlist failed, try manual parsing

        if not paths:
             # Fallback parsing (handles some cases with spaces, quotes)
             temp_paths = raw_paths.replace('{', '').replace('}', '').split()
             current_path = ''
             for part in temp_paths:
                 potential_path = (current_path + ' ' + part).strip()
                 # Check standalone path first, stripping quotes
                 part_cleaned = part.strip("'\"")
                 if os.path.exists(part_cleaned):
                     if current_path: paths.append(current_path) # Add previous assembled path
                     paths.append(part_cleaned)
                     current_path = ''
                 elif os.path.exists(potential_path.strip("'\"")): # Combined parts form a path
                     current_path = potential_path.strip("'\"")
                 elif current_path: # Adding part failed, add last known good path
                     paths.append(current_path)
                     current_path = ''
                     # Check the standalone part again after reset
                     if os.path.exists(part_cleaned):
                         paths.append(part_cleaned)
                 # else: part is a fragment, ignore
             if current_path: # Add any remaining assembled path
                 paths.append(current_path)
             print(f"Parsed paths using fallback: {paths}")


        if not paths:
            messagebox.showwarning("Drop Error", "Could not parse dropped file/folder paths.")
            return

        items_to_process = []
        for path in paths:
              # path should already be cleaned and verified by parsing logic
             try:
                 if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    items_to_process.append(path)
                 elif os.path.isdir(path):
                    for fname in os.listdir(path):
                        fpath = os.path.join(path, fname)
                        if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                            items_to_process.append(fpath)
             except OSError as e:
                 print(f"Warning: Could not access path '{path}': {e}")
                 messagebox.showwarning("Access Error", f"Could not access:\n{path}\n\nError: {e}\n\nSkipping this item.")

        if not items_to_process:
            messagebox.showinfo("No Images Found", "The dropped items contain no processable image files.")
            return

        self.start_processing(items_to_process, "drop")

    def clear_results(self):
        """Clear previous results and reset benchmarks."""
        for widget in self._result_widgets:
             try:
                 widget.destroy()
             except tk.TclError:
                 pass # Widget might already be destroyed
        self._result_widgets = []
        self.results_canvas.yview_moveto(0) # Scroll to top
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        self.benchmark_label.config(text="Benchmark: -") # Reset benchmark display
        # Don't reset counters here, start_processing does that

    def add_result_item(self, image_tk, text_summary, source_name, summary_counts):
        """Adds a new result item widget to the scrollable results_frame."""
        males = summary_counts.get("Male", 0)
        females = summary_counts.get("Female", 0)
        others = summary_counts.get("Else", 0)
        errors = summary_counts.get("Error", 0)
        total_persons = males + females

        border_color = BORDER_COLOR_ELSE # Default: Else
        if errors > 0:
            border_color = BORDER_COLOR_ERROR
        elif total_persons > 0:
            if males > females: border_color = BORDER_COLOR_MALE
            elif females > males: border_color = BORDER_COLOR_FEMALE
            else: border_color = BORDER_COLOR_MIXED # Equal M/F
        elif others == 0 and total_persons == 0 and errors == 0: # No persons/objects/errors detected
             border_color = BORDER_COLOR_NONE

        # --- Create Widgets ---
        # Outer frame provides the colored border
        outer_frame = tk.Frame(self.results_frame, bg=border_color, bd=0)
        # Inner frame holds the content with padding
        item_frame = ttk.Frame(outer_frame, padding=5)
        item_frame.grid_columnconfigure(1, weight=1) # Text column expands

        # Source Name
        source_label = ttk.Label(item_frame, text=source_name, wraplength=LIST_THUMBNAIL_SIZE[0] + 300, font=("Segoe UI", 9, "italic"), anchor=tk.W)
        source_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 3))

        # Image Thumbnail (handle None case for errors)
        if image_tk:
            img_label = ttk.Label(item_frame, image=image_tk)
            img_label.image = image_tk # Keep reference
        else:
            # Placeholder for error items without image
            # Estimate width/height based on thumbnail size for consistency
            ph_width = int(LIST_THUMBNAIL_SIZE[0] / 7) # Rough estimate
            ph_height = int(LIST_THUMBNAIL_SIZE[1] / 15)
            img_label = ttk.Label(item_frame, text="[Image Error]", relief="solid", borderwidth=1, width=ph_width, height=ph_height, anchor=tk.CENTER)
            img_label.image = None # Explicitly set no image

        img_label.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(0, 10)) # Use N+S+E+W

        # Text Summary
        summary_label = ttk.Label(item_frame, text=text_summary, wraplength=450, justify=tk.LEFT, anchor=tk.NW)
        summary_label.grid(row=1, column=1, sticky="nsew")

        # --- Packing ---
        item_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        outer_frame.pack(pady=5, padx=5, fill=tk.X) # Fill width of the canvas frame

        self._result_widgets.append(outer_frame)

        # Scroll to show the newly added item if it's near the bottom
        self.results_canvas.update_idletasks() # Ensure canvas size is updated
        self.results_canvas.yview_moveto(1.0) # Scroll to bottom


    # --- Processing Logic (Worker Thread) ---

    def _processing_worker(self, items, source_type):
        """Worker function: handles URL fetching or file processing."""
        worker_image_count = 0
        worker_total_size_bytes = 0
        has_errors = False

        try:
            items_to_process_with_size = [] # Store tuples: (item_data_or_path, source_name, size_bytes)
            total_items = 0

            if source_type == "url":
                url = items[0]
                self.results_queue.put({"type": "text_only", "data": f"Fetching images from: {url}...\n"})
                image_sources = self._scrape_images_from_url(url) # Returns list of (bytes, url)
                if not image_sources:
                    self.results_queue.put({"type": "text_only", "data": f"No suitable images found or fetch failed for URL: {url}\n"})
                    # Send finish signal immediately if no images, include 0 counts
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

            else: # File, Drop/Dir
                source_name_prefix = "File" if source_type == "file" else "Drop/Dir"
                total_items = len(items)
                self.results_queue.put({"type": "text_only", "data": f"Analyzing {total_items} local images...\n"})

                for item_path in items:
                    source_name = os.path.basename(item_path)
                    size_bytes = 0
                    try:
                        if os.path.exists(item_path):
                           size_bytes = os.path.getsize(item_path)
                        else:
                           print(f"Warning: File not found during worker processing: {item_path}")
                           # Let _analyze_image handle the FileNotFoundError for consistent reporting
                    except OSError as e:
                        print(f"Warning: Could not get size for {item_path}: {e}")
                        size_bytes = -1 # Indicate error getting size

                    items_to_process_with_size.append((item_path, source_name, size_bytes))


            # --- Process Items ---
            processed_count = 0
            for item_data_or_path, source_name, size_bytes in items_to_process_with_size:
                processed_count += 1
                # Update status infrequently
                if total_items > 1 and (processed_count % 5 == 0 or processed_count == total_items or processed_count == 1):
                     self.results_queue.put({"type": "status", "data": f"Processing {processed_count}/{total_items}..."})

                try:
                    # Add size to worker total *before* analysis (if size is known)
                    if size_bytes > 0: # Exclude size errors (-1) or zero-byte files for benchmark avg
                       worker_total_size_bytes += size_bytes

                    self._analyze_image(item_data_or_path, source_name, size_bytes)
                    worker_image_count += 1 # Count attempted analysis

                except FileNotFoundError as fnf_err:
                     print(f"Error (Worker): {fnf_err}")
                     # Send specific error result item
                     self.results_queue.put({
                         "type": "result_item", "image_tk": None,
                         "text": f"Error: File not found",
                         "summary": {"Error": 1}, "source_name": source_name,
                         "size_bytes": 0 # Don't count size for errors
                     })
                     has_errors = True
                except (ValueError, TypeError) as val_err: # Catch decode/type errors etc.
                     print(f"Error (Worker) processing {source_name}: {val_err}")
                     self.results_queue.put({
                         "type": "result_item", "image_tk": None,
                         "text": f"Error: {val_err}",
                         "summary": {"Error": 1}, "source_name": source_name,
                         "size_bytes": 0
                     })
                     has_errors = True
                except Exception as item_err: # Catch unexpected errors during item processing
                     print(f"Unexpected Error (Worker) processing {source_name}: {item_err}")
                     import traceback
                     traceback.print_exc()
                     self.results_queue.put({
                         "type": "result_item", "image_tk": None,
                         "text": f"Unexpected Error: {item_err}",
                         "summary": {"Error": 1}, "source_name": source_name,
                         "size_bytes": 0
                     })
                     has_errors = True

            # Send final status with benchmark data
            final_status_msg = "Error" if has_errors else "Finished"
            self.results_queue.put({
                "type": "status",
                "data": final_status_msg,
                "count": worker_image_count, # Total analyzed or attempted
                "size": worker_total_size_bytes, # Sum of sizes before analysis attempt
                "errors": has_errors # Pass error flag
                })

        except Exception as worker_err: # Catch errors in worker setup/loop logic
            print(f"Fatal Error in Processing Worker: {worker_err}")
            import traceback
            traceback.print_exc()
            self.results_queue.put({"type": "text_only", "data": f"A fatal error occurred during processing: {worker_err}\n"})
            # Ensure a final status is sent even on fatal worker error
            self.results_queue.put({"type": "status", "data": "Error", "count": worker_image_count, "size": worker_total_size_bytes, "errors": True})


    def _scrape_images_from_url(self, url):
        """Fetches HTML, finds image tags, downloads, returns list of (image_bytes, source_url)."""
        images = []
        processed_urls = set() # Avoid downloading the same image URL multiple times
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                       'Accept': 'image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'} # Be more specific about acceptable image types
            session = requests.Session()
            session.headers.update(headers)

            response = session.get(url, timeout=20, stream=True, allow_redirects=True) # Follow redirects
            response.raise_for_status()

            final_url = response.url # Get URL after redirects
            content_type = response.headers.get('content-type', '').lower()

            if 'html' not in content_type:
                 if 'image' in content_type:
                     print(f"URL points directly to an image: {final_url}")
                     img_data = response.content
                     if img_data: return [(img_data, final_url)]
                     else:
                         self.results_queue.put({"type": "text_only", "data": f"Failed download direct image: {final_url}\n"})
                         return []
                 else:
                     self.results_queue.put({"type": "text_only", "data": f"URL does not point to HTML/Image: {final_url} (Type: {content_type})\n"})
                     return []

            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img')
            download_count = 0
            min_size_bytes = 2000

            for img_tag in img_tags:
                img_url = None
                lazy_attrs = ['data-src', 'data-lazy-src', 'data-original', 'data-lazy']
                for attr in lazy_attrs:
                    if img_tag.has_attr(attr):
                        img_url = img_tag[attr]
                        break
                if not img_url: img_url = img_tag.get('src')
                if not img_url: continue

                try:
                    img_url = urljoin(final_url, img_url.strip()) # Use final_url as base
                except ValueError:
                     print(f"Skipping invalid image URL fragment: {img_url}")
                     continue # Skip if urljoin fails

                # Basic URL validation and filtering
                parsed_img_url = urlparse(img_url)
                if not all([parsed_img_url.scheme in ['http', 'https'], parsed_img_url.netloc]): continue # Needs scheme and domain
                if img_url in processed_urls: continue # Already tried this URL
                if parsed_img_url.path.lower().endswith(('.gif', '.svg', '.ico')): continue # Skip common non-photographic formats

                processed_urls.add(img_url)

                try:
                    print(f"Attempting download: {img_url}")
                    # Use stream=True for potentially large images
                    img_response = session.get(img_url, timeout=15, stream=True) # Shorter timeout for individual images
                    img_response.raise_for_status()
                    img_content_type = img_response.headers.get('content-type', '').lower()

                    if 'image' not in img_content_type or img_content_type.endswith(('gif', 'svg+xml')):
                         # print(f"Skipping non-target image content: {img_url} (Type: {img_content_type})")
                         continue

                    # Read content respecting stream
                    img_data = img_response.content # Read all content now
                    if img_data and len(img_data) > min_size_bytes:
                         images.append((img_data, img_url))
                         download_count += 1
                         print(f"Downloaded image {download_count} ({len(img_data)} bytes)")
                         if download_count >= 30: # Limit downloads
                              self.results_queue.put({"type": "text_only", "data": "Reached download limit (30) for this URL.\n"})
                              break
                    # else: print(f"Skipping small/empty image: {img_url}")

                except requests.exceptions.MissingSchema:
                    print(f"Skipping invalid URL (Missing Schema): {img_url}")
                except requests.exceptions.RequestException as req_err:
                    print(f"Failed download {img_url}: {req_err}")
                except Exception as e:
                     print(f"Generic error downloading image {img_url}: {e}")

        except requests.exceptions.RequestException as e:
            self.results_queue.put({"type": "text_only", "data": f"Failed to fetch base URL {url}: {e}\n"})
        except Exception as e:
            self.results_queue.put({"type": "text_only", "data": f"Error scraping URL {url}: {e}\n"})

        print(f"Finished scraping. Found {len(images)} suitable images.")
        return images


    def _analyze_image(self, image_path_or_data, source_name, size_bytes):
        """Analyzes a single image and sends a result package to the queue.
           Raises exceptions on loading/analysis failure."""
        img_tk = None
        processed_image = None
        # Ensure all keys exist for consistent summary structure
        summary = {"Male": 0, "Female": 0, "Else": 0, "Error": 0}
        detections = []
        result_text_summary = "Analysis Error." # Default message

        try:
            # --- Load image ---
            if isinstance(image_path_or_data, str):
                if not os.path.exists(image_path_or_data):
                    raise FileNotFoundError(f"File not found: {image_path_or_data}")
                img_cv = cv2.imread(image_path_or_data)
                if img_cv is None: raise ValueError(f"Could not read image file: {source_name}")
            elif isinstance(image_path_or_data, bytes):
                img_np = np.frombuffer(image_path_or_data, np.uint8)
                img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if img_cv is None: raise ValueError(f"Could not decode image data for: {source_name}")
            else: raise TypeError("Invalid image input type")

            # --- 1. Person Detection (YOLO) ---
            results_yolo = yolo_model.predict(img_cv, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD_PERSON, device=device, verbose=False)
            processed_image = img_cv.copy() # Work on copy even if no detections

            if results_yolo and len(results_yolo[0].boxes) > 0:
                person_boxes = results_yolo[0].boxes.xyxy.cpu().numpy().astype(int)
                for box in person_boxes:
                    x1, y1, x2, y2 = box
                    # Basic sanity check for box coordinates + clipping
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_cv.shape[1], x2), min(img_cv.shape[0], y2)
                    if x1 >= x2 or y1 >= y2: continue # Skip zero-area boxes after clipping

                    person_crop = img_cv[y1:y2, x1:x2]
                    if person_crop.size == 0: continue

                    # --- 2. Gender Classification (DeepFace) ---
                    gender = "Else"; color = COLOR_ELSE
                    try:
                        # Using retinaface for potentially better face detection
                        result_deepface = DeepFace.analyze(
                            img_path=person_crop,
                            actions=['gender'],
                            detector_backend='retinaface', # Changed backend
                            enforce_detection=True,
                            silent=True
                        )
                        if isinstance(result_deepface, list) and len(result_deepface) > 0:
                             analysis = result_deepface[0] # Take first detected face
                             dom_gender = analysis.get('dominant_gender')
                             if dom_gender == 'Man': gender = "Male"; color = COLOR_MALE
                             elif dom_gender == 'Woman': gender = "Female"; color = COLOR_FEMALE
                    except ValueError as e:
                         # Handle case where detector_backend couldn't find a face
                         if "Face could not be detected" in str(e) or "Detector backend cannot find face" in str(e):
                              gender = "Else"; color = COLOR_ELSE
                              # print(f"DeepFace Detector ({'retinaface'}) could not find face in YOLO crop for {source_name}. Marking as 'Else'.")
                         else: # Other DeepFace ValueError
                              gender = "Error"; color = COLOR_ELSE
                              print(f"DeepFace ValueError {source_name} (Box: {box}): {e}")
                    except Exception as e: # Catch any other DeepFace/analysis error
                         gender = "Error"; color = COLOR_ELSE
                         print(f"DeepFace Exception {source_name} (Box: {box}): {e}")

                    detections.append({'box': tuple(box), 'gender': gender, 'color': color})
                    summary[gender] = summary.get(gender, 0) + 1
            else:
                 detections.append({'box': None, 'gender': 'No persons detected', 'color': None})

            # --- 3. Draw Boxes & Prepare Output ---
            if detections and detections[0]['box'] is not None:
                 for det in detections:
                     if det['box']: # Check box exists
                         x1, y1, x2, y2 = det['box']
                         color = det['color']
                         label = det['gender']
                         # Draw rectangle on the *copied* image
                         cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                         # Label text params
                         font_scale = 0.5
                         thickness = 1
                         font = cv2.FONT_HERSHEY_SIMPLEX
                         (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                         # Ensure label background doesn't go off-screen
                         label_y = y1 - 5
                         bg_y1 = max(label_y - h, 0)
                         bg_y2 = label_y + 5 # Adjust background height slightly
                         # Draw background rectangle and text
                         cv2.rectangle(processed_image, (x1, bg_y1), (x1 + w, bg_y2), color, -1)
                         cv2.putText(processed_image, label, (x1, label_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

                 result_text_summary = f"Male: {summary['Male']}, Female: {summary['Female']}, Other: {summary['Else'] + summary['Error']}"
            else:
                 result_text_summary = "No persons detected."

            # Convert final processed image for Tkinter
            img_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail(LIST_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil)

        # Exceptions caught by _processing_worker, just prepare message here
        finally:
            # Always send a result package, even on error (img_tk might be None)
            self.results_queue.put({
                "type": "result_item",
                "image_tk": img_tk, # Will be None if conversion failed or error occurred before this
                "text": result_text_summary, # Contains error message if needed
                "summary": summary, # Includes error count
                "source_name": source_name,
                # Include size only if analysis *potentially* completed (img_tk created)
                "size_bytes": size_bytes if img_tk is not None and size_bytes >= 0 else 0
            })

    # --- Queue Checker (Runs in GUI Thread) ---
    def check_queue(self):
        """Periodically check the queue for messages."""
        try:
            while True: # Process all available messages
                message = self.results_queue.get_nowait()
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

                    # Update benchmark counts *only if* analysis was successful (image_tk exists)
                    if image_tk is not None:
                        self.total_images_processed += 1
                        # Only add size if it was positive (valid size was obtained)
                        if size_bytes > 0:
                           self.total_size_processed_bytes += size_bytes

                elif msg_type == "text_only":
                     print(f"Log: {message.get('data').strip()}")

                elif msg_type == "status":
                    msg_data = message.get("data")
                    if msg_data in ["Finished", "Error"]:
                        # --- Processing Finished ---
                        self.set_processing_state(False) # Enable controls etc.
                        end_time = time.monotonic()
                        duration = 0
                        if self.start_time:
                             duration = end_time - self.start_time

                        # Use counts received *from the worker* for final benchmark
                        final_image_count = message.get("count", 0)
                        final_total_size = message.get("size", 0)
                        has_errors = message.get("errors", False)

                        # Use counts accumulated in GUI thread for display consistency (total_images_processed reflects successful ones)
                        # final_image_count = self.total_images_processed
                        # final_total_size = self.total_size_processed_bytes

                        ips = (final_image_count / duration) if duration > 0 else 0
                        total_mb = final_total_size / (1024 * 1024)

                        final_status = "Idle." if msg_data == "Finished" else "Finished with errors."
                        benchmark_text = (
                            f"Benchmark: Processed {final_image_count} items " # Changed label to items
                            f"({total_mb:.2f} MB) in {duration:.2f}s "
                            f"({ips:.2f} items/sec)"
                        )

                        self.status_label.config(text=f"Status: {final_status}")
                        self.benchmark_label.config(text=benchmark_text)
                        self.start_time = None # Reset start time

                    else: # Intermediate status update (e.g., "Processing 5/10...")
                         if self.processing_active: # Only update if still processing
                              self.status_label.config(text=f"Status: {msg_data}")

        except queue.Empty:
            pass # No more messages in the queue
        except Exception as e:
            # Catch unexpected errors during GUI update
            print(f"Error processing queue message: {e}")
            import traceback
            traceback.print_exc()
            # Try to recover GUI state
            try:
                self.status_label.config(text="Status: GUI Update Error!")
                self.benchmark_label.config(text="Benchmark: Error")
                if self.processing_active:
                    self.set_processing_state(False) # Ensure processing stops if GUI breaks
            except Exception as e2:
                 print(f"Further error trying to update GUI after queue error: {e2}")

        finally:
            # Reschedule the check only if the window still exists
            if self.winfo_exists():
                 self.after(100, self.check_queue)


# --- Main Execution ---
if __name__ == "__main__":
    if not MODELS_LOADED:
        print("Application cannot start because models failed to load. Check console/log.")
        sys.exit(1)

    # Initialize app variable
    app = None
    try:
        app = GenderDetectorApp() # Create the app instance

        # Set theme *after* TkinterDnD.Tk() is initialized
        style = ttk.Style(app) # Pass app instance to Style
        available_themes = style.theme_names()
        # Order of preference for themes
        preferred_themes = ['clam', 'vista', 'xpnative', 'winnative', 'aqua']
        chosen_theme = 'default' # Fallback
        for theme in preferred_themes:
            if theme in available_themes:
                try:
                    style.theme_use(theme)
                    chosen_theme = theme
                    print(f"Using theme: {chosen_theme}")
                    break
                except tk.TclError:
                    print(f"Warning: Could not use theme '{theme}'.")

        if chosen_theme == 'default': print("Using default theme.")

        app.mainloop() # Start the main loop

    except KeyboardInterrupt:
        print("\nExiting application.")
    except Exception as e:
         # Catch unexpected errors during app initialization or mainloop
         print(f"\nAn unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()
    finally:
        # Optional: Add any cleanup code here if needed
        if app and app.winfo_exists():
            print("Attempting to destroy application window...")
            app.destroy()
        print("Application finished.")
        sys.exit(0) # Ensure clean exit