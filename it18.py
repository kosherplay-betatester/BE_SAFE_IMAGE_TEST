import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scrollbar, RIGHT, BOTTOM, Y, X, TOP, LEFT
import threading
from PIL import Image, ImageTk
import torch
import clip
from io import BytesIO
import requests
from torchvision import transforms
import numpy as np
from bs4 import BeautifulSoup
import urllib.parse
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import webbrowser
import time

# --- Model Loading ---

# Load models on startup
model_loading_label = None # Global reference to the loading label

def load_models():
    global model_detection, model_classification, device, model_loading_label
    try:
        model_loading_label.config(text="Loading YOLOv5...")
        model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model_loading_label.config(text="Loading CLIP...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_classification, preprocess = clip.load("ViT-B/32", device=device)
        model_loading_label.config(text="Models loaded! Ready.", fg="green")
        analyze_button.config(state="normal")  # Enable Analyze button
    except Exception as e:
        model_loading_label.config(text=f"Error loading models: {e}", fg="red")

# Start model loading in a separate thread
threading.Thread(target=load_models).start()

# Categories for CLIP
categories = ["man", "woman", "object"]

# --- Global Variables ---
image_queue = Queue()
result_queue = Queue()
processing_thread = None
executor = ThreadPoolExecutor(max_workers=4)

start_time = 0
total_images_tested = 0
blue_images = 0
pink_images = 0
gray_images = 0

# --- Functions ---

# ... (load_image_from_url, get_image_urls, process_images_from_queue, 
#      process_image, open_url remain the same) ...

def create_result_frame(image_url, result):
    """Creates a visually appealing frame for each image result."""
    # ... (This function remains the same) ...

def update_results():
    """Updates the GUI with results from the result queue."""
    global processing_thread, start_time

    while not result_queue.empty():
        image_url, result = result_queue.get()
        progress_bar["value"] += 1

        if isinstance(result, str):
            # Create a label within the frame for error messages
            frame = tk.Frame(result_text, bd=2, relief="groove", bg="#f5f5f5")
            frame.pack(pady=5, padx=10, fill=tk.X)
            error_label = tk.Label(frame, text=f"{image_url}: {result}", wraplength=500, justify="left", bg="#f5f5f5")
            error_label.pack(side=tk.LEFT, padx=5)
        else:
            create_result_frame(image_url, result)

    if processing_thread and image_queue.empty() and result_queue.empty():
        processing_thread.join()
        processing_thread = None
        end_time = time.time()
        messagebox.showinfo(
            "Info", f"Image analysis complete!\nTotal time: {end_time - start_time:.2f} seconds"
        )
        progress_bar["value"] = progress_bar["maximum"]
        progress_label.config(text=f"Completed in {end_time - start_time:.2f} seconds (Using {device.upper()})")

    # Update time and statistics labels
    elapsed_time = time.time() - start_time if start_time > 0 else 0
    time_label.config(text=f"Time: {elapsed_time:.2f} seconds")
    stats_label.config(
        text=f"Tested: {total_images_tested} | Blue: {blue_images} | Pink: {pink_images} | Gray: {gray_images}"
    )

    window.after(100, update_results)

def analyze_url():
    """Starts the image analysis process."""
    global processing_thread, start_time, total_images_tested, blue_images, pink_images, gray_images

    webpage_url = url_entry.get()
    if webpage_url:
        image_urls = get_image_urls(webpage_url)
        if image_urls:
            start_time = time.time()
            total_images_tested = 0
            blue_images = 0
            pink_images = 0
            gray_images = 0

            progress_bar["value"] = 0
            progress_bar["maximum"] = len(image_urls)
            progress_bar["mode"] = "determinate"
            progress_label.config(text="Processing (Using {})...".format(device.upper()))

            for url in image_urls:
                image_queue.put(url)

            processing_thread = threading.Thread(target=process_images_from_queue)
            processing_thread.start()
            update_results()
        else:
            messagebox.showinfo("Info", "No images found on the provided URL.")

# --- GUI Setup ---

window = tk.Tk()
window.title("Image Content Analyzer")
# Make the window almost maximized
window.state('zoomed') 
window.configure(bg="#f0f0f0")

# --- URL Input ---
url_label = tk.Label(
    window, text="Enter a webpage URL:", bg="#f0f0f0", font=("Arial", 12)
)
url_label.pack(pady=(10, 0))

url_entry = tk.Entry(window, width=70, font=("Arial", 12))
url_entry.pack(pady=(0, 10))

# --- Analyze Button ---
analyze_button = tk.Button(
    window,
    text="Analyze",
    command=analyze_url,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12),
    relief=tk.RAISED,
    bd=3,
    state="disabled"  # Initially disabled until models are loaded
)
analyze_button.pack(pady=10)

# --- Progress Bar ---
progress_label = tk.Label(window, text="Loading models...", bg="#f0f0f0", font=("Arial", 10))
progress_label.pack()
model_loading_label = progress_label  # Assign to the global reference

progress_bar = ttk.Progressbar(window, orient="horizontal", mode="determinate", length=600)
progress_bar.pack(pady=5)

# --- Results Area ---
result_label = tk.Label(
    window, text="Results:", bg="#f0f0f0", font=("Arial", 14, "bold")
)
result_label.pack(pady=(10, 0))

# --- Make the result area scrollable ---
result_frame = tk.Frame(window)
result_frame.pack(expand=True, fill="both")

canvas = tk.Canvas(result_frame)
canvas.pack(side=LEFT, fill="both", expand=True)

scrollbar = Scrollbar(result_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

result_text = tk.Frame(canvas, bg="#f0f0f0")
canvas.create_window((0, 0), window=result_text, anchor="nw")

# --- Time and Statistics ---
time_label = tk.Label(window, text="Time: 0 seconds", bg="#f0f0f0", font=("Arial", 10))
time_label.pack()

stats_label = tk.Label(
    window, text="Tested: 0 | Blue: 0 | Pink: 0 | Gray: 0", bg="#f0f0f0", font=("Arial", 10)
)
stats_label.pack()

window.after(100, update_results)
window.mainloop()