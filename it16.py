import tkinter as tk
from tkinter import ttk, messagebox, Scrollbar, RIGHT, BOTTOM, Y, TOP, LEFT
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
import os

# --- Model Loading ---

# Global reference for the loading label
model_loading_label = None

def load_models():
    global model_detection, model_classification, device, model_loading_label
    try:
        model_loading_label.config(text="Loading YOLOv5...")
        model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model_loading_label.config(text="Loading CLIP...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_classification, preprocess = clip.load("ViT-B/32", device=device)
        model_loading_label.config(text="Models loaded! Ready.", fg="green")
        analyze_button.config(state="normal")
    except Exception as e:
        model_loading_label.config(text=f"Error loading models: {e}", fg="red")

# Categories for CLIP
categories = ["man", "woman", "object"]

# --- Global Variables ---
image_queue = Queue()
result_queue = Queue()
processing_thread = None

# --- Automatic Thread Count Detection ---
cpu_thread_count = os.cpu_count() or 1
gpu_thread_count = torch.cuda.device_count() * 4 if torch.cuda.is_available() else 0

# Set max_workers based on available threads
max_workers = cpu_thread_count + gpu_thread_count
executor = ThreadPoolExecutor(max_workers=max_workers)

start_time = 0
total_images_tested = 0
blue_images = 0
pink_images = 0
gray_images = 0

# --- Functions ---

def load_image_from_url(url):
    """Loads an image from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except (requests.exceptions.RequestException, IOError) as e:
        return f"Error loading image: {e}"

def get_image_urls(webpage_url):
    """Extracts all image URLs from a given webpage URL."""
    if not webpage_url.startswith(('http://', 'https://')):
        messagebox.showerror("Invalid URL", "Please enter a valid URL starting with http:// or https://")
        return []

    try:
        response = requests.get(webpage_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        image_urls = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                src = urllib.parse.urljoin(webpage_url, src)
                image_urls.append(src)
        return image_urls
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error", f"Error fetching webpage: {e}")
        return []

def process_images_from_queue():
    """Processes images from the queue in a separate thread."""
    global processing_thread, total_images_tested, blue_images, pink_images, gray_images

    while True:
        image_url = image_queue.get()
        if image_url is None:
            break

        image = load_image_from_url(image_url)
        if isinstance(image, str):
            result_queue.put((image_url, image))  # Error loading image
        else:
            results = process_image(image, image_url)
            result_queue.put((image_url, results))

            if isinstance(results, list):  # Check if results is a list
                total_images_tested += 1
                for _, border_color in results:
                    if border_color == "blue":
                        blue_images += 1
                    elif border_color == "pink":
                        pink_images += 1
                    else:
                        gray_images += 1

def process_image(image, image_url):
    """Processes a single image using YOLOv5 and CLIP."""
    results = model_detection(image)
    crops = []
    for *box, conf, cls in results.xyxy[0]:
        if results.names[int(cls)] == "person":
            box = [int(x) for x in box]
            bbox_area = (box[2] - box[0]) * (box[3] - box[1])
            total_image_area = image.width * image.height
            bbox_proportion = bbox_area / total_image_area
            if bbox_proportion < 0.1111:
                return "Person too small in image, skipped."
            crop = image.crop(box)
            crops.append(crop)

    if crops:
        image_inputs = torch.stack([preprocess(crop) for crop in crops]).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)
        with torch.no_grad():
            logits_per_image, _ = model_classification(image_inputs, text_inputs)
            probs = torch.softmax(logits_per_image, dim=-1).cpu().numpy()

        results = []
        for crop_probs in probs:
            scores = {category: probability for category, probability in zip(categories, crop_probs)}
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            highest_category = sorted_scores[0][0]

            if highest_category == "woman":
                border_color = "pink"
            elif highest_category == "man":
                border_color = "blue"
            else:
                border_color = "gray"

            results.append((sorted_scores, border_color))
        return results
    else:
        return "No people detected in the image."

def open_url(event):
    """Opens the URL in a web browser when the link is clicked."""
    webbrowser.open_new(event.widget.cget("text"))

def update_results():
    """Updates the GUI with results from the result queue."""
    global processing_thread, start_time

    if not result_queue.empty():
        image_url, result = result_queue.get()
        progress_bar['value'] += 1

        if isinstance(result, str):
            # Create a frame for error messages
            frame = tk.Frame(inner_result_frame, bd=2, relief="groove")
            frame.pack(pady=5, padx=10, fill=tk.X)
            error_label = tk.Label(
                frame, text=f"{image_url}: {result}", wraplength=500, justify="left"
            )
            error_label.pack(side=tk.LEFT, padx=5)
        else:
            create_result_frame(image_url, result)

    if processing_thread and image_queue.empty() and result_queue.empty():
        processing_thread.join()
        processing_thread = None
        end_time = time.time()
        messagebox.showinfo(
            "Info",
            f"Image analysis complete!\nTotal time: {end_time - start_time:.2f} seconds",
        )
        progress_bar["value"] = progress_bar["maximum"]
        progress_label.config(
            text=f"Completed in {end_time - start_time:.2f} seconds (Using {device.upper()})"
        )

    elapsed_time = time.time() - start_time if start_time > 0 else 0
    time_label.config(text=f"Time: {elapsed_time:.2f} seconds")
    stats_label.config(
        text=f"Tested: {total_images_tested} | Blue: {blue_images} | Pink: {pink_images} | Gray: {gray_images}"
    )

    window.after(100, update_results)

def analyze_url():
    """Starts the image analysis process."""
    global processing_thread, start_time, total_images_tested, blue_images, pink_images, gray_images
    global inner_result_frame

    # Clear previous results
    for widget in inner_result_frame.winfo_children():
        widget.destroy()

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
            progress_label.config(
                text="Processing (Using {})...".format(device.upper())
            )

            for url in image_urls:
                image_queue.put(url)

            processing_thread = threading.Thread(target=process_images_from_queue)
            processing_thread.start()
            update_results()
        else:
            messagebox.showinfo("Info", "No images found on the provided URL.")

def create_result_frame(image_url, result):
    """Creates a frame to display an image and its analysis results."""
    frame = tk.Frame(inner_result_frame, bd=2, relief="groove")
    frame.pack(pady=5, padx=10, fill=tk.X)

    try:
        image = load_image_from_url(image_url)
        if isinstance(image, str):
            tk.Label(frame, text=image).pack()
        else:
            image.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(image)

            canvas = tk.Canvas(frame, width=100, height=100, highlightthickness=5)
            canvas.pack(side=tk.LEFT, padx=5)

            for sorted_scores, border_color in result:
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                if border_color == "pink":
                    canvas.config(highlightbackground="pink")
                elif border_color == "blue":
                    canvas.config(highlightbackground="blue")
                else:
                    canvas.config(highlightbackground="gray")
                canvas.image = photo

                result_label = tk.Label(frame, text=f"Scores: ", fg=border_color)
                result_label.pack(anchor="w")
                for category, probability in sorted_scores:
                    if category == sorted_scores[0][0]:
                        result_label = tk.Label(
                            frame,
                            text=f"  **{category.capitalize()} Score: {probability * 100:.2f}**",
                            fg=border_color,
                        )
                        result_label.pack(anchor="w")
                    elif category == sorted_scores[1][0]:
                        result_label = tk.Label(
                            frame,
                            text=f"  *{category.capitalize()} Score: {probability * 100:.2f}*",
                            fg=border_color,
                        )
                        result_label.pack(anchor="w")
                    else:
                        result_label = tk.Label(
                            frame,
                            text=f"  {category.capitalize()} Score: {probability * 100:.2f}",
                            fg=border_color,
                        )
                        result_label.pack(anchor="w")

            link_label = tk.Label(frame, text=image_url, fg="blue", cursor="hand2")
            link_label.pack(anchor="w")
            link_label.bind("

# --- Create GUI ---

window = tk.Tk()
window.title("Image Content Analyzer")
window.state("zoomed")
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
    state="disabled",
)
analyze_button.pack(pady=10)

# --- Progress Bar ---
progress_label = tk.Label(
    window, text="Loading models...", bg="#f0f0f0", font=("Arial", 10)
)
progress_label.pack()
model_loading_label = progress_label

progress_bar = ttk.Progressbar(
    window, orient="horizontal", mode="determinate", length=600
)
progress_bar.pack(pady=5)

# --- Results Area (Scrollable) ---
result_label = tk.Label(
    window, text="Results:", bg="#f0f0f0", font=("Arial", 14, "bold")
)
result_label.pack(pady=(10, 0))

result_frame = tk.Frame(window)
result_frame.pack(expand=True, fill="both")

canvas = tk.Canvas(result_frame)
canvas.pack(side=LEFT, fill="both", expand=True)

scrollbar = Scrollbar(result_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind(
    "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

inner_result_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=inner_result_frame, anchor="nw")

# --- Time and Statistics ---
time_label = tk.Label(window, text="Time: 0 seconds", bg="#f0f0f0", font=("Arial", 10))
time_label.pack()

stats_label = tk.Label(
    window, text="Tested: 0 | Blue: 0 | Pink: 0 | Gray: 0", bg="#f0f0f0", font=("Arial", 10)
)
stats_label.pack()

# Start model loading after GUI is ready
window.after(100, load_models) 
window.after(100, update_results)
window.mainloop()