import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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

# Load models
model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = "cuda" if torch.cuda.is_available() else "cpu"
model_classification, preprocess = clip.load("ViT-B/32", device=device)
categories = ["man", "woman", "object"]

# Global variables
image_queue = Queue()
result_queue = Queue()
processing_thread = None
executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on your system

# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Function to get all image URLs from a webpage
def get_image_urls(webpage_url):
    if not webpage_url.startswith(('http://', 'https://')):
        messagebox.showerror("Error", "Invalid URL. Please enter a valid URL starting with http:// or https://")
        return []

    response = requests.get(webpage_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_urls = []
    for img in soup.find_all('img'):
        src = img.get('src')
        if src:
            src = urllib.parse.urljoin(webpage_url, src)
            image_urls.append(src)
    return image_urls

# Function to process images from the queue
def process_images_from_queue():
    global processing_thread

    while True:
        image_url = image_queue.get()
        if image_url is None:
            break  # Signal to stop the thread

        try:
            image = load_image_from_url(image_url)
            results = process_image(image, image_url)
            result_queue.put((image_url, results))
        except Exception as e:
            result_queue.put((image_url, f"Error: {e}"))

# Function to process an image
def process_image(image, image_url):
    results = model_detection(image)
    crops = []
    for *box, conf, cls in results.xyxy[0]:
        if results.names[int(cls)] == "person":
            box = [int(x) for x in box]
            bbox_area = (box[2] - box[0]) * (box[3] - box[1])
            total_image_area = image.width * image.height
            bbox_proportion = bbox_area / total_image_area
            if bbox_proportion < 0.1111:
                return "Person is too pixelized, skipped."
            crop = image.crop(box)
            crops.append(crop)

    if crops:
        image_inputs = torch.stack([preprocess(crop) for crop in crops]).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)
        with torch.no_grad():
            image_features = model_classification.encode_image(image_inputs)
            text_features = model_classification.encode_text(text_inputs)
            logits_per_image, _ = model_classification(image_inputs, text_inputs)
            probs = torch.softmax(logits_per_image, dim=-1).cpu().numpy()

        results = []
        for i, crop_probs in enumerate(probs):
            scores = {category: probability for category, probability in zip(categories, crop_probs)}
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            highest_category = sorted_scores[0][0]
            woman_score = scores["woman"]
            man_score = scores["man"]
            if woman_score < 0.50 and man_score < 0.50:
                return "Scores for both 'woman' and 'man' are below 50%, skipped."
            results.append(sorted_scores)

        return results 
    else:
        return "No people detected in the image."


# Function to start analyzing images
def analyze_url():
    global processing_thread

    webpage_url = url_entry.get()
    if webpage_url:
        image_urls = get_image_urls(webpage_url)
        if image_urls:
            progress_bar["value"] = 0
            progress_bar["maximum"] = len(image_urls)

            for url in image_urls:
                image_queue.put(url)

            processing_thread = threading.Thread(target=process_images_from_queue)
            processing_thread.start()

            update_results() 
        else:
            messagebox.showinfo("Info", "No images found on the provided URL.")

# Function to update results in the GUI
def update_results():
    global processing_thread

    if not result_queue.empty():
        image_url, result = result_queue.get()
        if isinstance(result, str):
            result_text.insert(tk.END, f"{image_url}: {result}\n")
        else:
            result_text.insert(tk.END, f"{image_url}:\n")
            for sorted_scores in result:
                for category, probability in sorted_scores:
                    if category == sorted_scores[0][0]:
                        result_text.insert(tk.END, f"  **{category.capitalize()} Score: {probability * 100:.2f}**\n")
                    elif category == sorted_scores[1][0]:
                        result_text.insert(tk.END, f"  *{category.capitalize()} Score: {probability * 100:.2f}*\n")
                    else:
                        result_text.insert(tk.END, f"  {category.capitalize()} Score: {probability * 100:.2f}\n")

        progress_bar["value"] += 1
        progress_bar.update()

    if processing_thread and image_queue.empty() and result_queue.empty():
        image_queue.put(None) # Signal the thread to stop
        processing_thread.join() 
        processing_thread = None
        messagebox.showinfo("Info", "Image analysis complete!")

    window.after(100, update_results)  # Check for updates every 100ms

# GUI Setup
window = tk.Tk()
window.title("Image Content Analyzer")

url_label = tk.Label(window, text="Enter a webpage URL to analyze:")
url_label.pack()

url_entry = tk.Entry(window, width=50)
url_entry.pack()

analyze_button = tk.Button(window, text="Analyze", command=analyze_url)
analyze_button.pack()

progress_bar = ttk.Progressbar(window, orient="horizontal", mode="determinate")
progress_bar.pack()

result_label = tk.Label(window, text="Results:")
result_label.pack()

result_text = tk.Text(window, wrap=tk.WORD)
result_text.pack()

window.after(100, update_results)  # Initial call to start checking for results

window.mainloop()