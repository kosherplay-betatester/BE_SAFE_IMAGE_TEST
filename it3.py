import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import clip
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
import cv2

# Load the object detection model (YOLOv5)
model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the classification model (ViT)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_classification, preprocess = clip.load("ViT-B/32", device=device)

categories = ["man", "woman"]

# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Function to get all image URLs from a webpage
def get_image_urls(webpage_url):
    response = requests.get(webpage_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_urls = []
    for img in soup.find_all('img'):
        src = img.get('src')
        if src:
            # Convert relative URLs to absolute URLs
            src = urllib.parse.urljoin(webpage_url, src)
            image_urls.append(src)
    return image_urls

# Function to get a thumbnail from a video
def get_video_thumbnail(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        return image
    return None

# Function to process an image
def process_image(image, result_queue):
    try:
        # Ensure image is RGB (3 channels)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Resize image to a consistent size
        image = image.resize((640, 640))

        # Load image into GPU memory if available
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

        # Run object detection on the image
        results = model_detection(image_tensor)

        # For each detected person, run classification
        crops = []
        for det in results.xyxy[0]:
            *box, conf, cls = det
            if model_detection.names[int(cls)] == 'person':
                box = [int(x) for x in box]
                bbox_area = (box[2] - box[0]) * (box[3] - box[1])
                total_image_area = image.width * image.height
                bbox_proportion = bbox_area / total_image_area
                if bbox_proportion < 0.1111:
                    continue
                crop = image.crop(box)
                crops.append(crop)

        if crops:
            image_inputs = torch.stack([preprocess(crop).to(device) for crop in crops])
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)

            # Calculate features
            with torch.no_grad():
                image_features = model_classification.encode_image(image_inputs)
                text_features = model_classification.encode_text(text_inputs)

            # Calculate the similarity of the image to each category
            with torch.no_grad():
                logits_per_image, _ = model_classification(image_inputs, text_inputs)
                probs = torch.softmax(logits_per_image, dim=-1).cpu().numpy()

            results = []
            for i, crop_probs in enumerate(probs):
                crop_result = {'crop': crops[i], 'probs': crop_probs}
                results.append(crop_result)
            result_queue.put(results)
    except Exception as e:
        result_queue.put(f"Error processing image: {e}")

# GUI setup
class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.image_urls = []
        self.result_queue = queue.Queue()
        self.current_image_index = 0

        self.setup_ui()

    def setup_ui(self):
        self.root.title("Image Content Analyzer")
        self.root.state('zoomed')  # Maximize the window

        self.url_frame = tk.Frame(self.root, padx=10, pady=10)
        self.url_frame.pack()

        self.url_label = tk.Label(self.url_frame, text="Enter URL:")
        self.url_label.pack(side=tk.LEFT)

        self.url_entry = tk.Entry(self.url_frame, width=50)
        self.url_entry.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(self.url_frame, text="Load URL", command=self.load_url)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.paste_link_label = tk.Label(self.root, text="Or drag and drop files or links here:", pady=10)
        self.paste_link_label.pack()
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop)

        self.progress = ttk.Progressbar(self.root, length=100, mode='determinate')
        self.progress.pack(pady=10)

        self.log_text = tk.Text(self.root, height=5, state=tk.DISABLED)
        self.log_text.pack(fill=tk.X, padx=10, pady=5)

        self.table_frame = tk.Frame(self.root, padx=10, pady=10)
        self.table_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.table_frame)
        self.scroll_y = tk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(self.table_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)

        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    def load_url(self):
        webpage_url = self.url_entry.get()
        if webpage_url:
            threading.Thread(target=self.fetch_images_from_url, args=(webpage_url,)).start()
        else:
            messagebox.showwarning("Input Error", "Please enter a URL.")

    def fetch_images_from_url(self, webpage_url):
        try:
            self.image_urls = get_image_urls(webpage_url)
            if self.image_urls:
                self.update_log(f"Found {len(self.image_urls)} images. Starting processing...")
                self.start_processing()
            else:
                self.update_log("No images found at the provided URL.", "warning")
        except Exception as e:
            self.update_log(f"Error loading URL: {e}", "error")

    def drop(self, event):
        dropped_files = self.root.tk.splitlist(event.data)
        for file in dropped_files:
            if os.path.isdir(file):
                for root, _, files in os.walk(file):
                    for name in files:
                        if name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov')):
                            self.image_urls.append(os.path.join(root, name))
            elif file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov')):
                self.image_urls.append(file)
            elif file.startswith('http'):  # Check if the dropped item is a URL
                threading.Thread(target=self.fetch_images_from_url, args=(file,)).start()
                return
        self.start_processing()

    def start_processing(self):
        if self.image_urls:
            self.progress['value'] = 0
            self.current_image_index = 0
            self.root.update_idletasks()
            self.load_next_image()
        else:
            self.update_log("No images to process.", "error")

    def load_next_image(self):
        if self.current_image_index < len(self.image_urls):
            media_url = self.image_urls[self.current_image_index]
            self.progress['value'] = (self.current_image_index + 1) / len(self.image_urls) * 100
            self.root.update_idletasks()

            threading.Thread(target=self.process_media_thread, args=(media_url,)).start()
        else:
            self.update_log("Processing complete.", "info")

    def process_media_thread(self, media_url):
        if media_url.startswith('http'):
            image = load_image_from_url(media_url)
        elif media_url.lower().endswith(('.mp4', '.avi', '.mov')):
            image = get_video_thumbnail(media_url)
        else:
            image = Image.open(media_url)
        process_image(image, self.result_queue)

        # Retrieve results
        results = self.result_queue.get()
        if isinstance(results, str):  # Check if there's an error message
            self.update_log(results, "error")
        else:
            self.display_results(image, results)

    def display_results(self, image, results):
        # Display the original image
        img = ImageTk.PhotoImage(image)
        image_label = tk.Label(self.scrollable_frame, image=img)
        image_label.image = img
        image_label.grid(row=self.current_image_index, column=0, padx=5, pady=5)

        result_text = tk.Text(self.scrollable_frame, width=50, height=10)
        result_text.grid(row=self.current_image_index, column=1, padx=5, pady=5)

        for result in results:
            crop = result['crop']
            crop_img = ImageTk.PhotoImage(crop)
            crop_label = tk.Label(self.scrollable_frame, image=crop_img)
            crop_label.image = crop_img
            crop_label.grid(row=self.current_image_index, column=2, padx=5, pady=5)

            scores = {category: probability for category, probability in zip(categories, result['probs'])}
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

            for category, probability in sorted_scores:
                if category == sorted_scores[0][0]:
                    result_text.insert(tk.END, f"{category.capitalize()} Score: {probability * 100:.2f}\n")
                elif category == sorted_scores[1][0]:
                    result_text.insert(tk.END, f"*{category.capitalize()} Score: {probability * 100:.2f}*\n")
                else:
                    result_text.insert(tk.END, f"{category.capitalize()} Score: {probability * 100:.2f}\n")

            result_text.insert(tk.END, f"Image probably includes: a {sorted_scores[0][0].capitalize()} \
                                        {sorted_scores[0][1] * 100:.2f}% or a {sorted_scores[1][0].capitalize()} \
                                        {sorted_scores[1][1] * 100:.2f}%\n")

        self.current_image_index += 1
        self.load_next_image()

    def update_log(self, message, level="info"):
        self.log_text.config(state=tk.NORMAL)
        if level == "error":
            self.log_text.insert(tk.END, f"ERROR: {message}\n", "error")
            self.log_text.tag_config("error", foreground="red")
        elif level == "warning":
            self.log_text.insert(tk.END, f"WARNING: {message}\n", "warning")
            self.log_text.tag_config("warning", foreground="orange")
        else:
            self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)

# Main
if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()
