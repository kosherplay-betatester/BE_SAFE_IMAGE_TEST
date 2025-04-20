import tkinter as tk
from tkinter import filedialog, ttk
from tkinterdnd2 import *
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
import io

class GenderDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gender Detector")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load models
        self.yolo_model = YOLO('yolov8n.pt').to(self.device)
        self.gender_model = self.load_gender_model().to(self.device)
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())

        # GUI setup
        self.setup_gui()
        self.processed_images = []
        self.is_processing = False

    def load_gender_model(self):
        # Placeholder for gender classification model
        # Replace with a pre-trained ResNet or similar model fine-tuned for gender
        # Example: torchvision.models.resnet18(pretrained=True)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.eval()
        return model

    def setup_gui(self):
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(fill="both", expand=True)

        # Drag-and-drop support
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)

        # URL entry
        tk.Label(self.root, text="Enter URL:").pack()
        self.url_entry = tk.Entry(self.root, width=50)
        self.url_entry.pack()
        tk.Button(self.root, text="Fetch Images from URL", command=self.fetch_url_images).pack()

        # File/Directory selection
        tk.Button(self.root, text="Select Image/Directory", command=self.select_files).pack()

        # Progress bar
        self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress.pack()

        # Result display
        self.result_text = tk.Text(self.root, height=10, width=80)
        self.result_text.pack()

    def handle_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        self.process_files(files)

    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not files:
            files = filedialog.askdirectory()
            if files:
                files = [os.path.join(files, f) for f in os.listdir(files) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.process_files(files)

    def fetch_url_images(self):
        url = self.url_entry.get()
        if not url:
            return
        threading.Thread(target=self.scrape_images_from_url, args=(url,), daemon=True).start()

    def scrape_images_from_url(self, url):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img')
            image_urls = []
            for img in img_tags:
                src = img.get('src')
                if src:
                    src = urllib.parse.urljoin(url, src)
                    if src.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_urls.append(src)

            files = []
            for img_url in image_urls:
                try:
                    img_data = requests.get(img_url, timeout=5).content
                    img_name = os.path.join("temp", os.path.basename(img_url))
                    os.makedirs("temp", exist_ok=True)
                    with open(img_name, 'wb') as f:
                        f.write(img_data)
                    files.append(img_name)
                except Exception as e:
                    print(f"Error downloading {img_url}: {e}")
            self.process_files(files)
        except Exception as e:
            self.result_text.insert(tk.END, f"Error fetching URL: {e}\n")

    def process_files(self, files):
        if self.is_processing:
            return
        self.is_processing = True
        self.processed_images = []
        self.result_text.delete(1.0, tk.END)
        self.progress['value'] = 0
        self.progress['maximum'] = len(files)

        def process_file(file):
            try:
                img = cv2.imread(file)
                if img is None:
                    return None
                results = self.detect_persons(img)
                annotated_img = self.annotate_image(img, results)
                return file, annotated_img, results
            except Exception as e:
                return file, None, f"Error: {e}"

        def update_progress(future):
            result = future.result()
            if result:
                file, annotated_img, results = result
                if annotated_img is not None:
                    self.processed_images.append((file, annotated_img))
                    self.display_results(file, results)
            self.progress['value'] += 1
            if self.progress['value'] == self.progress['maximum']:
                self.is_processing = False
                self.display_image()

        for file in files:
            future = self.executor.submit(process_file, file)
            future.add_done_callback(update_progress)

    def detect_persons(self, img):
        results = self.yolo_model(img, classes=[0], conf=0.5)  # Class 0 for persons
        persons = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_img = img[y1:y2, x1:x2]
            gender = self.classify_gender(person_img)
            persons.append({'box': (x1, y1, x2, y2), 'gender': gender})
        return persons

    def classify_gender(self, img):
        # Placeholder for gender classification
        # Preprocess image: resize to 224x224, normalize, convert to tensor
        img = cv2.resize(img, (224, 224))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Replace with actual gender model inference
            # Example output: 0 for male, 1 for female, 2 for unknown
            output = torch.rand(1, 3).to(self.device)  # Dummy output
            gender_idx = torch.argmax(output, dim=1).item()
        
        return ['Male', 'Female', 'Unknown'][gender_idx]

    def annotate_image(self, img, persons):
        img_copy = img.copy()
        for person in persons:
            x1, y1, x2, y2 = person['box']
            gender = person['gender']
            color = (255, 0, 0) if gender == 'Male' else (255, 192, 203) if gender == 'Female' else (0, 255, 0)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_copy, gender, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return img_copy

    def display_results(self, file, results):
        if isinstance(results, str):
            self.result_text.insert(tk.END, f"{file}: {results}\n")
        else:
            self.result_text.insert(tk.END, f"{file}:\n")
            for i, person in enumerate(results):
                self.result_text.insert(tk.END, f"  Person {i+1}: {person['gender']}\n")
        self.result_text.see(tk.END)

    def display_image(self):
        if not self.processed_images:
            return
        file, annotated_img = self.processed_images[0]  # Display first image
        img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((800, 600), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = GenderDetectorApp(root)
    root.mainloop()