# Import Necessary Modules:

import sys
import concurrent.futures
import psutil
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit, QPushButton, QTextBrowser, QProgressBar, QLabel
from PyQt5.QtCore import QTimer, Qt, QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
import torch
from PIL import Image
import clip
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import urllib.parse
import gc
import base64


# Define the Image Processing Functions:

# Load the object detection model (YOLOv5)
model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the classification model (ViT)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_classification, preprocess = clip.load("ViT-B/32", device=device)

categories = ["man", "woman", "object"]

# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Function to get all image URLs from a webpage
def get_image_urls(webpage_url):
    if not webpage_url.startswith(('http://', 'https://')):
        return []
    
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
                return (image_url, "Person is too pixelized, skipped.", None)

            crop = image.crop(box)
            crops.append(crop)

    if crops:
        image_inputs = torch.stack([preprocess(crop) for crop in crops]).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)

        with torch.no_grad():
            image_features = model_classification.encode_image(image_inputs)
            text_features = model_classification.encode_text(text_inputs)

        with torch.no_grad():
            logits_per_image, _ = model_classification(image_inputs, text_inputs)
            probs = torch.softmax(logits_per_image, dim=-1).cpu().numpy()

        for i, crop_probs in enumerate(probs):
            scores = {category: probability for category, probability in zip(categories, crop_probs)}
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            highest_category = sorted_scores[0][0]
            woman_score = scores["woman"]
            man_score = scores["man"]

            if woman_score < 0.50 and man_score < 0.50:
                return (image_url, "Scores for both 'woman' and 'man' are below 50%, skipped.", None)

            if highest_category == "woman":
                border_color = "pink"
            elif highest_category == "man":
                border_color = "blue"
            else:
                border_color = "gray"

            # Resize the cropped image to a thumbnail (128x128)
            thumbnail = crop.resize((128, 128), Image.LANCZOS)
            thumbnail_bytes = BytesIO()
            thumbnail.save(thumbnail_bytes, format='PNG')
            thumbnail_base64 = base64.b64encode(thumbnail_bytes.getvalue()).decode('utf-8')

            html = f"""
            <div style="border: 5px solid {border_color}; padding: 5px; display: inline-block;">
                <img src='data:image/png;base64,{thumbnail_base64}' style='width: 128px; height: 128px;' />
            </div>
            <div style="display: flex; align-items: center;">
                <a href='{image_url}' style='color: {border_color}; margin-right: 5px;'>{image_url}</a>
                <div style="width: 10px; height: 10px; background-color: {border_color}; border-radius: 50%;'></div>
            </div>
            """
            for category, probability in sorted_scores:
                if category == sorted_scores[0][0]:
                    html += f"**{category.capitalize()} Score: {probability * 100:.2f}**<br>"
                elif category == sorted_scores[1][0]:
                    html += f"*{category.capitalize()} Score: {probability * 100:.2f}*<br>"
                else:
                    html += f"{category.capitalize()} Score: {probability * 100:.2f}<br>"
            html += f"Image probably includes: a {sorted_scores[0][0].capitalize()} \
                        {sorted_scores[0][1]*100:.2f}% or a {sorted_scores[1][0].capitalize()} {sorted_scores[1][1]*100:.2f}%"
            return (image_url, None, html)

    return (image_url, "No crops found.", None)


# Define the Worker and MainWindow Classes:

class WorkerSignals(QObject):
    result = pyqtSignal(tuple)

class Worker(QRunnable):
    def __init__(self, image_url):
        super().__init__()
        self.image_url = image_url
        self.signals = WorkerSignals()

    def run(self):
        image = load_image_from_url(self.image_url)
        result = process_image(image, self.image_url)
        self.signals.result.emit(result)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Content Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        self.layout = QVBoxLayout()

        self.url_label = QLabel("Enter a webpage URL to analyze:")
        self.layout.addWidget(self.url_label)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter a webpage URL to analyze")
        self.layout.addWidget(self.url_input)

        self.test_button = QPushButton("Test URL")
        self.test_button.clicked.connect(self.test_url)
        self.layout.addWidget(self.test_button)

        self.results_browser = QTextBrowser()
        self.layout.addWidget(self.results_browser)

        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.executor = QThreadPool()
        self.cpu_load_threshold = 80.0  # CPU load threshold in percentage
        self.pending_tasks = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_cpu_load)
        self.timer.start(1000)  # Check every second

    def check_cpu_load(self):
        cpu_load = psutil.cpu_percent()
        if cpu_load < self.cpu_load_threshold and self.pending_tasks:
            task = self.pending_tasks.pop(0)
            task.signals.result.connect(self.display_result)
            self.executor.start(task)

    def test_url(self):
        url = self.url_input.text()
        if not url.startswith(('http://', 'https://')):
            self.results_browser.append("Invalid URL. Please enter a valid URL starting with http:// or https://")
            return
        image_urls = get_image_urls(url)
        if image_urls:
            self.results_browser.append(f"Found {len(image_urls)} images on {url}")
            self.progress_bar.setMaximum(len(image_urls))
            self.progress_bar.setValue(0)
            for image_url in image_urls:
                worker = Worker(image_url)
                self.pending_tasks.append(worker)
            self.check_cpu_load()
        else:
            self.results_browser.append("No images found on the provided URL.")

    @pyqtSlot(tuple)
    def display_result(self, result):
        image_url, error_message, html = result
        if error_message:
            self.results_browser.append(f"<a href='{image_url}' style='color: gray;'>{image_url}</a> - {error_message}")
        elif html:
            self.results_browser.append(html)
        else:
            self.results_browser.append(f"<a href='{image_url}' style='color: gray;'>{image_url}</a> - No result.")
        self.progress_bar.setValue(self.progress_bar.value() + 1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
