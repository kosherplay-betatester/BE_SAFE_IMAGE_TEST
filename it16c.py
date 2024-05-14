import sys
import psutil
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit, QPushButton, QTextBrowser, QProgressBar, QLabel, QSlider
from PyQt5.QtCore import QTimer, Qt, QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
import torch
from PIL import Image, UnidentifiedImageError
import clip
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import urllib.parse
import gc
import base64

# Load the object detection model (YOLOv5)
model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the classification model (ViT)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_classification, preprocess = clip.load("ViT-B/32", device=device)

categories = ["man", "woman", "object"]

def load_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image from URL {url}")
        return None

def get_image_urls(webpage_url):
    if not webpage_url.startswith(('http://', 'https://')):
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

def process_image(image, image_url):
    if image is None:
        return (image_url, "Cannot identify image.", None)
    image = image.resize((640, 640))
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
            if highest_category == "woman":
                border_color = "pink"
            elif highest_category == "man":
                border_color = "blue"
            else:
                border_color = "gray"
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
                html += f"{category.capitalize()} Score: {probability * 100:.2f}%<br>"
            return (image_url, None, html)
    return (image_url, "No crops found.", None)

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
        self.threads_label = QLabel("Adjust the number of threads:")
        self.layout.addWidget(self.threads_label)
        self.threads_slider = QSlider(Qt.Horizontal)
        self.threads_slider.setMinimum(1)
        self.threads_slider.setMaximum(psutil.cpu_count())
        self.threads_slider.setValue(psutil.cpu_count(logical=False))
        self.threads_slider.valueChanged.connect(self.update_thread_count)
        self.layout.addWidget(self.threads_slider)
        self.threads_count_label = QLabel(f"Number of threads: {self.threads_slider.value()}")
        self.layout.addWidget(self.threads_count_label)
        self.status_label = QLabel("Status: Idle")
        self.layout.addWidget(self.status_label)
        self.results_browser = QTextBrowser()
        self.layout.addWidget(self.results_browser)
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)
        self.executor = QThreadPool()
        self.executor.setMaxThreadCount(self.threads_slider.value())
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)  # Update status every second

    def update_thread_count(self):
        self.executor.setMaxThreadCount(self.threads_slider.value())
        self.threads_count_label.setText(f"Number of threads: {self.threads_slider.value()}")

    def update_status(self):
        cpu_load = psutil.cpu_percent()
        self.status_label.setText(f"Status: {self.executor.activeThreadCount()} threads running, CPU load: {cpu_load:.1f}%")

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
                worker.signals.result.connect(self.display_result)
                self.executor.start(worker)
        else:
            self.results_browser.append("No images found on the provided URL.")

    @pyqtSlot(tuple)
    def display_result(self, result):
        image_url, error_message, html = result
        if error_message:
            self.results_browser.append(f"<a href='{image_url}' style='color: gray;'>{image_url}</a> - {error_message}")
        elif html:
            self.results_browser.append(html)
        self.progress_bar.setValue(self.progress_bar.value() + 1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
