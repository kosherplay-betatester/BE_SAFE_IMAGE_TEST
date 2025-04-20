#!/usr/bin/env python3
"""
Gender Detection App using YOLOv8 and DeepFace

Requirements:
    pip install ultralytics deepface opencv-python PyQt5 requests beautifulsoup4 numpy torch torchvision
"""

import sys
import os
import cv2
import requests
import numpy as np
import torch
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QScrollArea, QGridLayout, QToolBar, QAction,
    QFileDialog, QStatusBar, QPushButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QPixmap, QImage, QIcon
from ultralytics import YOLO
from deepface import DeepFace

class DetectionWorker(QThread):
    """
    Worker thread for running detection and gender classification on a single image.
    """
    result = pyqtSignal(str, QImage)

    def __init__(self, source, model, device):
        super().__init__()
        self.source = source
        self.model = model
        self.device = device

    def run(self):
        img = cv2.imread(self.source)
        if img is None:
            return
        results = self.model(img, device=self.device)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.model.names[cls_id]
            if label == 'person':
                crop = img[y1:y2, x1:x2]
                try:
                    analysis = DeepFace.analyze(crop, actions=['gender'], enforce_detection=False)
                    gender = analysis.get('gender', 'Unknown')
                except Exception:
                    gender = 'Unknown'
                color = (255, 0, 0) if gender.lower().startswith('m') else (255, 105, 180)
                text = gender
            else:
                color = (0, 255, 0)
                text = label
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.result.emit(self.source, qimg)

class MainWindow(QMainWindow):
    """
    Main application window.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gender Detection App")
        self.resize(1200, 800)
        self.workers = []  # Keep references to workers

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt')
        try:
            DeepFace.analyze(np.zeros((100,100,3), dtype=np.uint8), actions=['gender'], enforce_detection=False)
        except Exception:
            pass

        self.init_ui()

    def init_ui(self):
        # Toolbar with actions
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        open_files_action = QAction(QIcon.fromTheme('document-open'), "Open Images...", self)
        open_files_action.triggered.connect(self.open_files)
        toolbar.addAction(open_files_action)

        open_dir_action = QAction(QIcon.fromTheme('folder-open'), "Open Directory...", self)
        open_dir_action.triggered.connect(self.open_directory)
        toolbar.addAction(open_dir_action)

        clear_action = QAction(QIcon.fromTheme('edit-clear'), "Clear Results", self)
        clear_action.triggered.connect(self.clear_results)
        toolbar.addAction(clear_action)

        # Central widget layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # URL input row
        url_layout = QHBoxLayout()
        self.urlInput = QLineEdit()
        self.urlInput.setPlaceholderText("Enter URL and press Enter")
        self.urlInput.returnPressed.connect(self.on_url_entered)
        url_layout.addWidget(self.urlInput)
        fetch_btn = QPushButton("Fetch Images")
        fetch_btn.clicked.connect(self.on_url_entered)
        url_layout.addWidget(fetch_btn)
        main_layout.addLayout(url_layout)

        # Drag & Drop label
        self.dropArea = QLabel("Drag & drop images or directories here")
        self.dropArea.setFixedHeight(80)
        self.dropArea.setAlignment(Qt.AlignCenter)
        self.dropArea.setStyleSheet("border: 2px dashed #aaa; background:#f9f9f9;")
        self.dropArea.installEventFilter(self)
        main_layout.addWidget(self.dropArea)

        # Scroll area with grid layout
        self.scrollArea = QScrollArea()
        self.scrollWidget = QWidget()
        self.gridLayout = QGridLayout(self.scrollWidget)
        self.gridLayout.setSpacing(10)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.scrollWidget)
        main_layout.addWidget(self.scrollArea)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.update_status()

    def update_status(self):
        count = self.scrollWidget.layout().count()
        self.status.showMessage(f"Results: {count}")

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", os.getcwd(),
                                                "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if files:
            self.process_images(files)

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        if directory:
            files = []
            for root, _, fs in os.walk(directory):
                for f in fs:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', 'gif')):
                        files.append(os.path.join(root, f))
            self.process_images(files)

    def clear_results(self):
        for i in reversed(range(self.gridLayout.count())):
            widget = self.gridLayout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.update_status()

    def eventFilter(self, source, event):
        if source is self.dropArea:
            if event.type() == QEvent.DragEnter and event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
            if event.type() == QEvent.Drop:
                paths = [u.toLocalFile() for u in event.mimeData().urls()]
                self.process_images(self.expand_paths(paths))
                return True
        return super().eventFilter(source, event)

    def expand_paths(self, paths):
        image_paths = []
        for path in paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', 'gif')):
                            image_paths.append(os.path.join(root, f))
            elif path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', 'gif')):
                image_paths.append(path)
        return image_paths

    def on_url_entered(self):
        url = self.urlInput.text().strip()
        if not url:
            return
        try:
            r = requests.get(url, timeout=5)
            soup = BeautifulSoup(r.text, 'html.parser')
            imgs = soup.find_all('img')
            temp_dir = os.path.join(os.getcwd(), 'temp_images')
            os.makedirs(temp_dir, exist_ok=True)
            files = []
            for i, tag in enumerate(imgs):
                src = tag.get('src')
                if not src: continue
                img_url = urljoin(url, src)
                try:
                    resp = requests.get(img_url, timeout=5)
                    arr = np.frombuffer(resp.content, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None: continue
                    path = os.path.join(temp_dir, f"img_{i}.jpg")
                    cv2.imwrite(path, img)
                    files.append(path)
                except: continue
            self.process_images(files)
        except Exception as e:
            self.status.showMessage(f"Error: {e}")

    def process_images(self, image_paths):
        columns = 3
        for idx, path in enumerate(image_paths):
            worker = DetectionWorker(path, self.model, self.device)
            worker.result.connect(self.add_result)
            worker.finished.connect(lambda w=worker: self.cleanup_worker(w))
            self.workers.append(worker)
            worker.start()

    def add_result(self, path, qimg):
        lbl = QLabel()
        pix = QPixmap.fromImage(qimg)
        lbl.setPixmap(pix.scaled(350, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        idx = self.gridLayout.count()
        row, col = divmod(idx, 3)
        self.gridLayout.addWidget(lbl, row, col)
        self.update_status()

    def cleanup_worker(self, worker):
        if worker in self.workers:
            self.workers.remove(worker)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
