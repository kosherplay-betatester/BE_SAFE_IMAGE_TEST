import os
import sys
import threading
import queue
import requests
import numpy as np
import cv2
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QPushButton, QFileDialog, QScrollArea, 
                            QGridLayout, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from bs4 import BeautifulSoup
import traceback
import urllib.parse

class GenderDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gender Detection App")
        self.setMinimumSize(1000, 800)
        
        # Initialize UI first so we can update status
        self.init_ui()
        
        # Then initialize models with status updates
        self.status_label.setText("Loading models...")
        QApplication.processEvents()
        self.init_models()
        self.status_label.setText("Ready")
        
    def init_models(self):
        try:
            # Initialize YOLO model for person detection
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.yolo_model.to(self.device)
            self.yolo_model.classes = [0]  # Only detect persons (class 0 in COCO)
            
            # Initialize face detection with Haar Cascade
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(face_cascade_path):
                print(f"Error: Face cascade file not found at {face_cascade_path}")
                # Try to find it in other possible locations
                if os.path.exists('haarcascade_frontalface_default.xml'):
                    face_cascade_path = 'haarcascade_frontalface_default.xml'
                else:
                    print("Downloading face cascade file...")
                    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                    response = requests.get(url)
                    with open('haarcascade_frontalface_default.xml', 'wb') as f:
                        f.write(response.content)
                    face_cascade_path = 'haarcascade_frontalface_default.xml'
            
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if self.face_cascade.empty():
                print("Warning: Failed to load face cascade classifier")
                
            # Create a thread pool for processing
            self.processing_queue = queue.Queue()
            self.result_queue = queue.Queue()
            self.worker_threads = []
            self.num_threads = max(1, os.cpu_count() - 1)  # Use all but one CPU core
            self.stop_threads = False
            
            # Start worker threads
            for _ in range(self.num_threads):
                t = threading.Thread(target=self.worker_function)
                t.daemon = True
                t.start()
                self.worker_threads.append(t)
                
            print("Models initialized successfully")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to initialize models: {str(e)}")
    
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Status label
        self.status_label = QLabel("Loading...")
        main_layout.addWidget(self.status_label)
        
        # URL input section
        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        self.url_input = QLineEdit()
        url_button = QPushButton("Fetch Images")
        url_button.clicked.connect(self.fetch_images_from_url)
        
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        url_layout.addWidget(url_button)
        
        main_layout.addLayout(url_layout)
        
        # File/directory selection section
        file_layout = QHBoxLayout()
        file_button = QPushButton("Select Files")
        file_button.clicked.connect(self.select_files)
        dir_button = QPushButton("Select Directory")
        dir_button.clicked.connect(self.select_directory)
        
        file_layout.addWidget(file_button)
        file_layout.addWidget(dir_button)
        file_layout.addStretch()
        
        main_layout.addLayout(file_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Drop area with instruction
        self.drop_area = DropArea(self)
        self.drop_area.setMinimumHeight(100)
        self.drop_area.file_dropped.connect(self.process_dropped_files)
        main_layout.addWidget(self.drop_area)
        
        # Results area
        results_label = QLabel("Results:")
        main_layout.addWidget(results_label)
        
        # Scroll area for results
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        
        scroll_area.setWidget(self.results_widget)
        main_layout.addWidget(scroll_area)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Create result updater thread
        self.result_updater = ResultUpdater(self.result_queue)
        self.result_updater.result_ready.connect(self.update_results)
        self.result_updater.start()
    
    def select_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_paths:
            self.process_images(file_paths)
    
    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
            
            if image_files:
                self.process_images(image_files)
            else:
                self.status_label.setText("No image files found in the selected directory")
    
    def fetch_images_from_url(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Required", "Please enter a URL")
            return
        
        # Update status
        self.status_label.setText(f"Fetching images from {url}...")
        QApplication.processEvents()
        
        # Start a new thread to fetch images
        fetch_thread = threading.Thread(target=self.download_images_from_url, args=(url,))
        fetch_thread.daemon = True
        fetch_thread.start()
    
    def download_images_from_url(self, url):
        try:
            # Add http:// if missing from URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            print(f"Fetching images from URL: {url}")
            
            # Make request to the URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Get base URL for resolving relative paths
            parsed_url = urllib.parse.urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all image elements
            img_tags = soup.find_all('img')
            print(f"Found {len(img_tags)} img tags")
            
            # Extract image URLs
            img_urls = set()  # Use a set to avoid duplicates
            
            for img in img_tags:
                # Try different possible image attributes
                src = img.get('src') or img.get('data-src') or img.get('data-original')
                if src:
                    img_urls.add(src)
            
            # Process background images in style attributes
            for tag in soup.find_all(lambda t: t.has_attr('style') and 'url(' in t['style']):
                style = tag['style']
                start_idx = style.find('url(')
                if start_idx != -1:
                    start_idx += 4  # Skip 'url('
                    end_idx = style.find(')', start_idx)
                    if end_idx != -1:
                        bg_url = style[start_idx:end_idx].strip('\'"')
                        img_urls.add(bg_url)
            
            # Resolve relative URLs
            resolved_urls = []
            for img_url in img_urls:
                # Skip data URLs
                if img_url.startswith('data:'):
                    continue
                    
                # Handle various URL formats
                if img_url.startswith('//'):
                    img_url = parsed_url.scheme + ':' + img_url
                elif img_url.startswith('/'):
                    img_url = base_url + img_url
                elif not img_url.startswith(('http://', 'https://')):
                    # If this is a relative URL, join with the base URL
                    img_url = urllib.parse.urljoin(url, img_url)
                
                resolved_urls.append(img_url)
            
            print(f"Found {len(resolved_urls)} unique image URLs")
            
            # Create temp directory for downloaded images
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_images')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download images
            downloaded_images = []
            for i, img_url in enumerate(resolved_urls):
                try:
                    print(f"Downloading image {i+1}/{len(resolved_urls)}: {img_url}")
                    
                    # Update status
                    self.status_label.setText(f"Downloading image {i+1}/{len(resolved_urls)}...")
                    QApplication.processEvents()
                    
                    img_response = requests.get(img_url, headers=headers, timeout=5)
                    
                    # Skip if not successful
                    if img_response.status_code != 200:
                        print(f"Failed to download {img_url}: HTTP {img_response.status_code}")
                        continue
                    
                    # Check content type or file extension
                    content_type = img_response.headers.get('Content-Type', '')
                    is_image = (
                        content_type.startswith('image/') or 
                        any(img_url.lower().endswith(ext) for ext in 
                            ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))
                    )
                    
                    if not is_image:
                        print(f"Skipping non-image content: {img_url} ({content_type})")
                        continue
                    
                    # Save image file
                    file_path = os.path.join(temp_dir, f"img_{i}.jpg")
                    with open(file_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    # Validate image by trying to open it
                    img = cv2.imread(file_path)
                    if img is None or img.size == 0:
                        print(f"Invalid image file: {file_path}")
                        os.remove(file_path)
                        continue
                    
                    # If we got here, it's a valid image
                    downloaded_images.append(file_path)
                    print(f"Successfully downloaded: {file_path}")
                    
                except Exception as e:
                    print(f"Error downloading {img_url}: {e}")
            
            # Process downloaded images
            if downloaded_images:
                print(f"Successfully downloaded {len(downloaded_images)} images")
                self.status_label.setText(f"Processing {len(downloaded_images)} images...")
                QApplication.processEvents()
                self.process_images(downloaded_images)
            else:
                print("No valid images found at the URL")
                self.status_label.setText("No valid images found at the URL")
        
        except Exception as e:
            print(f"Error fetching images from URL: {e}")
            traceback.print_exc()
            self.status_label.setText(f"Error: {str(e)}")
    
    def process_dropped_files(self, file_paths):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # If it's a directory, add all image files
                for root, _, files in os.walk(file_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(os.path.join(root, file))
            elif any(file_path.lower().endswith(ext) for ext in image_extensions):
                # If it's an image file, add it directly
                image_files.append(file_path)
        
        if image_files:
            self.process_images(image_files)
        else:
            self.status_label.setText("No image files found in the dropped items")
    
    def process_images(self, image_paths):
        # Clear previous results
        self.clear_results()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(image_paths))
        self.progress_bar.setValue(0)
        
        # Add images to processing queue
        for i, path in enumerate(image_paths):
            self.processing_queue.put((i, path))
    
    def detect_gender(self, person_img):
        """
        Detect gender using facial features and image characteristics
        This is a simple heuristic approach
        """
        # Detect faces in the image
        gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # If faces found, analyze each face
        gender_votes = []
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = person_img[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue
                    
                # Get face features for classification
                # This is a simple heuristic method
                hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
                
                # Calculate features
                h_mean = np.mean(hsv[:,:,0])  # Hue
                s_mean = np.mean(hsv[:,:,1])  # Saturation
                v_mean = np.mean(hsv[:,:,2])  # Value (brightness)
                
                # Simple rule-based heuristic for gender
                # These rules are arbitrary and not accurate - just for demonstration
                # A real gender detector would use a trained model
                if s_mean < 60:  # Lower saturation often in male faces
                    gender_votes.append('Male')
                else:
                    gender_votes.append('Female')
        else:
            # If no faces, analyze the whole person
            hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:,:,0])
            s_mean = np.mean(hsv[:,:,1])
            
            # Another simple heuristic for full body images
            # Again, this is not accurate and is just for demonstration
            height, width = person_img.shape[:2]
            aspect_ratio = height / max(width, 1)
            
            if aspect_ratio > 2.2:  # Tall, thin figure might be female
                gender_votes.append('Female')
            else:
                gender_votes.append('Male')
        
        # Determine gender by majority vote or default
        if not gender_votes:
            return 'Unknown'
        elif gender_votes.count('Female') > gender_votes.count('Male'):
            return 'Female'
        else:
            return 'Male'
    
    def worker_function(self):
        while not self.stop_threads:
            try:
                # Get an image from the queue with a timeout
                index, img_path = self.processing_queue.get(timeout=0.1)
                
                # Process the image
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        self.result_queue.put((index, img_path, None, "Could not read image"))
                        continue
                    
                    # Get original dimensions for later display
                    original_height, original_width = img.shape[:2]
                    display_img = img.copy()
                    
                    # Detect persons with YOLO
                    with torch.no_grad():
                        results = self.yolo_model(img)
                    
                    # Get person detections
                    detections = results.xyxy[0].cpu().numpy()
                    persons = []
                    
                    # Filter for person class (0) and high confidence
                    for *box, conf, cls in detections:
                        if cls == 0 and conf > 0.5:  # Person class with good confidence
                            x1, y1, x2, y2 = map(int, box)
                            # Make sure box is within image bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(original_width, x2)
                            y2 = min(original_height, y2)
                            
                            # Extract person ROI if it's big enough
                            if x2 > x1 and y2 > y1:
                                person_img = img[y1:y2, x1:x2].copy()
                                if person_img.size > 0:
                                    persons.append((person_img, (x1, y1, x2, y2), conf))
                    
                    # If no persons detected, process the whole image
                    if not persons:
                        person_img = img.copy()
                        persons = [(person_img, (0, 0, original_width, original_height), 1.0)]
                    
                    # Process each detected person
                    for person_img, (x1, y1, x2, y2), conf in persons:
                        try:
                            # Detect gender
                            gender = self.detect_gender(person_img)
                            
                            # Set color based on gender
                            if gender == 'Male':
                                color = (255, 0, 0)  # Blue for male (BGR)
                                thickness = 2
                            elif gender == 'Female':
                                color = (203, 192, 255)  # Pink for female
                                thickness = 2
                            else:
                                color = (0, 255, 0)  # Green for unknown
                                thickness = 2
                            
                            # Draw bounding box
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, thickness)
                            
                            # Add text label
                            text = f"{gender} ({conf:.2f})" if conf < 1.0 else gender
                            font_scale = 0.6
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
                            
                            # Ensure text background is within image
                            text_x = x1
                            text_y = max(y1 - 5, text_size[1] + 5)
                            
                            # Draw text with background
                            cv2.rectangle(display_img, 
                                        (text_x, text_y - text_size[1] - 5),
                                        (text_x + text_size[0], text_y + 5),
                                        (50, 50, 50), -1)
                            cv2.putText(display_img, text, (text_x, text_y),
                                        font, font_scale, (255, 255, 255), 1)
                                        
                        except Exception as e:
                            print(f"Error processing person in image: {e}")
                    
                    # Convert result image to QImage for display
                    rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_img.shape
                    bytes_per_line = ch * w
                    q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    # Put result in queue
                    self.result_queue.put((index, img_path, q_img, None))
                    
                except Exception as e:
                    # Handle errors
                    print(f"Error processing image {img_path}: {e}")
                    traceback.print_exc()
                    self.result_queue.put((index, img_path, None, str(e)))
                
                finally:
                    # Mark task as done
                    self.processing_queue.task_done()
                
            except queue.Empty:
                # Queue is empty, just continue
                pass
            except Exception as e:
                print(f"Worker error: {e}")
    
    def update_results(self, index, img_path, q_img, error):
        # Update progress bar
        self.progress_bar.setValue(self.progress_bar.value() + 1)
        
        # Check if all images are processed
        if self.progress_bar.value() >= self.progress_bar.maximum():
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"Processed {self.progress_bar.maximum()} images")
        
        # Display result
        if error:
            # Display error message
            label = QLabel(f"Error: {os.path.basename(img_path)}: {error}")
            label.setWordWrap(True)
            self.results_layout.addWidget(label, index // 3, index % 3)
        else:
            # Create result widget
            result_widget = QWidget()
            result_layout = QVBoxLayout(result_widget)
            result_layout.setContentsMargins(5, 5, 5, 5)
            
            # Add image
            img_label = QLabel()
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale pixmap to fit nicely in the grid
            scaled_pixmap = pixmap.scaled(
                QSize(300, 300),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            img_label.setPixmap(scaled_pixmap)
            img_label.setAlignment(Qt.AlignCenter)
            
            # Add file name
            filename_label = QLabel(os.path.basename(img_path))
            filename_label.setAlignment(Qt.AlignCenter)
            filename_label.setWordWrap(True)
            
            result_layout.addWidget(img_label)
            result_layout.addWidget(filename_label)
            
            # Add to grid
            row = index // 3
            col = index % 3
            self.results_layout.addWidget(result_widget, row, col)
    
    def clear_results(self):
        # Clear the results layout
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def closeEvent(self, event):
        # Stop worker threads
        self.stop_threads = True
        
        # Stop result updater thread
        self.result_updater.stop()
        self.result_updater.wait()
        
        # Clean up temp directory
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_images')
        if os.path.exists(temp_dir):
            try:
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temp directory: {e}")
        
        super().closeEvent(event)


class DropArea(QLabel):
    file_dropped = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drop image files or folders here")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f8f8f8;
            }
            QLabel:hover {
                background-color: #f0f0f0;
                border-color: #999;
            }
        """)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        file_paths = []
        for url in event.mimeData().urls():
            file_paths.append(url.toLocalFile())
        
        self.file_dropped.emit(file_paths)


class ResultUpdater(QThread):
    result_ready = pyqtSignal(int, str, QImage, str)
    
    def __init__(self, result_queue):
        super().__init__()
        self.result_queue = result_queue
        self.running = True
    
    def run(self):
        while self.running:
            try:
                # Get result from queue with timeout
                index, img_path, q_img, error = self.result_queue.get(timeout=0.1)
                
                # Emit signal to update UI
                self.result_ready.emit(index, img_path, q_img, error)
                
                # Mark task as done
                self.result_queue.task_done()
            except queue.Empty:
                # Queue is empty, just continue
                pass
            except Exception as e:
                print(f"Result updater error: {e}")
    
    def stop(self):
        self.running = False


if __name__ == "__main__":
    # Set higher DPI awareness for better display on high-res screens
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
        
    app = QApplication(sys.argv)
    window = GenderDetectionApp()
    window.show()
    sys.exit(app.exec_())