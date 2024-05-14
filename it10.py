import pygame
import torch
from PIL import Image, UnidentifiedImageError
import open_clip
from io import BytesIO
import requests
import numpy as np
from bs4 import BeautifulSoup
import urllib.parse

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
pygame.display.set_caption('Image Content Analyzer')

# Load the object detection model (YOLOv5)
model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the classification model (ViT) using open_clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model_classification, preprocess, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

categories = ["man", "woman", "object"]

# Function to load image from URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure we got a successful response
        image = Image.open(BytesIO(response.content))
        return image
    except (requests.RequestException, UnidentifiedImageError) as e:
        print(f"Error loading image from {url}: {e}")
        return None

# Function to get all image URLs from a webpage
def get_image_urls(webpage_url):
    try:
        response = requests.get(webpage_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        image_urls = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                # Convert relative URLs to absolute URLs
                src = urllib.parse.urljoin(webpage_url, src)
                image_urls.append(src)
        return image_urls
    except requests.RequestException as e:
        print(f"Error fetching URLs from {webpage_url}: {e}")
        return []

# Function to process an image
def process_image(image):
    if image is None:
        return None, None

    # Run object detection on the image
    results = model_detection(image)

    # For each detected person, run classification
    crops = []
    for *box, conf, cls in results.xyxy[0]:
        # Only process detections that are people
        if results.names[int(cls)] == "person":
            box = [int(x) for x in box]
            # Calculate the proportion of the image the bounding box occupies
            bbox_area = (box[2] - box[0]) * (box[3] - box[1])
            total_image_area = image.width * image.height
            bbox_proportion = bbox_area / total_image_area

            # If the bounding box is less than 5% of the total image, skip
            if bbox_proportion < 0.1111:
                continue

            crop = image.crop(box)
            crops.append(crop)

    if crops:
        image_inputs = torch.stack([preprocess(crop).to(device) for crop in crops])
        text_inputs = open_clip.tokenize([f"a photo of a {c}" for c in categories]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model_classification.encode_image(image_inputs)
            text_features = model_classification.encode_text(text_inputs)

        # Calculate the similarity of the image to each category
        with torch.no_grad():
            logits_per_image = image_features @ text_features.T
            probs = torch.softmax(logits_per_image, dim=-1).cpu().numpy()

        return probs, crops

    return None, None

def draw_image(image, pos, border_color):
    mode = image.mode
    size = image.size
    data = image.tobytes()

    py_image = pygame.image.fromstring(data, size, mode)
    pygame.draw.rect(screen, border_color, (*pos, size[0], size[1]), 5)
    screen.blit(py_image, pos)

def clean_text(text):
    return text.replace('\x00', '')

def main():
    running = True
    input_url = ""
    images = []
    results = []

    font = pygame.font.Font(None, 32)
    input_box = pygame.Rect(100, 50, 600, 32)
    button_box = pygame.Rect(710, 50, 100, 32)
    drop_area = pygame.Rect(100, 100, 600, 100)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_inactive
    active = False
    text = ''
    button_color = pygame.Color('gray')
    drop_color = pygame.Color('lightskyblue3')

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                if button_box.collidepoint(event.pos):
                    input_url = text
                    text = ''
                    image_urls = get_image_urls(input_url)
                    images = [load_image_from_url(url) for url in image_urls]
                    images = [image for image in images if image is not None]
                    results = [process_image(image) for image in images]
                color = color_active if active else color_inactive
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_v and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        # Paste text from clipboard
                        pasted_text = pygame.scrap.get(pygame.SCRAP_TEXT).decode('utf-8')
                        text += clean_text(pasted_text)
                    elif event.key == pygame.K_RETURN:
                        input_url = text
                        text = ''
                        image_urls = get_image_urls(input_url)
                        images = [load_image_from_url(url) for url in image_urls]
                        images = [image for image in images if image is not None]
                        results = [process_image(image) for image in images]
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode
            if event.type == pygame.DROPFILE:
                input_url = event.file
                image_urls = get_image_urls(input_url)
                images = [load_image_from_url(url) for url in image_urls]
                images = [image for image in images if image is not None]
                results = [process_image(image) for image in images]

        screen.fill((30, 30, 30))
        txt_surface = font.render(clean_text(text), True, color)
        width = max(600, txt_surface.get_width() + 10)
        input_box.w = width
        screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
        pygame.draw.rect(screen, color, input_box, 2)

        # Draw the "Start Test" button
        button_text = font.render('Start Test', True, pygame.Color('white'))
        pygame.draw.rect(screen, button_color, button_box)
        screen.blit(button_text, (button_box.x + 10, button_box.y + 5))

        # Draw the drag-and-drop area
        drop_text = font.render('Drag & Drop URL Here', True, pygame.Color('white'))
        pygame.draw.rect(screen, drop_color, drop_area, 2)
        screen.blit(drop_text, (drop_area.x + 20, drop_area.y + 35))

        y_offset = 220
        for result, image in zip(results, images):
            probs, crops = result
            if probs is not None:
                for crop_probs, crop in zip(probs, crops):
                    scores = {category: probability for category, probability in zip(categories, crop_probs)}
                    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                    highest_category = sorted_scores[0][0]
                    border_color = pygame.Color('gray')  # Default to gray

                    if highest_category == "woman":
                        border_color = pygame.Color('pink')
                    elif highest_category == "man":
                        border_color = pygame.Color('blue')

                    draw_image(crop, (50, y_offset), border_color)
                    y_offset += crop.height + 20

        pygame.display.flip()

if __name__ == "__main__":
    pygame.scrap.init()
    main()
    pygame.quit()
