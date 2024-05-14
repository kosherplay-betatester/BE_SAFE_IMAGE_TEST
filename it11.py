import streamlit as st
import torch
from PIL import Image
import clip
from io import BytesIO
import requests
from torchvision import transforms
import numpy as np
from bs4 import BeautifulSoup
import urllib.parse

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
        st.error("Invalid URL. Please enter a valid URL starting with http:// or https://")
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
                st.write("Person is too pixelized, skipped.")
                continue

            crop = image.crop(box)
            crops.append(crop)

    if crops:
        image_inputs = torch.stack([preprocess(crop) for crop in crops]).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model_classification.encode_image(image_inputs)
            text_features = model_classification.encode_text(text_inputs)

        # Calculate the similarity of the image to each category
        with torch.no_grad():
            logits_per_image, _ = model_classification(image_inputs, text_inputs)
            probs = torch.softmax(logits_per_image, dim=-1).cpu().numpy()

        for i, crop_probs in enumerate(probs):
            # Store the scores in a dictionary
            scores = {category: probability for category, probability in zip(categories, crop_probs)}

            # Sort the scores in descending order
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

            # Determine the highest score category
            highest_category = sorted_scores[0][0]
            if highest_category == "woman":
                border_color = "pink"
            elif highest_category == "man":
                border_color = "blue"
            else:
                border_color = "gray"

            # Display the image with the appropriate border color
            st.markdown(
                f"""
                <div style="border: 5px solid {border_color}; padding: 5px; display: inline-block;">
                    <img src="{image_url}" width="100%" />
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display the link colored with the image border color and add a colored circle
            st.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <a href="{image_url}" style="color: {border_color}; margin-right: 5px;">{image_url}</a>
                    <div style="width: 10px; height: 10px; background-color: {border_color}; border-radius: 50%;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Print the probabilities
            for category, probability in sorted_scores:
                # Highlight the top score in bold, the second score in italics
                if category == sorted_scores[0][0]:
                    st.write(f"**{category.capitalize()} Score: {probability * 100:.2f}**")
                elif category == sorted_scores[1][0]:
                    st.write(f"*{category.capitalize()} Score: {probability * 100:.2f}*")
                else:
                    st.write(f"{category.capitalize()} Score: {probability * 100:.2f}")

            # Summarize the two highest scores
            st.write(f"\nImage probably includes: a {sorted_scores[0][0].capitalize()} \
                        {sorted_scores[0][1]*100:.2f}% or a {sorted_scores[1][0].capitalize()} {sorted_scores[1][1]*100:.2f}%")

# Streamlit app
st.title("Image Content Analyzer")

# Add a text input for the user to enter a URL
webpage_url = st.text_input("Enter a webpage URL to analyze", "")

if st.button("Test URL"):
    if webpage_url:
        image_urls = get_image_urls(webpage_url)
        if image_urls:
            st.write(f"Found {len(image_urls)} images on {webpage_url}")
            for image_url in image_urls:
                image = load_image_from_url(image_url)
                process_image(image, image_url)
        else:
            st.write("No images found on the provided URL.")
