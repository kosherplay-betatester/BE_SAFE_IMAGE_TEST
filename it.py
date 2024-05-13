# Import necessary libraries
import streamlit as st
import torch
from PIL import Image
import clip
from io import BytesIO
import requests
from torchvision import transforms
import cv2
import numpy as np
import urllib

# Load the object detection model (YOLOv5)
model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the classification model (ViT)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_classification, preprocess = clip.load("ViT-B/32", device=device)

categories = ["man", "woman"]
#categories = ["man", "woman", "boy", "girl" ,"hasidic jew" ,"object" ,"naked man" , "naked woman"]
# categories = ["man", "woman", "boy", "girl","baby" ,"nudity"]


# Function to load image from URL
def load_image_from_url(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

# Streamlit app
st.title("Image Content Analyzer")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])
image_url = st.text_input('Enter image URL here...')

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif image_url:
    try:
        image = load_image_from_url(image_url)
    except:
        st.write("Invalid URL or unable to fetch image from URL.")
        image = None

if image is not None:
    st.image(image, caption='Uploaded Image.', use_column_width=True)

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

            # If the bounding box is less than 5% of the total image, skip ,Default value is 0.05
            if bbox_proportion < 0.0001:
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
            st.image(crops[i], caption='Detected person.', use_column_width=True)

            # Store the scores in a dictionary
            scores = {category: probability for category, probability in zip(categories, crop_probs)}

            # Sort the scores in descending order
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

            # Print the probabilities
            for category, probability in sorted_scores:
               # Highlight the top score in bold, the second score in italics
                if category == sorted_scores[0][0]:
                    st.write(f"{category.capitalize()} Score: {probability * 100:.2f}")
                elif category == sorted_scores[1][0]:
                    st.write(f"*{category.capitalize()} Score: {probability * 100:.2f}*")
                else:
                    st.write(f"{category.capitalize()} Score: {probability * 100:.2f}")
                    # Summarize the two highest scores
            st.write(f"\nImage probably includes: a {sorted_scores[0][0].capitalize()} \
            {sorted_scores[0][1]*100:.2f}% or a {sorted_scores[1][0].capitalize()} {sorted_scores[1][1]*100:.2f}%")