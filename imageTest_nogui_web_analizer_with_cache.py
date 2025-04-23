import os
import sys
import time
import io

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import torch
from ultralytics import YOLO
import clip

from PIL import Image, UnidentifiedImageError
import numpy as np

# --- Configuration ---
CONF_THRESHOLD = 0.45
CATEGORIES = ["man", "woman", "object"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE = os.path.join(os.getcwd(), "results.txt")
MAX_IMAGES = 100  # maximum number of images to scrape per URL


def load_models():
    det_model = YOLO("yolo11n.pt")
    det_model.to(DEVICE)
    clf_model, preprocess = clip.load("ViT-L/14@336px", device=DEVICE)
    tokens = clip.tokenize([f"a photo of a {c}" for c in CATEGORIES]).to(DEVICE)
    with torch.no_grad():
        text_feats = clf_model.encode_text(tokens)
    return det_model, clf_model, preprocess, text_feats


def scrape_images_from_url(url, max_images=MAX_IMAGES):
    out = []
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    try:
        resp = sess.get(url, timeout=15)
        ctype = resp.headers.get("content-type", "").lower()
        # direct image
        if "image" in ctype and "html" not in ctype:
            out.append((resp.content, url))
            return out
        soup = BeautifulSoup(resp.text, "html.parser")
        seen = set()
        for img in soup.find_all("img"):
            src = img.get("data-src") or img.get("src")
            if not src:
                continue
            full = urljoin(resp.url, src)
            if full in seen:
                continue
            seen.add(full)
            try:
                r2 = sess.get(full, timeout=10)
                if "image" in r2.headers.get("content-type", "").lower() and len(r2.content) > 2000:
                    out.append((r2.content, full))
                    if len(out) >= max_images:
                        break
            except:
                continue
    except:
        pass
    return out


def safe_open_bytes(data):
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def classify_image(img, det_model, clf_model, preprocess, text_feats):
    summary = {"Male": 0, "Female": 0, "Else": 0, "Error": 0}
    try:
        res = det_model.predict(
            source=np.array(img),
            device=DEVICE,
            conf=CONF_THRESHOLD,
            classes=[0],  # person
            verbose=False
        )[0]
        if not res.boxes:
            summary["Else"] = 1
            return summary

        crops = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            crop = img.crop((x1, y1, x2, y2))
            crops.append(preprocess(crop).to(DEVICE))

        batch = torch.stack(crops)
        with torch.no_grad():
            feats = clf_model.encode_image(batch)
            logits = feats @ text_feats.t()
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for p in probs:
            idx, conf = int(p.argmax()), float(p.max())
            if idx == 0 and conf >= 0.5:
                summary["Male"] += 1
            elif idx == 1 and conf >= 0.5:
                summary["Female"] += 1
            else:
                summary["Else"] += 1

    except:
        summary = {"Male": 0, "Female": 0, "Else": 0, "Error": 1}

    return summary


def load_existing_results():
    if not os.path.isfile(RESULTS_FILE):
        return [], set()
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # skip header + blank line
    if lines and lines[0].startswith("Total processing time"):
        entry_lines = lines[2:]
    else:
        entry_lines = lines
    tested = set()
    for L in entry_lines:
        if ": " in L:
            path = L.split(": ", 1)[0]
            tested.add(path)
    return entry_lines, tested


def main():
    url = input("Enter URL to scan images from: ").strip()
    existing_lines, tested = load_existing_results()

    print("Loading models…")
    det_model, clf_model, preprocess, text_feats = load_models()

    items = scrape_images_from_url(url)
    if not items:
        print("No images found at that URL.")
        sys.exit(0)

    # filter out already tested URLs
    new_items = [(data, u) for data, u in items if u not in tested]
    if not new_items:
        print("✅ All scraped image URLs already tested; nothing to do.")
        return

    print(f"Testing {len(new_items)} new images (skipping {len(items)-len(new_items)} cached)…")
    t0 = time.time()
    new_lines = []

    for data, img_url in new_items:
        img = safe_open_bytes(data)
        summ = classify_image(img, det_model, clf_model, preprocess, text_feats) if img else {"Male":0,"Female":0,"Else":0,"Error":1}
        line = (
            f"{img_url}: "
            f"Male={summ['Male']} "
            f"Female={summ['Female']} "
            f"Else={summ['Else']} "
            f"Error={summ['Error']}\n"
        )
        new_lines.append(line)
        print(" •", line.strip())

    elapsed = time.time() - t0
    ips = len(new_items) / elapsed if elapsed > 0 else 0.0

    # rewrite results.txt
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(f"Total processing time this run: {elapsed:.2f} seconds — {ips:.2f} images/sec\n\n")
        for L in existing_lines:
            f.write(L)
        for L in new_lines:
            f.write(L)

    print(f"\n✅ Done in {elapsed:.2f}s ({ips:.2f} imgs/s). Results updated in:\n  {RESULTS_FILE}")


if __name__ == "__main__":
    main()
