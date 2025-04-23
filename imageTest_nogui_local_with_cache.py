import os
import sys
import time

import torch
from ultralytics import YOLO
import clip

from PIL import Image, UnidentifiedImageError
import numpy as np

# --- Configuration ---
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CONF_THRESHOLD = 0.45
CATEGORIES = ["man", "woman", "object"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE = os.path.join(os.getcwd(), "results.txt")


def load_models():
    det_model = YOLO("yolo11n.pt")
    det_model.to(DEVICE)
    clf_model, preprocess = clip.load("ViT-L/14@336px", device=DEVICE)
    tokens = clip.tokenize([f"a photo of a {c}" for c in CATEGORIES]).to(DEVICE)
    with torch.no_grad():
        text_feats = clf_model.encode_text(tokens)
    return det_model, clf_model, preprocess, text_feats


def find_images(root):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn.lower())[1] in IMAGE_EXTENSIONS:
                yield os.path.join(dp, fn)


def safe_open(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None


def classify(img, det_model, clf_model, preprocess, text_feats):
    summary = {"Male":0, "Female":0, "Else":0, "Error":0}
    try:
        res = det_model.predict(source=np.array(img),
                                device=DEVICE,
                                conf=CONF_THRESHOLD,
                                classes=[0],
                                verbose=False)[0]
        if not res.boxes:
            summary["Else"] = 1
            return summary

        crops = []
        for b in res.boxes:
            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
            crops.append(preprocess(img.crop((x1,y1,x2,y2))).to(DEVICE))

        batch = torch.stack(crops)
        with torch.no_grad():
            feats = clf_model.encode_image(batch)
            logits = feats @ text_feats.t()
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        for p in probs:
            idx, conf = int(p.argmax()), float(p.max())
            if idx==0 and conf>=0.5:
                summary["Male"] += 1
            elif idx==1 and conf>=0.5:
                summary["Female"] += 1
            else:
                summary["Else"] += 1

    except:
        summary = {"Male":0, "Female":0, "Else":0, "Error":1}

    return summary


def load_existing_results():
    """
    Returns:
      existing_lines: list of lines after the header+blank
      tested_paths: set of full paths already in results.txt
    """
    if not os.path.isfile(RESULTS_FILE):
        return [], set()

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # strip out header (first line) and the following blank line
    if lines and lines[0].startswith("Total processing time"):
        lines = lines[2:]
    # now lines are only entry lines
    tested = set()
    for L in lines:
        if ": " in L:
            path = L.split(": ", 1)[0]
            tested.add(path)
    return lines, tested


def main():
    root = input("Enter directory path to scan: ").strip()
    if not os.path.isdir(root):
        print(f"❌ Not a directory: {root}")
        sys.exit(1)

    det_model, clf_model, preprocess, text_feats = load_models()
    existing_lines, tested = load_existing_results()

    all_imgs = list(find_images(root))
    to_test  = [p for p in all_imgs if p not in tested]
    if not to_test:
        print("✅ No new images to test; all paths already in results.txt")
        return

    print(f"Testing {len(to_test)} new images (skipping {len(all_imgs)-len(to_test)} cached)…")
    t0 = time.time()
    new_lines = []
    for p in to_test:
        img = safe_open(p)
        summ = classify(img, det_model, clf_model, preprocess, text_feats) if img else {"Male":0,"Female":0,"Else":0,"Error":1}
        line = f"{p}: Male={summ['Male']} Female={summ['Female']} Else={summ['Else']} Error={summ['Error']}\n"
        new_lines.append(line)
        print(" •", line.strip())

    elapsed = time.time() - t0

    # rewrite results.txt
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(f"Total processing time this run: {elapsed:.2f} seconds\n\n")
        for L in existing_lines:
            f.write(L)
        for L in new_lines:
            f.write(L)

    print(f"\n✅ Done in {elapsed:.2f}s. Updated {RESULTS_FILE}")


if __name__ == "__main__":
    main()
