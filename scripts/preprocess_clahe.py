import cv2
import os
import glob
from pathlib import Path
from tqdm import tqdm

def apply_clahe(img_path, out_path):
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        return
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(str(out_path), final)

def main():
    # Paths
    base_data = Path("data/yolo")
    test_data = Path("data/ClearSAR/data/images/test")
    out_base = Path("data/yolo_clahe")
    
    print("Starting CLAHE Preprocessing...")
    
    # Process Folds 0-4 (Train/Val)
    all_images = list(base_data.glob("fold*/images/*/*.png"))
    # Process Test set
    test_images = list(test_data.glob("*.png"))
    
    total_images = all_images + test_images
    print(f"Total images to process: {len(total_images)}")
    
    for img_path in tqdm(total_images):
        if "test" in str(img_path):
            rel_path = "test/" + img_path.name
        else:
            # data/yolo/foldN/images/val/xxx.png -> foldN/images/val/xxx.png
            rel_path = str(img_path.relative_to(base_data))
        
        out_path = out_base / rel_path
        if out_path.exists():
            continue
        apply_clahe(img_path, out_path)

if __name__ == "__main__":
    main()
