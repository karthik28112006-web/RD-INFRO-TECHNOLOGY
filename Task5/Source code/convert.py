import os
from PIL import Image

# Change this to 'with_mask' or 'without_mask' depending on which folder you are cleaning
TARGET_FOLDER = "dataset/without_mask" 
OUTPUT_FORMAT = "JPEG"  # Use "JPEG" or "PNG"
NEW_EXT = ".jpg"        # Use ".jpg" or ".png"

for filename in os.listdir(TARGET_FOLDER):
    ext = os.path.splitext(filename)[1].lower()
    
    # Skip files that are already in the correct format
    if ext == NEW_EXT:
        continue
        
    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"]:
        img_path = os.path.join(TARGET_FOLDER, filename)
        try:
            with Image.open(img_path) as img:
                # Convert RGBA (transparent) to RGB if saving as JPG
                if OUTPUT_FORMAT == "JPEG" and img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # Create new filename
                new_filename = os.path.splitext(filename)[0] + NEW_EXT
                new_path = os.path.join(TARGET_FOLDER, new_filename)
                
                # Save and delete old file
                img.save(new_path, OUTPUT_FORMAT)
                os.remove(img_path)
                print(f"Converted: {filename} -> {new_filename}")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")