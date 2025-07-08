import cv2
import os
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm 

register_heif_opener()

def load_img_arrays(folder_path):
    files = os.listdir(folder_path)
    images = []
    img_names = []
    for file in tqdm(files):
        print(file)
        ext = file.split('.')[-1]

        file_path = os.path.join(folder_path, file)

        if ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            img = cv2.imread(file_path)
        elif ext == 'HEIC':  # Handle HEIC images
            img = Image.open(file_path)  # Open using Pillow
            img = np.asarray(img)  # Convert to numpy array
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        else:
            print(f"Unsupported file type: {file}")
            continue

        print(file, img.shape)
        images.append(img)
        img_names.append(file)
        # break
    # images = np.array(images)  # Uncomment if needed as a numpy array
    return images, img_names