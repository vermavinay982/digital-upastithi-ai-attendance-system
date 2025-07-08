import cv2
import torch
from ultralytics import YOLO
import os
from tqdm import tqdm
from config import root_path

def detect_faces(np_images, img_names):
    # Load YOLO face detection model (Use a trained face model or custom weights)
    # model = YOLO("yolov8n-face.pt")  # Replace with the correct face detection model
    model = YOLO("detector\yoloface\yolov8l-face.pt")  # Replace with the correct face detection model

    faces = []
    face_img = []
    for idx, image in tqdm(enumerate(np_images)):
        # Run face detection
        results = model.predict(image)

        # Create an output directory for face crops
        output_dir = os.path.join(root_path, "face_crops")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process results
        for i, result in enumerate(results):
            for j, box in enumerate(result.boxes):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop face from the image
                face_crop = image[y1:y2, x1:x2]
                faces.append(face_crop)
                face_img.append(img_names[idx])
                # Save the face crop
                face_crop_path = os.path.join(output_dir, f"face_{i}_{j}.jpg")
                cv2.imwrite(face_crop_path, face_crop)
                # print(f"Face Write Saved: {face_crop_path}")
        print("Face extraction complete.")
    return faces, face_img