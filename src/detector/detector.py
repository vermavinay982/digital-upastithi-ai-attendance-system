# from detector.mediapipe.mediapipe_det import detect_faces
from detector.yoloface.yoloface_algo import detect_faces
import os
import cv2

class Detector:
    def __init__(self):
        pass

    def detect(self, np_imgs, img_names, op_path):
        faces = []
        faces, face_img = detect_faces(np_imgs, img_names)

        os.makedirs(op_path, exist_ok=True)
        for i, face in enumerate(faces):
            img_src = face_img[i]

            img_path = f"{op_path}/{img_src.replace('.','_')}_{i}.png"
            area = len(face[0])*len(face)
            # if area<10_00_000:
            #     continue

            flag = cv2.imwrite(img_path, face)
            # print(f"Written: {flag}, {img_path} - {len(face)}, {len(face[0])} area: {area}")
            if not flag:
                print("Error writing detected faces: detector.py: {img_path}")
        return faces