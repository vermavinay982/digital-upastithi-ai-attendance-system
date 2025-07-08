import numpy as np
from deepface import DeepFace 
from tqdm import tqdm 

class ExtractorModel:
    def __init__(self):
        pass

    def load_model(self):
        pass

    def infer_model(self, img_path):
        embedding = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=False)
        # embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)
        for emb in embedding:
            return emb["embedding"]
        return None

    def extract_single_feature(self, img_paths):
        features = []
        for img_path in img_paths:
            feature = self.infer_model(img_path)
            if feature is not None:
                features.append(feature)
        return features
    
    def extract_img_feature(self, img_path):
        features = []
        feature = self.infer_model(img_path)
        if feature is not None: features.append(feature)
        return features
    
    def extract_mean_features(self, img_paths):
        features = self.extract_single_feature(img_paths)
        return np.mean(features, axis=0)  # Return averaged feature vector

    def extract_mapped_feature(self, img_paths):
        # To get mapping per image with feature
        feature_dict = {}
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Define valid image extensions

        print("==> Extracting Face Features")
        for img_path in tqdm(img_paths):
            if img_path.lower().endswith(valid_extensions):  # Check if the file has a valid extension
                feature = self.infer_model(img_path)
                feature_dict[img_path] = feature
            else:
                print(f"Skipping non-image file: {img_path}")  # Log skipped files for debugging
        return feature_dict

    

    
if __name__=="__main__":
    ExtractorModel()

# def get_features(img_paths):
#     features = []
#     for img_path in img_paths:
#         img = Image.open(img_path).convert("RGB")
#         img_tensor = self.transform(img).unsqueeze(0)
#         with torch.no_grad():
#             feature = self.model.predict(img_tensor).squeeze()
#         features.append(feature)
#     return np.mean(features, axis=0)  # Return averaged feature vector

#     return None