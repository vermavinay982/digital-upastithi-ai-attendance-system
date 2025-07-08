from PIL import Image
import torchvision.transforms as transforms
# from deepface.DeepFace import build_model
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

class ExtractorModel:
    def __init__(self):
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self):
        # return build_model("VGG-Face") # tensorflow version
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # Remove last classification layer
        model.eval()
        return model

    def infer_model(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            feature = self.model(img_tensor).squeeze()
        return feature

    def extract_img_feature(self, img_path):
        features = []
        feature = self.infer_model(img_path)
        feature = feature.detach().cpu().numpy()
        feature = np.array(feature, dtype='float64')
        feature = list(feature)
        if feature is not None: features.append(feature)
        return features
    
    def extract_single_feature(self, img_paths):
        features = []
        for img_path in img_paths:
            feature = self.infer_model(img_path)
            features.append(feature)
        return features
    
    def extract_mean_features(self, img_paths):
        features = self.extract_single_feature(img_paths)
        return np.mean(features, axis=0)  # Return averaged feature vector

    def extract_mapped_feature(self, img_paths):
        # To get mapping per image with feature
        feature_dict = {}
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Define valid image extensions

        for img_path in img_paths:
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