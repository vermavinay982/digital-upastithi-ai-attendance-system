import os
# from recognizer.deepface.deepface_algo import ExtractorModel
from recognizer.deepface.deepface_algo_updated import ExtractorModel
import json
import pickle
from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np

class Recognizer:
    def __init__(self, gt_folder, pickled_path=None):
        self.model = ExtractorModel()
        self.gt_folder = gt_folder
        self.gt_feature_dict = {}
        self.pickled_path = pickled_path
        self.load_feature_dict()
        """
        gt_feature_dict = {
            akshay: features,
            alia: features,
            .
            .
        } 
        """

    def load_feature_dict(self):
        if not (self.pickled_path is None):
            if os.path.exists(self.pickled_path):
                with open(self.pickled_path, 'rb') as f:
                    self.gt_feature_dict = pickle.load(f)
                    print(f"Features Loaded Successfully: {self.pickled_path}")
                    return

        # self.pickled_path = 'gt_features_temp.pkl'
        self.generate_features()
        with open(self.pickled_path, 'wb') as f:
            pickle.dump(self.gt_feature_dict, f)
            print(f"Features Saved Successfully: {self.pickled_path} : {len(self.gt_feature_dict)} Features")
        
        # Print all registered members and feature lengths
        # for gt_label in self.gt_feature_dict.keys():
        #     print(f"feature: {gt_label} {self.gt_feature_dict[gt_label].shape}")

    # Custom handler for NumPy types
    @staticmethod
    def convert_numpy(obj):
        if isinstance(obj, np.float32):
            return float(obj)  # Convert np.float32 to float
        elif isinstance(obj, np.int32):
            return int(obj)  # Convert np.int32 to int
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        raise TypeError(f"Type {type(obj)} not serializable")

    def recognize_old(self, detector_op_folder, json_path='results.json'):
        img_paths = [os.path.join(detector_op_folder, img_path) for img_path in os.listdir(detector_op_folder)]
        # print(img_paths, ">>>>>>>>>>>>>")
        pred_feature_dict = self.model.extract_mapped_feature(img_paths)

        pred_results = {}
        for key in pred_feature_dict.keys():
            pred_feats = pred_feature_dict[key]
            predictions = []
            for label in self.gt_feature_dict.keys():
                gt_feats = self.gt_feature_dict[label]
                score = cosine(pred_feats, gt_feats)
                # Rank-based predictions
                pred = {
                    'label': label,
                    'score': score
                }
                predictions.append(pred)
            pred_results[key] = sorted(predictions, key=lambda x: x['score'], reverse=False)

        # Writing JSON with proper handling of NumPy types
        with open(json_path, 'w') as f:
            json.dump(pred_results, f, indent=2, default=self.convert_numpy)
            print("Results Written Successfully")

        names = []
        fr_thresh = 0.7
        for key in pred_results.keys():
            predictions = pred_results[key]
            label, score = predictions[0]['label'], predictions[0]['score']

            if score < fr_thresh:
                names.append([label, score])

        return json_path

    def recognize(self, detector_op_folder, json_path='results.json'):
        img_paths = [os.path.join(detector_op_folder, img_path) for img_path in os.listdir(detector_op_folder)]
        # print(img_paths, ">>>>>>>>>>>>>")

        # each cropped image got one feature out - map[im path] = feature
        pred_feature_dict = self.model.extract_mapped_feature(img_paths)

        # print("===>", self.gt_feature_dict.keys())
        print(f"== Comparing Faces to find your best one == GT:[{len(self.gt_feature_dict.keys())}]")
        pred_results = {}
        for label in tqdm(self.gt_feature_dict.keys()): # 90+1 class of gt
            gt_feats = self.gt_feature_dict[label]
            predictions = []
            for key in pred_feature_dict.keys(): # given image cropped paths
                pred_feats = pred_feature_dict[key]
                score = cosine(pred_feats, gt_feats)
                # Rank-based predictions
                pred = {
                    'label': key,
                    'score': score
                }
                predictions.append(pred)
            pred_results[label] = sorted(predictions, key=lambda x: x['score'], reverse=False)

        # Writing JSON with proper handling of NumPy types
        with open(json_path, 'w') as f:
            json.dump(pred_results, f, indent=2, default=self.convert_numpy)
            print("Results Written Successfully")

        # names = []
        # fr_thresh = 0.7
        # for key in pred_results.keys():
        #     predictions = pred_results[key]
        #     label, score = predictions[0]['label'], predictions[0]['score']

        #     if score < fr_thresh:
        #         names.append([label, score])

        return json_path
    
    def generate_features(self, limit=10):
        gt_labels = os.listdir(self.gt_folder)
        count = []
        print("== Generating Features ==")
        for gt_label in tqdm(gt_labels):  # Students' names
            gt_img_folder = os.path.join(self.gt_folder, gt_label)
            # Images for each student
            img_paths = [os.path.join(gt_img_folder, img_path) for img_path in os.listdir(gt_img_folder)][:limit]
            features = self.model.extract_mean_features(img_paths)
            self.gt_feature_dict[gt_label] = features
            count.append(len(img_paths))
            # print(img_paths)
        print(f"Total Images for Features: {sum(count)}, Faces: {len(count)}, Avg Faces:{sum(count)/len(count)}")
        pass
