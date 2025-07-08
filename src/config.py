import os
# gt_folder_path = f"../data/gt_single_photo"
root_path = r"../run_files/"
data_path = r"../data/"

gt_folder_path = "gt_with_roll_cleaned"
gt_folder_path = os.path.join(data_path, gt_folder_path)

image_op_folder = "results"

stud_details_path = "students_details.json"
stud_details_path = os.path.join(data_path, stud_details_path)

# feature_pickle_path = None
# feature_pickle_path = "E:\code\DL_ClassFR\src\gt_features_90clusters.pkl"
# feature_pickle_path = "gt_features_cleaned_5_facenet128.pkl"
feature_pickle_path = "gt_features_cleaned_5_facenet512.pkl"
feature_pickle_path = os.path.join(data_path, feature_pickle_path)
# label = "attendance"
label = "pred"

FR_THRESH = 0.3
use_TF = True
