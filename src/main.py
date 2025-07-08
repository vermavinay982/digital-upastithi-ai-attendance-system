# PIPELINE


import numpy as np 
import os
import sys
sys.path.append("../")
from detector.detector import Detector
from recognizer.recognize import Recognizer
from utils import load_img_arrays
from time import time 

from config import gt_folder_path, image_op_folder, root_path, feature_pickle_path, use_TF

if use_TF:
    from recognizer.face_preprocessing import clean_images

from recognizer.face_preprocessing import clean_images
from collections import defaultdict
from glob import glob 

def run_pipeline(folder_path, run_detector=True, run_recognizer=True, feature_pickle_path=None, clean_images_flag=True):
    det_folder_path = os.path.join(root_path, "detected_faces")
    
    if os.path.exists(det_folder_path):
        old_path = f"{det_folder_path}_old_{time()}" # to save old results
        os.rename(det_folder_path, old_path)
        print(f"Renamed Old Detection Folder to - {old_path} | Delete if not required")
        
        # os.remove(det_folder_path)
        # print(f"Deleted Old Detection Folder to - Brutal way to save memory")
        
    if run_detector:
        facedetector = Detector()
        np_imgs, img_names = load_img_arrays(folder_path)
        print("Total Images:", len(np_imgs))
        # TODO: create annotation of detection, to view detected faces
        faces = facedetector.detect(np_imgs, img_names, op_path=det_folder_path)
    
    if use_TF:
        if clean_images_flag:
            clean_images(det_folder_path)

    if run_recognizer:
        recognizer = Recognizer(gt_folder_path, pickled_path=feature_pickle_path)
        op_json_path = os.path.join(root_path, "results.json")
        json_path = recognizer.recognize(det_folder_path, json_path=op_json_path)
    print(f"Execution Completed: {json_path}")
    return json_path

def filter_results_simple(data, thresh=1):
    """
    filtered output
    'mt24127': {'label': '../data/detected_faces\\airdrop_img_jpg_23.png', 'score': 0.6566423466667584},
    '2022310': {'label': '../data/detected_faces\\airdrop_img_jpg_27.png', 'score': 0.711351715507748}}
    """
    filtered_matches = {}
    for roll, imgs in data.items():
        img = imgs[0]
        label = img['label']
        score = img['score']

        if score < thresh:
            filtered_matches[roll] = {
                'label': label,
                'score': score
            }
    return filtered_matches

def filter_results_greedy(data, thresh=1):
    """
    filtered output
    'mt24127': {'label': '../data/detected_faces\\airdrop_img_jpg_23.png', 'score': 0.6566423466667584},
    '2022310': {'label': '../data/detected_faces\\airdrop_img_jpg_27.png', 'score': 0.711351715507748}}
    """
    data_img_temp = defaultdict(list)
    for roll in data.keys():
        for img in data[roll]: # each roll matched with n faces, but 1 face can be associated with 1 roll only, so if face is already taken, dont assign it again, single initialization with the best one
            label = img['label']
            # label = os.path.basename(label)
            score = img['score']
            d = {
                'roll':roll,
                'score': score
                }
            data_img_temp[label].append(d)

    data_img = {}
    for img in data_img_temp.keys():
        data_img[img] = sorted(data_img_temp[img], key=lambda x:x['score'])

    # print(len(data), len(data_img), data_img.keys())
    matrix = np.ones((len(data), len(data_img)))

    data.keys()
    data_img.keys()

    idx_data = {k:i for i, k in enumerate(data.keys())}
    idx_data_img = {k:i for i, k in enumerate(data_img.keys())}

    dec_idx_data = {k:i for i,k in idx_data.items()}
    dec_idx_data_img = {k:i for i,k in idx_data_img.items()}
    # dec_idx_data

    # data already has images sorted in perfect order, so fine
    for roll in data.keys():
        for img in data[roll]: # each roll matched with n faces, but 1 face can be associated with 1 roll only, so if face is already taken, dont assign it again, single initialization with the best one
            label = img['label']
            # label = os.path.basename(label)
            score = img['score']

            r = idx_data[roll]
            i = idx_data_img[label]
            matrix[r][i] = score
    # np.savetxt("matrix_new.csv", matrix, delimiter=",", fmt="%.6f")

    # import seaborn as sns
    # plt.figure(figsize=(18, 12))
    # sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={'size': 6})
    # # plt.title("Matrix Visualization (Heatmap)", fontsize=16)
    # plt.xlabel("Image Match - Column Index")
    # plt.ylabel("Students - Row Index")
    # plt.show()
    def assign_images(sim_matrix):
        n_students, n_images = sim_matrix.shape
        assigned_students = set()
        assigned_images = set()
        matches = []
        # Create list of (student_idx, image_idx, sim_score)
        pairs = [(i, j, sim_matrix[i][j]) for i in range(n_students) for j in range(n_images)]
        # Sort by similarity score (descending)
        pairs.sort(key=lambda x: x[2])
        for student, image, score in pairs:
            if student not in assigned_students and image not in assigned_images:
                matches.append([student, image, score])
                assigned_students.add(student)
                assigned_images.add(image)
                if len(assigned_images) == n_images:
                    break
        return matches
    matches = assign_images(matrix)
    # print("Matches:", matches)
    filtered_matches = {}
    for match in matches:
        stu_id, img_id, score = match 
        stu_name = dec_idx_data[stu_id]
        img_name = dec_idx_data_img[img_id]
        # print("Matches::", stu_name, img_name, score)
        if score<thresh: 
            filtered_matches[stu_name]={"label":img_name, "score":score}

    print(f"Length of filtered Matches: {len(filtered_matches)}")
    # print(filtered_matches)
    return filtered_matches


# to generate final results for comparing results

import csv

def write_results(file_name='gt_list.csv', data_dict={}):
    keys_list = list(data_dict.keys())
    # Writing the list to a CSV file
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Roll', 'Score'])  # Adding a header
        for key in keys_list:
            writer.writerow([key, round(data_dict[key]['score'], 3)])
    print(f"written csv at:", file_name)

if __name__=="__main__":
    # Where clicked pictures are stored
    # folder_path = r"../data/06-02-2025" # jpeg 
    # folder_path = r"../data/04-02-2025" # heic 
    # # folder_path = r"../data/test" # jpeg 

    # run_detector, run_recognizer = True, True
    # run_detector = False
    # # run_recognizer = False
    # # feature_pickle_path = None
    # # feature_pickle_path = "gt_features_90clusters.pkl"
    # json_path = run_pipeline(folder_path, run_detector=run_detector, run_recognizer=run_recognizer, feature_pickle_path=feature_pickle_path)

    ############################################################################################ FOLDER PROCESSING
    import json
    # month_attd_folder = r"E:\code\DL_ClassFR\RECORDS\captured_images\day_attendance"
    # month_attd_folder = r"E:\code\DL_ClassFR\RECORDS\captured_images\evaluation"
    # month_attd_folder = r"E:\code\DL_ClassFR\RECORDS\day_single"
    month_attd_folder = r"E:\code\DL_ClassFR\RECORDS\before_midsem_photo"
    attendance_folders = os.listdir(month_attd_folder)
    # attendance_folders = glob(month_attd_folder + os.path.sep + "*" + os.path.sep)
    # print(attendance_folders)
    for date in attendance_folders:
        date_path = os.path.join(month_attd_folder, date)
        if os.path.isfile(date_path):
            continue # no processing of files just folders

        print(date_path)
        folder_path = date_path
        run_detector = True
        run_recognizer = True

        json_path = run_pipeline(folder_path, run_detector=run_detector, run_recognizer=run_recognizer, feature_pickle_path=feature_pickle_path)
        new_json_path = json_path.replace("results", f"results_{date}_{time()}")  
        os.rename(json_path, new_json_path)

        det_faces = os.path.join(root_path, "detected_faces")
        try:
            os.rename(det_faces, f"{det_faces}_{date}")
        except Exception as e:
            os.rename(f"{det_faces}_{date}", f"{det_faces}_{date}_{time()}")
            os.rename(det_faces, f"{det_faces}_{date}")
            print("Error in Main", e)

        ################# WRITING AND GENERATING OUTPUT CSV 
        # new_json_path = os.path.join(root_path, "results_old_1745758856.9915643.json")
        with open(new_json_path, 'r') as f: data = json.load(f)
        # filtered_matches = filter_results_simple(data, thresh=1)
        filtered_matches = filter_results_greedy(data, thresh=1)

        # print(filtered_matches)
        op_csv_path = os.path.join(month_attd_folder, f"pred_{date}.csv")
        write_results(op_csv_path, filtered_matches)
        # break


