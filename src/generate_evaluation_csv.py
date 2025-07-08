import json
import os 
from time import time

from main import run_pipeline, filter_results_greedy, write_results
from config import root_path, feature_pickle_path, label, FR_THRESH


# month_attd_folder = r"E:\code\DL_ClassFR\RECORDS\captured_images\day_attendance"
# month_attd_folder = r"E:\code\DL_ClassFR\RECORDS\captured_images\evaluation"
# month_attd_folder = r"E:\code\DL_ClassFR\RECORDS\day_single"
# month_attd_folder = r"E:\code\DL_ClassFR\RECORDS\before_midsem_photo"
month_attd_folder = r"..\RECORDS\final_eval"



######################################################################
attendance_folders = os.listdir(month_attd_folder)
print(attendance_folders)

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
    new_json_path = os.path.join(month_attd_folder, os.path.basename(new_json_path))
    os.rename(json_path, new_json_path)
    print("Results JSON Saved")

    det_faces = os.path.join(root_path, "detected_faces")
    new_det_faces = os.path.join(root_path, f"{det_faces}_{date}_{time()}")
    try:
        os.rename(det_faces, new_det_faces)
    except Exception as e:
        print("Error in Main", e)

    ################# WRITING AND GENERATING OUTPUT CSV 
    # new_json_path = os.path.join(root_path, "results_old_1745758856.9915643.json")
    with open(new_json_path, 'r') as f: data = json.load(f)
    # filtered_matches = filter_results_simple(data, thresh=1)
    filtered_matches = filter_results_greedy(data, thresh=1)

    # print(filtered_matches)
    op_csv_path = os.path.join(month_attd_folder, f"{label}_{date}.csv")
    write_results(op_csv_path, filtered_matches)
    # break


######################################################################

# CREATING FINAL CSV REQUIRED
from merge_attendance import write_combined_csv
threshold = FR_THRESH # from config.py
output_csv_name = f"cse641_attendance.csv"
write_combined_csv(month_attd_folder, output_csv_name, threshold, label)

# FOR VARYING THRESHOLD
# for threshold in range(0,110,10):
#     threshold = threshold/100
#     output_csv_name = f"cse641_attendance_{threshold}.csv"
#     write_combined_csv(month_attd_folder, output_csv_name, threshold)
#     # break

######################################################################
