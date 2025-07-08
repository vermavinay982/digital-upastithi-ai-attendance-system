import streamlit as st
import json
import os
import shutil
from PIL import Image
from glob import glob
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


ITEMS_PER_PAGE = 100
global SHOW
SHOW = False

st.set_page_config(page_title='Digital Upastithi', layout="wide")

import sys
sys.path.append(".")
from config import gt_folder_path, image_op_folder, stud_details_path, FR_THRESH
from main import filter_results_greedy as filter_results
# from main import filter_results_simple as filter_results


op_folder = image_op_folder

def load_json(json_path=None):
    global SHOW
    # uploaded_file = st.file_uploader("Upload JSON File", type=["json"])
    # if uploaded_file is not None:
    #     return json.load(uploaded_file), uploaded_file.name
    # return None, None

    json_path = st.text_input(f"Results JSON Path", '../run_files/results.json')
    # image_folder = st.text_input(f"Images Path", '../run_files/detected_faces')
    image_folder = st.text_input(f"Images Folder", 'detected_faces')


    # print(f"JSON path received: {json_path}")
    if json_path is None:
        return None, None
    
    if True or SHOW or st.button("Show Results"):
        if os.path.exists(json_path):
            SHOW = True
            with open(json_path,'r') as f:
                return json.load(f), json_path, image_folder
    return None, None, None


# def load_json(json_path=None):
#     # if json_path is None:
#     uploaded_file = st.file_uploader("Upload JSON File", type=["json"])
#     if uploaded_file is not None:
#         return json.load(uploaded_file), uploaded_file.name
#     # else:
#     if json_path is None:
#         return None, None

#     return json.load(json_path), json_path

def save_attendance(data, UPLOAD_FOLDER, label="pred"):
    unique_labels = set()
    for page, page_data in data.items():
        for img_path, values in page_data.items():
            if values['correct']:  # Only save if marked correct
                student_name = values['label'].strip().capitalize()
                # print(values)
                student_name = f"{student_name}, {values['score']:.2f}"
                if student_name:
                    unique_labels.add(student_name)
                    # data_path = os.path.join(op_folder, label)
                    # os.makedirs(data_path, exist_ok=True)
                    # new_path = os.path.join(data_path, os.path.basename(img_path))
                    # shutil.copy(img_path, new_path)

    attendance_file = f"{label}_{UPLOAD_FOLDER}.csv"
    with open(attendance_file, "w") as f:
        f.write("\n".join(unique_labels))
    st.success(f"Saved records successfully!\nPath: {attendance_file}")

def show_attendance(UPLOAD_FOLDER, label="pred"):
    attendance_file = f"{label}_{UPLOAD_FOLDER}.csv"
    if os.path.exists(attendance_file):
        csv_data = "Roll, Score\n"
        with open(attendance_file, "r") as f:
            records = f.readlines()
        csv_data+="".join(records)
        # Add download button
        st.download_button(
            label="📥 Download Attendance CSV",
            data=csv_data,
            file_name=attendance_file,
            mime="text/csv"
        )
        
        st.subheader("Attendance Records")
        for record in records:
            st.write(record.strip())
            
    else:
        st.error("Attendance file not found!")



def main(json_path=None, UPLOAD_FOLDER=None):
    st.title("Digital Upasthiti - Viewer")
    json_data, json_path, image_folder = load_json(json_path)


    if json_data:
        user_data = {}
        total_rec = len(json_data)

        filtered_matches = filter_results(json_data)
        total_rec_filtered = len(filtered_matches)

        stud_details = {}
        if os.path.exists(stud_details_path):
            with open(stud_details_path,'r') as f:
                stud_details = json.load(f)
        # print(stud_details)
        
        # Pagination setup
        page_number = st.number_input("Page", min_value=1, max_value=(total_rec // ITEMS_PER_PAGE) + 1, step=1, value=1)
        start_idx = (page_number - 1) * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        paginated_items = list(json_data.items())[start_idx:end_idx]
        print(json_path)

        # if len(json_path.split('_'))==2:
        #     new_df = f"detected_faces_{json_path.split('_')[0]}"
        # new_df = f"detected_faces"
        new_df = image_folder

        cnt_idx = 1
        for idx, (cluster_name, results) in enumerate(paginated_items, start=start_idx + 1):
            img_path = glob(os.path.join(gt_folder_path, cluster_name, "*"))[0] # gt image

            if not (cluster_name in filtered_matches.keys()): continue # removing unassigned images


            if os.path.exists(img_path):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**Student ({cnt_idx}/{total_rec_filtered}):**")
                    cnt_idx+=1
                    # st.write(f"**Student ({idx}/{total_rec}):**")
                    image = Image.open(img_path).resize((100, 100))
                    # student_name = f"{cluster_name}"
                    try:

                        student_name = f"{cluster_name} - {stud_details.get(cluster_name).get('name').replace('detected_faces',f'{new_df}')}"
                    except Exception as e:
                        print(f"Exception printing name: {e}: {cluster_name}")
                        student_name = f"{cluster_name}"

                    st.image(image, caption=student_name, use_container_width=False)

                    img_path_fil = filtered_matches[cluster_name]["label"].replace('detected_faces',f'{new_df}')
                    fil_score = filtered_matches[cluster_name]["score"]
                    image = Image.open(img_path_fil).resize((100, 100))
                    st.image(image, caption=os.path.basename(img_path_fil), use_container_width=False)

                    # confident_result = results[0]['score'] < 0.3
                    confident_result = float(fil_score) <= FR_THRESH
                    is_correct = st.checkbox(f'"{cluster_name}"  Correct? ({fil_score:.2f})', confident_result)

                with col2:
                    st.write("**Similar Matches:**")
                    n = 5
                    sim_cols = st.columns(min(len(results[0:n]), 6))
                    for i, (col, result) in enumerate(zip(sim_cols, results[0:n])):
                        with col:
                            label = result['label'][-10:].capitalize()
                            # print(label)
                            st.write(f"{i+1}. ({result['score']:.2f}) {label}")
                            sim_image_path = result['label'].replace('detected_faces',f'{new_df}')
                            # files = glob(os.path.join(sim_image_path, "*"))
                            sim_image = Image.open(sim_image_path).resize((80, 80))
                            st.image(sim_image, use_container_width=False)

                    # choices = [result["label"].capitalize() for result in results] + ["None"]
                    # best_match = results[0]["label"].capitalize() if confident_result else "None"
                    # selected_label = st.selectbox(f"Label for {os.path.basename(img_path)}", choices, index=choices.index(best_match))
                    # other_label = st.text_input(f"Other label for {os.path.basename(img_path)}", "")
                    
                if page_number not in user_data.keys():
                    user_data[page_number] = {}

                user_data[page_number][img_path] = {"label": cluster_name, "correct": is_correct, "score": fil_score}
            st.write("______________________")

        if st.button("Save Attendance"):
            save_attendance(user_data, UPLOAD_FOLDER)

    if st.button("Show Attendance"):
        show_attendance(UPLOAD_FOLDER)

def run_main(inp_path=None, UPLOAD_FOLDER=None):
    main(json_path=inp_path, UPLOAD_FOLDER=UPLOAD_FOLDER)


if __name__ == "__main__":
    main()
