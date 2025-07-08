"""streamlit run .\gui\main_upload_streamlit.py

You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.32.54:8501
"""

import streamlit as st
import os
import sys
path = "./"
sys.path.append(path)
# print(os.listdir(path))
from main import run_pipeline # recognition pipeline
from main_streamlit import run_main # display pipeline
from time import time

from config import gt_folder_path, image_op_folder, root_path, feature_pickle_path
# st.set_page_config(page_title='Digital Upastithi', layout="wide")

def save_uploaded_file(uploaded_file, UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, f"{uploaded_file.name}")
    # file_path = os.path.join(UPLOAD_FOLDER, f"{str(time()).replace('.','_')}_{uploaded_file.name}")

    print(f"Writing: {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def handle_image_upload():
    UPLOAD_FOLDER = st.text_input(f"Folder Name", "")
    UPLOAD_FOLDER = os.path.join("../run_files/uploads", UPLOAD_FOLDER)
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png", "heic"], accept_multiple_files=True)
    image_paths = []
    try:
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file, UPLOAD_FOLDER)
                image_paths.append(file_path)
    except Exception as e:
        print(f"Failed Uploading Files: {e}")
        return False
    
    print("Written all the Files")
    return UPLOAD_FOLDER
    # captured_image = st.camera_input("Capture Image from Camera")
    # if captured_image:
    #     file_path = save_uploaded_file(captured_image)
    #     image_paths.append(file_path)
    if image_paths:
        st.success(f"Saved {len(image_paths)} image(s) to '{UPLOAD_FOLDER}'")
    return image_paths

def run_recognition(image_folder):
    # det_folder_path = "detected_faces"
    # gt_folder_path = "../data/GroundTruth"
    st.info("Running Recognizer...")
    st.success("Recognition Completed!")

def main():
    st.title("Digital Upasthiti - AI Attendance System")
    UPLOAD_FOLDER = handle_image_upload()
    json_path = None

    if st.button("Run Recognition"):

        if UPLOAD_FOLDER:
            json_path = run_pipeline(folder_path=UPLOAD_FOLDER, feature_pickle_path=feature_pickle_path)

            print(f"JSON Path: {json_path}")
        else:
            st.error("No images uploaded or captured!")
    run_main(json_path, os.path.basename(UPLOAD_FOLDER).split('.')[0])

if __name__ == "__main__":
    main()
