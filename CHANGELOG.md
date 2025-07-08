# Changelog

## 2 Apr
- main pipeline streamlined to work on other devices
- streamlit viewer integrated with full pipeline
- renamed main tk for tkinter version
- added main upload streamlit, which allows uploading photos from phone
- recognize now creates pickle by default, if pkl there, then save time

- move old folder of detected faces to time based folder
- removed extra logging
- json auto loading, custom name allowed
- fixed loading order of elements
- hide results until button pressed

## 12 Mar
- converted tkinter to streamlit web app, as per requirement

## 11 Mar 
- Added Make_cluster.py in src/clustering folder which takes input all face images and make unique faces folders

## 27 Feb
- yolo custom path solved (Meet)
- HEIC format code added (Meet)
- conditions to check images (Meet)
- FIXED: heic has issues in loading (Meet)
- created GUI to display results, Tkinter
- best match similar match displayed
- dropdown to select correct label for each student
- allow to read from json, and export the records for teacher's use
- correct box added, to only consider the entries which we mark are correct, reduce crowded results
- saving the images after result
- if less than threshold, None is allocated

## 24 Feb
- added the code for feature extractor as baseline
- default to writing images for cross platform compatibility and easy debugging
- writing rank based predictions for each detected face (results.json), it will help to debug later
- added pickle to save time in generating for ground truth images
- recognizer code written for deepface
- detector intermediate face writing enabled, for easy future model replacement and post processing issues
- recognizer code updated to allowing to change models, separately by any developer 

## 22 Feb
- updated yoloface as face detector, it is better


## 12 Feb
- updated the pipeline
- added mediapipe as face detector
