import cv2
import mediapipe as mp
import os
import shutil
from time import time
import shutil
from config import root_path

def separate_noisy_images(base_folder, result_folder):
    """
    Process images to filter those with visible eyes.

    Args:
        base_folder (str): Path to the folder containing input images.
        result_folder (str): Path to the folder where results will be stored.

    Returns:
        dict: A summary of the processing, including counts of total, saved, and noisy images.
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # Paths for result subfolders
    saved_folder = os.path.join(result_folder, "Good")
    noise_folder = os.path.join(result_folder, "Noise")

    # Create result subfolders if they don’t exist
    os.makedirs(saved_folder, exist_ok=True)
    os.makedirs(noise_folder, exist_ok=True)

    # Counters to keep track of processing results
    total_images = 0
    saved_count = 0
    noise_count = 0

    # Process each image in the base folder
    for img_name in os.listdir(base_folder):
        img_path = os.path.join(base_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            # Skip if image cannot be read
            print(f"⚠️ Skipping {img_name}: Unable to read the image.")
            continue

        total_images += 1
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Check if face landmarks are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                try:
                    # Extract landmarks for left and right eyes
                    left_eye = face_landmarks.landmark[33]  # Left eye
                    right_eye = face_landmarks.landmark[263]  # Right eye

                    # Validate if eyes are visible
                    left_eye_visible = left_eye.x > 0 and left_eye.y > 0
                    right_eye_visible = right_eye.x > 0 and right_eye.y > 0

                    if left_eye_visible or right_eye_visible:
                        # Save images with visible eyes
                        shutil.copy(img_path, os.path.join(saved_folder, img_name))
                        saved_count += 1
                        # print(f"✅ Saved: {img_name}")
                    else:
                        # Save noisy images if eyes are not visible
                        shutil.copy(img_path, os.path.join(noise_folder, img_name))
                        noise_count += 1
                        # print(f"❌ Noise: {img_name} (Eyes not detected)")
                except:
                    # Handle any errors while processing landmarks
                    shutil.copy(img_path, os.path.join(noise_folder, img_name))
                    noise_count += 1
                    # print(f"❌ Noise: {img_name} (Error processing landmarks)")
        else:
            # Save images without detected faces as noisy
            shutil.copy(img_path, os.path.join(noise_folder, img_name))
            noise_count += 1
            # print(f"❌ Noise: {img_name} (No face detected)")

    # Print summary of processing results
    print("\n📊 Process Summary:")
    print(f"📸 Total Images Processed: {total_images}")
    print(f"✅ Saved Images: {saved_count}")
    print(f"❌ Noise Images: {noise_count}")

    # Return summary as a dictionary
    return {
        "total_images": total_images,
        "Good_images": saved_count,
        "noise_images": noise_count
    }



def detect_frontal_faces(input_folder, frontal_output_folder, others_output_folder, confidence_threshold=0.8):
    """
    Detect frontal faces in images using the YuNet model.

    Args:
        input_folder (str): Path to the folder containing input images.
        frontal_output_folder (str): Path to the folder where images with frontal faces will be saved.
        others_output_folder (str): Path to the folder where other images will be saved.
        confidence_threshold (float): Confidence threshold for face detection.

    Returns:
        None
    """
    print("Doing front Face Detection")
    # Load YuNet model
    model_path = "recognizer/face_detection_yunet_2023mar.onnx"  # Ensure the model is downloaded and available
    yunet = cv2.FaceDetectorYN_create(
        model_path,
        "",
        (320, 320),
        confidence_threshold,
        0.3,  # Non-maximum suppression (NMS) threshold
        5000  # Maximum number of candidates
    )
    
    # Create output directories for classified images
    os.makedirs(frontal_output_folder, exist_ok=True)
    os.makedirs(others_output_folder, exist_ok=True)

    # Process each image in the input folder
    for img_name in os.listdir(input_folder):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Skip unsupported file formats
            continue
            
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            # Skip if image cannot be read
            continue
            
        # Set input size for the detector
        h, w = img.shape[:2]
        yunet.setInputSize((w, h))
        
        # Detect faces in the image
        _, faces = yunet.detect(img)
        
        # Save image based on detection results
        if faces is not None and len(faces) > 0:
            output_path = os.path.join(frontal_output_folder, img_name)  # Save to "Frontal"
            cv2.imwrite(output_path, img)
        else:
            output_path = os.path.join(others_output_folder, img_name)  # Save to "Others"
            cv2.imwrite(output_path, img)
    print("Completed: front Face Detection")


def clean_images(detected_faces_path=None):
    final_results_path = "Results/noise_remove"
    final_results_path = os.path.join(root_path, final_results_path)

    # Remove noisy images and detect those with visible eyes
    result = separate_noisy_images(detected_faces_path, final_results_path)
    print(result)

    frontal_output_folder=os.path.join(os.path.dirname(final_results_path), "Final_front")
    input_folder = os.path.join(final_results_path, "Good")
    others_output_folder=os.path.join(os.path.dirname(final_results_path), "Final_others")

    # Further classify images with detected frontal faces
    detect_frontal_faces(
        input_folder=input_folder,
        frontal_output_folder=frontal_output_folder,
        others_output_folder=others_output_folder,
        confidence_threshold=0.9  # Adjust as needed
    )
    os.rename(detected_faces_path, f"{detected_faces_path}_noisy_{time()}")
    os.rename(frontal_output_folder, detected_faces_path)

    os.rename(os.path.join(root_path, "Results"), os.path.join(root_path, f"Results_old_{time()}"))
    # shutil.rmtree(final_results_path)
    print("removed directory")


# Example Usage:

if __name__ == "__main__":
    # final_results_path = "E:\code\DL_ClassFR\data\detected_faces"
    detected_faces_path = "E:\code\DL_ClassFR\data\detected_faces"
    clean_images(detected_faces_path)