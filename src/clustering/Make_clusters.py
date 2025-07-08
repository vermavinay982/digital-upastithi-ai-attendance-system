import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
import shutil
import matplotlib.pyplot as plt
from deepface import DeepFace

'''
All faces path Google Drive : "https://drive.google.com/drive/folders/1RCdTCHyDftlMI2XuRqBZNdoTo_H2BPeX?usp=sharing"
'''

def setup_gpu():
    """
    Configure TensorFlow to use the GPU, or fallback to CPU if GPU setup fails.
    """
    print("[INFO] Setting up TensorFlow GPU configuration...")
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                print("[INFO] GPU is successfully configured.")
            except RuntimeError as e:
                print(f"[ERROR] RuntimeError during GPU setup: {e}")
                print("[INFO] Falling back to CPU...")
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            print("[INFO] No GPU found, using CPU...")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    except Exception as e:
        print(f"[ERROR] Exception during GPU setup: {e}")
        print("[INFO] Falling back to CPU...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def cluster_faces_with_pca(input_directory, output_directory, model_name="Facenet", eps=0.5, min_samples=5, n_components=50):
    """
    Cluster face embeddings using DBSCAN with PCA-reduced embeddings.
    """
    # Load images and extract embeddings
    print("[INFO] Extracting embeddings for face images...")
    image_paths = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    embeddings, valid_image_paths = [], []

    for img_path in tqdm(image_paths, desc="Extracting embeddings"):
        try:
            embedding = DeepFace.represent(img_path, model_name=model_name, enforce_detection=False)[0]["embedding"]
            embeddings.append(embedding)
            valid_image_paths.append(img_path)
        except Exception as e:
            print(f"[WARNING] Skipping {img_path}: {str(e)}")

    if not embeddings:
        print("[ERROR] No valid embeddings extracted. Exiting.")
        return

    # Normalize and reduce dimensions
    embeddings = normalize(np.array(embeddings))
    print(f"[DEBUG] Normalized embeddings shape: {embeddings.shape}")

    print("[INFO] Reducing dimensions with PCA...")
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"[DEBUG] PCA-reduced embeddings shape: {reduced_embeddings.shape}")

    # Perform clustering
    print("[INFO] Clustering embeddings...")
    clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    cluster_labels = clustering_model.fit_predict(reduced_embeddings)

    # Save clusters
    unique_clusters = set(cluster_labels)
    print(f"[INFO] Found {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} unique clusters.")
    os.makedirs(output_directory, exist_ok=True)

    for cluster_id in tqdm(unique_clusters, desc="Saving clusters"):
        cluster_dir = os.path.join(output_directory, f"cluster_{cluster_id}" if cluster_id != -1 else "noise")
        os.makedirs(cluster_dir, exist_ok=True)
        for i, label in enumerate(cluster_labels):
            if label == cluster_id:
                shutil.copy(valid_image_paths[i], os.path.join(cluster_dir, os.path.basename(valid_image_paths[i])))

    # Visualize clusters
    print("[INFO] Visualizing clusters...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap="rainbow", s=5)
    plt.colorbar(scatter, label="Cluster Label")
    plt.title("Cluster Visualization (PCA-reduced embeddings)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()

# Set up GPU/CPU for TensorFlow
setup_gpu()

# Define directories
input_directory = "/content/drive/MyDrive/DL_Project/All_Faces"
output_directory = "/content/drive/MyDrive/DL_Project/grouped_faces_all_2"

# Run the function 
cluster_faces_with_pca(input_directory, output_directory, eps=0.5, min_samples=4, n_components=30)
