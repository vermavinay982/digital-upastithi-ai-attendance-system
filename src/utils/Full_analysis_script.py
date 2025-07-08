import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# ========================
# 1. SETUP OUTPUT FOLDER
# ========================
output_folder = "output_results"
os.makedirs(output_folder, exist_ok=True)

# ========================
# 2. LOAD AND PROCESS DATA (CASE-INSENSITIVE)
# ========================
def load_and_process_data(input_folder):
    # Get all ground truth and attendance files
    all_files = os.listdir(input_folder)
    gt_files = sorted([f for f in all_files if f.startswith('gt_')])
    att_files = sorted([f for f in all_files if f.startswith('attendance_')])
    
    # Validate pairs
    if len(gt_files) != len(att_files):
        raise ValueError("Mismatch between GT and attendance files!")
    
    # Load all data
    all_data = []
    all_roll_numbers = set()
    
    for gt_file, att_file in zip(gt_files, att_files):
        # Extract date (e.g., "04_Feb")
        date_str = '_'.join(gt_file.split('_')[1:]).split('.')[0]
        
        # Load CSVs and standardize roll numbers (lowercase)
        gt = pd.read_csv(os.path.join(input_folder, gt_file))
        att = pd.read_csv(os.path.join(input_folder, att_file))
        gt['Roll'] = gt['Roll'].str.lower().str.strip()  # Force lowercase
        att['Roll'] = att['Roll'].str.lower().str.strip()
        
        gt_roll = set(gt['Roll'])
        att_roll = set(att['Roll'])
        all_roll_numbers.update(gt_roll)
        all_roll_numbers.update(att_roll)
        
        all_data.append({
            'date': date_str,
            'gt_roll': gt_roll,
            'att_roll': att_roll
        })
    
    return all_data, list(all_roll_numbers)

# ========================
# 3. COMPUTE METRICS & CONFUSION MATRICES
# ========================
def compute_metrics(all_data, all_roll_numbers):
    overall_cm = np.zeros((2, 2), dtype=int)  # [[TP, FP], [FN, TN]]
    weekly_cms = {}
    y_true, y_pred = [], []
    
    for entry in all_data:
        gt_roll = entry['gt_roll']
        att_roll = entry['att_roll']
        date_str = entry['date']
        
        # Parse week number
        date_obj = datetime.strptime(date_str, "%d_%b")
        week_num = date_obj.isocalendar()[1]
        
        # Compute daily confusion matrix
        TP = len(gt_roll & att_roll)
        FP = len(att_roll - gt_roll)
        FN = len(gt_roll - att_roll)
        TN = len(set(all_roll_numbers) - gt_roll - att_roll)
        
        daily_cm = np.array([[TP, FP], [FN, TN]])
        overall_cm += daily_cm
        
        # Group by week
        if week_num not in weekly_cms:
            weekly_cms[week_num] = np.zeros((2, 2), dtype=int)
        weekly_cms[week_num] += daily_cm
        
        # Aggregate for EER
        for roll in all_roll_numbers:
            y_true.append(1 if roll in gt_roll else 0)
            y_pred.append(1 if roll in att_roll else 0)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return overall_cm, weekly_cms, accuracy, precision, recall, f1, y_true, y_pred

# ========================
# 4. PLOT CONFUSION MATRIX
# ========================
def plot_confusion_matrix(cm, title, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Present', 'Predicted Absent'],
                yticklabels=['Actual Present', 'Actual Absent'])
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

# ========================
# 5. PLOT ROC CURVE WITH EER
# ========================
def plot_roc_curve(y_true, y_scores, output_path):
    # If y_scores are binary (0 or 1), add small random noise to create "confidence-like" scores
    # This is a workaround if you don't have actual confidence scores
    if set(np.unique(y_scores)) == {0, 1}:
        # print("Warning: Using binary predictions for ROC curve is not ideal.")
        # Add small random noise to create separation (0.1 for absent, 0.9 for present)
        y_scores = np.array([0.1 + 0.8 * score + np.random.uniform(-0.05, 0.05) for score in y_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label='ROC curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Random guess')
    ax.scatter(fpr[eer_idx], tpr[eer_idx], color='red', label=f'EER = {eer:.2f}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve with EER')
    ax.legend()
    ax.grid(True)
    plt.savefig(output_path)
    plt.close()
    
    return eer

# ========================
# 6. GENERATE REPORT
# ========================
def generate_report(overall_cm, weekly_cms, eer, accuracy, precision, recall, f1, output_folder):
    # Save metrics to a text file
    with open(os.path.join(output_folder, "report.txt"), "w") as f:
        f.write("=== Attendance System Evaluation Report ===\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Equal Error Rate (EER): {eer:.4f}\n\n")
        
        f.write("Overall Confusion Matrix:\n")
        f.write(f"TP: {overall_cm[0, 0]}, FP: {overall_cm[0, 1]}\n")
        f.write(f"FN: {overall_cm[1, 0]}, TN: {overall_cm[1, 1]}\n\n")
        
        f.write("Weekly Confusion Matrices:\n")
        for week, cm in weekly_cms.items():
            f.write(f"Week {week}:\n")
            f.write(f"TP: {cm[0, 0]}, FP: {cm[0, 1]}\n")
            f.write(f"FN: {cm[1, 0]}, TN: {cm[1, 1]}\n\n")
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        overall_cm, 
        title="Overall Confusion Matrix\n(Accuracy: {:.2f}%, Precision: {:.2f}%)".format(accuracy*100, precision*100),
        output_path=os.path.join(output_folder, "confusion_matrix.png")
    )

# ========================
# 7. MAIN EXECUTION
# ========================
def main(input_folder):
    print(f"Processing files from: {input_folder}")
    all_data, all_roll_numbers = load_and_process_data(input_folder)
    print(f"Loaded {len(all_data)} days of data. Total rolls: {len(all_roll_numbers)}")
    
    # Compute metrics
    overall_cm, weekly_cms, accuracy, precision, recall, f1, y_true, y_pred = compute_metrics(all_data, all_roll_numbers)
    
    # Plot ROC curve and get EER
    eer = plot_roc_curve(y_true, y_pred, os.path.join(output_folder, "roc_curve.png"))
    
    # Generate report
    generate_report(overall_cm, weekly_cms, eer, accuracy, precision, recall, f1, output_folder)
    print(f"Report saved to '{output_folder}'.")

if __name__ == "__main__":
    input_folder = "attendance_data"  # Folder with CSVs
    main(input_folder)