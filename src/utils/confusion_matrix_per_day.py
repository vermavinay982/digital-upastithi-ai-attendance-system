import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def generate_attendance_confusion_matrix(
    all_students_file, 
    gt_present_file, 
    pred_present_file, 
    output_file='confusion_matrix.png',
    dpi=300,
    figsize=(8, 6),
    cmap='Blues',
    title='Attendance Prediction Confusion Matrix'
):
    """
    Generate and save a confusion matrix for an image-based attendance system.
    
    Parameters:
    -----------
    all_students_file : str
        Path to CSV containing all student rolls (e.g., 'list_all.csv').
    gt_present_file : str
        Path to CSV containing ground truth present students (e.g., 'list_gt.csv').
    pred_present_file : str
        Path to CSV containing predicted present students (e.g., 'list_pred.csv').
    output_file : str, optional (default: 'confusion_matrix.png')
        Path to save the output image.
    dpi : int, optional (default: 300)
        Image resolution (dots per inch).
    figsize : tuple, optional (default: (8, 6))
        Figure size (width, height).
    cmap : str, optional (default: 'Blues')
        Colormap for the heatmap.
    title : str, optional (default: 'Attendance Prediction Confusion Matrix')
        Title of the plot.
    
    Returns:
    --------
    dict
        A dictionary containing performance metrics (accuracy, precision, recall, F1).
    """
    
    # Load data safely (handling missing files)
    try:
        all_students = pd.read_csv(all_students_file)['Roll'].tolist()
        gt_present = pd.read_csv(gt_present_file)['Roll'].tolist()
        pred_present = pd.read_csv(pred_present_file)['Roll'].tolist()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing input file: {e.filename}")

    # Check for empty lists
    if not all_students:
        raise ValueError("No students found in the 'all_students' list.")
    
    # Prepare labels (1 = Present, 0 = Absent)
    y_true = [1 if student in gt_present else 0 for student in all_students]
    y_pred = [1 if student in pred_present else 0 for student in all_students]

    # Compute confusion matrix and metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
    }

    # Plotting
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=cmap,
        xticklabels=['Predicted Absent', 'Predicted Present'],
        yticklabels=['Actually Absent', 'Actually Present'],
        cbar=False
    )
    
    plt.title(title, pad=20, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)

    # Add metrics text box
    metrics_text = (
        f"Accuracy: {metrics['accuracy']:.2f}\n"
        f"Precision: {metrics['precision']:.2f}\n"
        f"Recall: {metrics['recall']:.2f}\n"
        f"F1 Score: {metrics['f1_score']:.2f}"
    )
    plt.text(
        2.5, 0.5, 
        metrics_text, 
        bbox=dict(facecolor='white', alpha=0.5),
        fontsize=10
    )

    # Save and close
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrix saved to: {output_file}")
    return metrics


if __name__ == "__main__":

    # Generate confusion matrix and get metrics
    metrics = generate_attendance_confusion_matrix(
        r'C:\Users\MEET\Desktop\IIIT_D\Sem-2\DL\Evaluation\Data\all_04_Feb.csv', 
        r'C:\Users\MEET\Desktop\IIIT_D\Sem-2\DL\Evaluation\Project\attendance_data\gt_06_Feb.csv', 
        r'C:\Users\MEET\Desktop\IIIT_D\Sem-2\DL\Evaluation\Project\attendance_data\attendance_06_feb.csv',
        output_file='today_attendance_confusion.png'
    )

    # Log metrics (example)
    print("Today's Attendance Performance:")
    print(f"- Accuracy: {metrics['accuracy']:.2f}")
    print(f"- Missed Detections (False Negatives): {metrics['false_negatives']}")
    print(f"- Wrong Detections (False Positives): {metrics['false_positives']}")