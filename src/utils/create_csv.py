import pandas as pd
import os


# creating folder, putting gt and values for each

attendance_path = "attendance_gt_past.csv"

# Load the CSV (assuming it's named 'attendance.csv')
df = pd.read_csv(attendance_path, engine='python')  # Tab-separated values

# Replace empty strings and strip whitespace
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# Transpose data so we can process by date
dates = df.columns[1:]  # Skip the roll number column
roll_numbers = df.iloc[:, 0]

# Create output folder
output_dir = 'attendance_by_date'
os.makedirs(output_dir, exist_ok=True)

# For each date column, create a CSV with roll numbers marked 'P'
for date in dates:
    present_rolls = roll_numbers[df[date] == 'P']
    if not present_rolls.empty:
        op_folder = os.path.join(output_dir, f"{date.replace('-','_')}")
        os.makedirs(op_folder, exist_ok=True)
        op_path = os.path.join(op_folder, f"gt_{date.replace('-','_')}.csv")
        present_rolls.to_csv(op_path, index=False, header=True)

# For each date column, create a CSV with roll numbers marked 'P'
for date in dates:
    present_rolls = roll_numbers[df[date].isin(['P', 'A'])]
    if not present_rolls.empty:
        op_folder = os.path.join(output_dir, f"{date.replace('-','_')}")
        os.makedirs(op_folder, exist_ok=True)
        op_path = os.path.join(op_folder, f"all_{date.replace('-','_')}.csv")
        present_rolls.to_csv(op_path, index=False, header=True)

print("Attendance CSVs by date created in 'attendance_by_date' folder.")
