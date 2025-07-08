import pandas as pd
from glob import glob
import os
import json
from config import root_path
from config import data_path

def write_combined_csv(att_path=None, output_csv_name="cse641_attendance.csv", threshold=0.5, label="pred", use_thresh=True):
    input_csv_name =  "cse641_student_list.csv"
    input_csv = os.path.join(data_path, input_csv_name) 
    
    output_csv = os.path.join(att_path, output_csv_name) 

    # ALL STUDENTS AVAILABLE FOR ATTENDANCE
    df = pd.read_csv(input_csv) # Load existing CSV with roll numbers and dates
    df.set_index("Roll No", inplace=True)
    df.index = df.index.astype(str).str.upper() # Normalize roll numbers to string and uppercase for comparison

    # Process each attendance file
    att_files = glob(f"{att_path}/{label}_*.csv")

    print(f"\n\n[{len(att_files)}] Files found at [{att_path}], with label: {label}")
    for att_file in att_files:
        # Extract date in format DD-MM-YYYY
        date = os.path.basename(att_file).split(f"{label}_")[-1].split(".")[0].replace("_", "-")
        present_df = pd.read_csv(att_file) # Read today's attendance file
        


        if use_thresh:
            # NOW USING THE THRESHOLD VALUES 
            present_df['Score'].apply(lambda x: float(x))
            present_rolls = present_df[present_df['Score'] <= threshold]['Roll'].tolist()
        else:
            # NOT USING THE THRESHOLD VALUES 
            present_rolls = present_df['Roll'].astype(str).str.upper().tolist()

        print(present_rolls)
        df[date] = 'A' # Mark All Absent
        df.loc[df.index.isin(present_rolls), date] = 'P' # Mark Present
        # print(f"Processed ==> {date}")
        # print(present_rolls)
        print(f"Processed ==> {date}, [{len(present_rolls)}/{len(present_df)}]")

    df.to_csv(output_csv) # Save the updated CSV
    print(f"Attendance updated and saved to CSV - {output_csv}")

def final_attendance(att_path):
    # for threshold in range(0,110,10):
    #     threshold = threshold/100
    #     output_csv_name = f"cse641_attendance_{threshold}.csv"
    #     write_combined_csv(att_path, output_csv_name, threshold)
    #     # break
    
    threshold = 0.5
    output_csv_name = f"cse641_attendance_{threshold}.csv"
    write_combined_csv(att_path, output_csv_name, threshold, label="pred", use_thresh=False)


    

if __name__=="__main__":
    # att_path = r"E:\code\DL_ClassFR\RECORDS\before_midsem_photo"
    att_path = r"E:\code\DL_ClassFR\RECORDS\final_eval"
    final_attendance(att_path)