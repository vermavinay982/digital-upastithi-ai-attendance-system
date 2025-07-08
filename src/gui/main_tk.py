import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
import shutil

# Function to load JSON data
def load_json():
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, "r") as f:
            return json.load(f), file_path
    return None, None

def show_attendance(main_window):
    try:
        with open("attendance.txt", "r") as f:
            records = f.readlines()
        attendance_window = tk.Toplevel(main_window)
        attendance_window.title("Attendance Records")
        attendance_window.geometry("300x400")
        ttk.Label(attendance_window, text="Attendance Records", font=("Arial", 14)).pack(pady=10)
        for record in records:
            ttk.Label(attendance_window, text=record.strip(), font=("Arial", 12)).pack(anchor="w", padx=20)
    except FileNotFoundError:
        messagebox.showerror("Error", "Attendance file not found!")
        
# Function to open main application window
def open_main_window(json_data):
    main_window = tk.Toplevel(root)
    main_window.title("Digital Upasthiti - Viewer")
    main_window.geometry("600x800")

    def save_attendance():
        unique_labels = set()
        op_folder = "results"

        for img_path, dropdown in dropdown_vars.items():
            # correctness_vars: it makes us to select only the correct oness. 
            # which we want to keep in records
            if correctness_vars[img_path].get():  # Only save if marked correct
                # from the text box
                if checkbox_vars[img_path].get():
                    label = textbox_vars[img_path].get().strip().capitalize()
                else:
                    # from the drop down
                    label = dropdown.get()

                if label:
                    unique_labels.add(label)
                    #TODO: where the non selected ones will go?
                    data_path = os.path.join(op_folder, label)
                    os.makedirs(data_path, exist_ok=True)
                    new_path = os.path.join(data_path, os.path.basename(img_path))
                    shutil.copy(img_path, new_path)

        attendance_file = os.path.abspath("attendance.txt")
        with open(attendance_file, "w") as f:
            f.write("\n".join(unique_labels))
        messagebox.showinfo("Success", f"Saved records successfully!\nPath: {attendance_file}")



    frame = ttk.Frame(main_window)
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    save_button = ttk.Button(main_window, text="Save Attendance", command=save_attendance)
    save_button.pack(pady=5)
    show_button = ttk.Button(main_window, text="Show Attendance", command=lambda: show_attendance(root))
    show_button.pack(pady=5)

    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    dropdown_vars = {}
    checkbox_vars = {}
    textbox_vars = {}
    correctness_vars = {}
    dropdowns = {}
    textbox_entries = {}

    for img_path, results in json_data.items():
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).resize((50, 50), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image)
        img_frame = ttk.Frame(scrollable_frame, padding=10)
        img_frame.pack(fill=tk.X, padx=5, pady=5)
        img_label = ttk.Label(img_frame, image=img_tk)
        img_label.image = img_tk
        img_label.pack(side=tk.LEFT)
        controls_frame = ttk.Frame(img_frame)
        controls_frame.pack(side=tk.LEFT, padx=10)

        choices = [result["label"].capitalize() for result in results]
        choices += ["None"]

        if results[0]['score'] < 0.5:
            best_match = results[0]["label"].capitalize()
        else:
            best_match = choices[-1] # None

        selected_label = tk.StringVar(value=best_match)
        dropdown_vars[img_path] = selected_label
        dropdown = ttk.Combobox(controls_frame, textvariable=selected_label, values=choices, state="readonly")
        dropdown.pack()
        dropdowns[img_path] = dropdown
        
        checkbox_var = tk.BooleanVar()
        textbox_var = tk.StringVar()
        checkbox_vars[img_path] = checkbox_var
        textbox_vars[img_path] = textbox_var
        correctness_var = tk.BooleanVar()
        correctness_vars[img_path] = correctness_var

        def toggle_textbox(img_path=img_path):
            if checkbox_vars[img_path].get():
                textbox_entries[img_path].pack()
                dropdowns[img_path].config(state="disabled")
            else:
                textbox_entries[img_path].pack_forget()
                dropdowns[img_path].config(state="readonly")
        
        checkbox = ttk.Checkbutton(controls_frame, text="Other", variable=checkbox_var, command=toggle_textbox)
        checkbox.pack()
        correctness_checkbox = ttk.Checkbutton(controls_frame, text="Correct", variable=correctness_var)
        correctness_checkbox.pack()
        textbox_entry = ttk.Entry(controls_frame, textvariable=textbox_var)
        textbox_entries[img_path] = textbox_entry

        info_frame = ttk.Frame(img_frame)
        info_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(info_frame, text=f"Closest Match: {best_match} ({results[0]['score']:.2f})", font=("Arial", 10)).pack(anchor="w")
        
        ttk.Label(info_frame, text="Similar Matches", font=("Arial", 10)).pack(anchor="w")
        for i, result in enumerate(results[1:5]):
            ttk.Label(info_frame, text=f"{i+1}. {result['label'].capitalize()} ({result['score']:.2f})", font=("Arial", 10)).pack(anchor="w")

    main_window.mainloop()

# Create root window
root = tk.Tk()
root.title("Digital Upasthiti - Menu")
root.geometry("300x200")

def load_and_open():
    json_data, _ = load_json()
    if json_data:
        open_main_window(json_data)

ttk.Button(root, text="Load JSON", command=load_and_open).pack(pady=10)
ttk.Button(root, text="Show Attendance", command=lambda: show_attendance(root)).pack(pady=10)

root.mainloop()