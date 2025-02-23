import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg.txt"):
            new_name = filename.replace(".jpg.txt", ".txt")  # Keep .txt, remove .jpg
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

# Set the directory path
folder_path = r"C:\Users\garg1\OneDrive\Desktop\100K\dataset\test\labels"
rename_files(folder_path)