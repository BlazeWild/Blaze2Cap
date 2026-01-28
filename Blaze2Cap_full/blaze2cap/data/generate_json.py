import os
import json

# ================= CONFIGURATION =================
# 1. Get the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define paths relative to the script
# Point to dataset/Totalcapture_numpy_preprocessed (one level up from data/)
dataset_root = os.path.join(script_dir, "..", "dataset", "Totalcapture_numpy_preprocessed")

# 3. Output file (saved in dataset/Totalcapture_numpy_preprocessed)
output_dir = dataset_root
os.makedirs(output_dir, exist_ok=True)
output_json = os.path.join(output_dir, "dataset_map.json")

# 4. Split Logic
TRAIN_SUBJECTS = ['S1', 'S2', 'S3']
TEST_SUBJECTS = ['S1', 'S2', 'S3', 'S4', 'S5']
# =================================================

data_list = []

blaze_root = os.path.join(dataset_root, 'blazepose_synced')
gt_root = os.path.join(dataset_root, 'gt_localized_21')

print(f"Scanning dataset at: {os.path.relpath(dataset_root, script_dir)}")

if not os.path.exists(blaze_root):
    print(f"ERROR: Could not find folder: {blaze_root}")
    print("Make sure you run this script from the project root (Blaze2Cap).")
    exit()

# Walk through BlazePose folder
for root, dirs, files in os.walk(blaze_root):
    for filename in files:
        if filename.endswith('.npy') and filename.startswith("blazepose_"):
            
            # --- 1. Identify Subject (S1, S2, etc.) ---
            path_parts = root.split(os.sep)
            subject = None
            for part in path_parts:
                if part.startswith("S") and part[1:].isdigit() and len(part) < 4: 
                    subject = part
                    break
            
            if subject is None:
                continue

            # --- 2. Match GT File ---
            # Replace "blazepose_" with "gtl_" for localized GT files
            suffix = filename.replace("blazepose_", "")
            gt_filename = "gtl_" + suffix
            
            # Get the internal folder structure (e.g., "S1/acting1")
            rel_dir = os.path.relpath(root, blaze_root)
            
            source_path = os.path.join(root, filename)
            target_path = os.path.join(gt_root, rel_dir, gt_filename)
            
            if not os.path.exists(target_path):
                print(f"Warning: Missing GT for {filename}")
                continue

            # --- 3. Determine Splits ---
            is_train = subject in TRAIN_SUBJECTS
            is_test = subject in TEST_SUBJECTS
            
            # --- 4. Store PORTABLE Relative Paths ---
            # These paths will look like "blazepose_synced/S1/acting1/..." 
            # regardless of where the project is stored on the disk.
            entry = {
                "source": os.path.relpath(source_path, dataset_root).replace("\\", "/"),
                "target": os.path.relpath(target_path, dataset_root).replace("\\", "/"),
                "subject": subject,
                "split_train": is_train,
                "split_test": is_test
            }
            data_list.append(entry)

# Save to JSON
with open(output_json, 'w') as f:
    json.dump(data_list, f, indent=4)

print(f"Success! Saved {len(data_list)} pairs.")
print(f"File location: {os.path.relpath(output_json, script_dir)}")