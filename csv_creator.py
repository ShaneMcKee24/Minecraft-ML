import os
import csv

DATASET_DIR = "Data_Sources"   # your root folder
CSV_PATH = "labels.csv"   # output

rows = []
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(root, file)

            # Extract label from folder name
            label = os.path.basename(root)

            rows.append([image_path, label])

# Write CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label"])
    writer.writerows(rows)

print("CSV created:", CSV_PATH)
