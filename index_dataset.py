from pathlib import Path
import pandas as pd

# Point directly to the inner folder to strip the extra 'Dataset/' prefix
dataset_root = Path("Dataset/Dataset")

master_records = []

for csv_file in dataset_root.rglob("*.csv"):
  parts = csv_file.relative_to(dataset_root).parts

  if len(parts) >= 4:
    modality = parts[0]  # Vibration or Electric
    motor = parts[1]  # Motor-4 or Motor-2
    speed = parts[2]  # Speed (70, 75, 100, etc.)
    fault_severity = parts[3]  # Fault type and severity

    master_records.append({
        "Modality": modality,
        "Motor": motor,
        "Speed": speed,
        "Fault_Condition": fault_severity,
        "File_Path": str(csv_file),
    })

df_metadata = pd.DataFrame(master_records)

# Save the corrected index
df_metadata.to_csv("master_dataset_index.csv", index=False)

print("=== Corrected Dataset Index Table ===")
print(df_metadata.head(10))
print(f"\nTotal files indexed: {len(df_metadata)}")