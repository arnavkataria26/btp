from pathlib import Path
import re
import pandas as pd

dataset_root = Path("Dataset/Dataset")
catalog_records = []


def parse_fault_details(fault_string):
  """Splits labels like 'bearing pump 3' into ('Bearing Pump', 3)

  and 'healthy 1' into ('Healthy', 0).
  """
  match = re.search(r"^(.*?)\s*(\d+)?$", fault_string.strip())
  if match:
    fault_type = match.group(1).strip().title()
    severity = int(match.group(2)) if match.group(2) else 0

    if "Healthy" in fault_type:
      severity = 0  # Normal operation baseline

    return fault_type, severity
  return fault_string.title(), 0


for csv_file in dataset_root.rglob("*.csv"):
  parts = csv_file.relative_to(dataset_root).parts

  if len(parts) >= 4:
    modality = parts[0].title()  # Electric or Vibration
    motor = parts[1]  # Motor-2 or Motor-4
    speed_percent = int(parts[2])  # 70, 75, 100 (% of rated speed)
    raw_fault = parts[3]

    fault_type, severity_level = parse_fault_details(raw_fault)

    catalog_records.append({
        "Modality": modality,
        "Motor_ID": motor,
        "Speed_Percent": speed_percent,
        "Fault_Type": fault_type,
        "Severity_Level": severity_level,
        "File_Name": csv_file.name,
        "Full_File_Path": str(csv_file),
    })

# Save to a structured DataFrame
df_catalog = pd.DataFrame(catalog_records)
df_catalog.to_csv("human_readable_dataset_catalog.csv", index=False)

print("=== Cleaned Human-Readable Dataset Catalog ===")
print(
    df_catalog[[
        "Modality",
        "Motor_ID",
        "Speed_Percent",
        "Fault_Type",
        "Severity_Level",
    ]].head(15)
)
print(f"\nSuccessfully cataloged {len(df_catalog)} files.")