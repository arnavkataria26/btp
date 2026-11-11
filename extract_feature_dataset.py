from pathlib import Path
import numpy as np
import pandas as pd

# Load human-readable catalog
df_catalog = pd.read_csv("human_readable_dataset_catalog.csv")

extracted_rows = []

print(f"Processing {len(df_catalog)} telemetry files...")

for idx, row in df_catalog.iterrows():
  file_path = row["Full_File_Path"]

  if (idx + 1) % 100 == 0 or (idx + 1) == len(df_catalog):
    print(f"Progress: {idx + 1}/{len(df_catalog)} files processed")

  try:
    # Read CSV
    df_signal = pd.read_csv(file_path, header=0)

    # 1. Drop trailing empty columns caused by extra commas
    df_signal = df_signal.loc[:, ~df_signal.columns.str.contains("^Unnamed")]
    df_signal = df_signal.dropna(how="all", axis=1)

    # 2. Extract sensor channels only (ignoring time column at index 0)
    df_channels = df_signal.iloc[:, 1:]

    # 3. Coerce values to numeric to handle any formatting artifacts
    df_channels = df_channels.apply(pd.to_numeric, errors="coerce")

    # 4. Calculate RMS for valid physical channels
    channel_rms = np.sqrt(np.nanmean(df_channels.values**2, axis=0))

    record = {
        "Modality": row["Modality"],
        "Motor_ID": row["Motor_ID"],
        "Speed_Percent": row["Speed_Percent"],
        "Fault_Type": row["Fault_Type"],
        "Severity_Level": row["Severity_Level"],
        "Active_Channels": len(channel_rms),
    }

    # Add RMS features for actual physical channels present in the file
    for ch_idx, rms_val in enumerate(channel_rms):
      record[f"Ch_{ch_idx}_RMS"] = round(rms_val, 5)

    extracted_rows.append(record)

  except Exception as e:
    print(f"Error processing {file_path}: {e}")
    continue

# Create final DataFrame
df_features = pd.DataFrame(extracted_rows)

# Fill non-applicable channel columns for files with fewer sensors with 0.0 or leave structured
df_features.to_csv("unified_fault_features.csv", index=False)

print("\n=== Corrected Dataset Preview ===")
print(df_features.head())
print(f"\nDataset shape: {df_features.shape}")
print("Saved clean feature matrix to unified_fault_features.csv")