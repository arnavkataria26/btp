import pandas as pd

# Load the master index
df_index = pd.read_csv("master_dataset_index.csv")

# Grab the file path of the first sample
sample_file_path = df_index.iloc[0]["File_Path"]
print(f"Loading sample file: {sample_file_path}")

# Read the raw CSV signal (without headers in raw file)
df_signal = pd.read_csv(sample_file_path, header=None)

# Display raw shape and first 5 rows
print("\nSignal DataFrame Shape:", df_signal.shape)
print("\nFirst 5 rows of raw telemetry signal:")
print(df_signal.head())