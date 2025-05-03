import os
import pandas as pd


_dir = os.path.dirname(os.path.abspath(__file__))
filename = "helpdesk.csv"
data_file_path = os.path.join(_dir, "quickstart\\data", filename)
print(f"Loading data from file: {data_file_path}")

# Load the CSV file into a DataFrame
df = pd.read_csv(data_file_path)
print(f"Data loaded successfully. Shape: {df.shape}")
print(f"Data columns: {df.columns.tolist()}")
print(f"Data types: {df.dtypes}")
print(f"Data head: {df.head()}")

# Convert CompleteTimestamp to datetime
df['CompleteTimestamp'] = pd.to_datetime(df['CompleteTimestamp'])

# Sort by CaseID and CompleteTimestamp
df = df.sort_values(by=['CaseID', 'CompleteTimestamp'])

# Create Pre-processor  column
df['PreActivityID'] = df.groupby('CaseID')['ActivityID'].shift(1)

# Fill NaN values with 0 and ensure LaggedActivityID is of integer type
df['PreActivityID'] = df['PreActivityID'].fillna(0).astype(int)

print(f"Data after processing: {df.head()}")

# Group by ActivityID and save each group to a different CSV
#for activity_id, group in df.groupby('ActivityID'):
#    filename = f'activity_{activity_id}.csv'
#    filename = os.path.join(_dir, "quickstart\\data\\IoT", filename)
#    group.to_csv(filename, index=False)

print("CSV files saved for each ActivityID.")

# Get the path to the simulated data file
_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(os.path.join(_dir, "quickstart\\data"), "IoT")
data_files = [os.path.join(data_file_path, f) for f in os.listdir(data_file_path) if f.endswith('.csv')]

print(f"###################### Loading data from {data_files}")