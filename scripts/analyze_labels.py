import pandas as pd
import sys

def analyze_labels(file_path):
    print(f"Analyzing {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"\nTotal Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing Values:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found.")

    # Check ID coverage
    ids = df['Participant'].unique()
    print(f"\nUnique Participants: {len(ids)}")
    print(f"ID Range: {min(ids)} - {max(ids)}")
    
    # Check Splits
    if 'split' in df.columns:
        print("\nSplit Distribution:")
        print(df['split'].value_counts())
    
    # Check Label Distribution
    if 'Depression_label' in df.columns:
        print("\nDepression Label Distribution:")
        print(df['Depression_label'].value_counts())
        
    if 'Depression_severity' in df.columns:
        print("\nDepression Severity Stats:")
        print(df['Depression_severity'].describe())

if __name__ == "__main__":
    analyze_labels(sys.argv[1])
