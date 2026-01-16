import pandas as pd
import glob
import os

def merge_csvs():
    # Pattern to match all parts
    input_pattern = "data/02_transcribed/transcripts_whisperx_part_*.csv"
    output_file = "data/02_transcribed/transcripts_whisperx_FINAL.csv"
    
    print(f"Looking for files: {input_pattern}")
    all_files = glob.glob(input_pattern)
    
    if not all_files:
        print("No part files found! Did the jobs finish?")
        return

    print(f"Found {len(all_files)} files. Merging...")
    
    # Read and concatenate
    df_list = []
    for filename in sorted(all_files):
        print(f"  - Reading {os.path.basename(filename)}")
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not df_list:
        print("No valid data found.")
        return

    final_df = pd.concat(df_list, ignore_index=True)
    
    # Sort just in case (optional)
    if 'Participant_ID' in final_df.columns:
        final_df = final_df.sort_values(by=['Participant_ID', 'start_time'])
    
    print(f"Total merged records: {len(final_df)}")
    
    final_df.to_csv(output_file, index=False)
    print(f"Saved merged result to: {output_file}")

if __name__ == "__main__":
    merge_csvs()
